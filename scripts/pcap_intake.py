"""
Continuous PCAP intake for a client-server handoff.

Two transport modes, same processing underneath -- pick whichever
matches how the other person's server actually delivers files:

  --watch-dir DIR
      Polls DIR for new *.pcap files (e.g. a shared folder, a mounted
      network share, or a directory the other server SFTPs/copies
      into). Good default if both sides can share a filesystem path.

          python scripts/pcap_intake.py --watch-dir incoming_pcaps

  --serve PORT
      Runs a plain HTTP server. The other server POSTs the raw pcap
      bytes to it directly (no multipart/form-data needed):

          curl -X POST --data-binary @capture.pcap http://HOST:PORT/pcap

      or from their Python code:

          import requests
          requests.post("http://HOST:PORT/pcap",
                         data=open("capture.pcap", "rb").read())

Either mode feeds packets into ONE long-lived DetectionPipeline
instance, so per-source stats (dest_fanout/port_fanout/flow_count) and
Heavy DL's per-host sequence history correctly carry over from one
file to the next -- exactly as if this were one continuous capture,
not independent replays. Deliberately does NOT call pipeline.flush()
after each file (that would force-finalize every in-flight flow just
because this particular file ended, artificially cutting off any
connection whose packets are split across a file-rotation boundary).
Flows still close on their own via the normal idle-timeout logic
(ingest.config.IngestConfig timeouts), checked here on a wall-clock
timer between files. flush() only runs at shutdown (Ctrl+C), to
finalize whatever's still open.

If it turns out each pcap IS a fully independent, self-contained
capture window (confirm this with whoever owns the server side), flip
FLUSH_AFTER_EACH_FILE to True below -- one-line change.
"""
import argparse
import http.server
import json
import logging
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.config import IngestConfig
from ingest.pcap_reader import read_pcap
from ingest.pipeline import DetectionPipeline
from ml_dl.alerts import build_alert
from ml_dl.config import ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pcap_intake")

# See docstring above -- set True only once you've confirmed each
# pcap is a self-contained capture window with no cross-file flows.
FLUSH_AFTER_EACH_FILE = False

IDLE_CHECK_INTERVAL_SEC = 30  # how often to sweep for idle-timed-out flows


def _result_to_record(result: dict) -> dict:
    flow = result["flow"]
    pred = result["prediction"]
    return {
        "flow_id": result["flow_id"],
        "timestamp": result["timestamp"],
        "source_ip": flow["src_ip"],
        "destination_ip": flow["dst_ip"],
        "source_port": flow["src_port"],
        "destination_port": flow["dst_port"],
        "protocol": flow["protocol"],
        "xgboost_class": pred["ml_verdict"],
        "xgboost_confidence": pred["ml_confidence"],
        "model_used": pred["model_used"],
        "dl_class": pred["dl_verdict"],
        "dl_confidence": pred["dl_confidence"],
        "shap_evidence": pred["shap_evidence"],
        "alert": build_alert(result["features"], pred, received_at=result["timestamp"])
                 if pred["dl_verdict"] != "benign" else None,
    }


class PcapIntake:
    """Owns the one long-lived pipeline. process_file() is the shared
    entry point for both the watch-folder loop and the HTTP handler."""

    def __init__(self, out_path: Path, config: IngestConfig = None):
        self.pipeline = DetectionPipeline(config or IngestConfig())
        self.out_path = out_path
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._out_f = open(self.out_path, "a")
        self._last_idle_check = time.time()
        self.n_results = 0
        self.n_alerts = 0

    def _handle_result(self, result: dict):
        record = _result_to_record(result)
        self._out_f.write(json.dumps(record) + "\n")
        self._out_f.flush()
        self.n_results += 1
        tag = "ALERT" if record["alert"] is not None else "     "
        if record["alert"] is not None:
            self.n_alerts += 1
        logger.info(
            "[%s] %s  %s -> %s:%s  xgb=%s(%.2f)  dl=%s(%.2f)  model=%s",
            tag, record["flow_id"], record["source_ip"], record["destination_ip"],
            record["destination_port"], record["xgboost_class"], record["xgboost_confidence"],
            record["dl_class"], record["dl_confidence"], record["model_used"],
        )

    def process_file(self, pcap_path: str):
        logger.info("Processing %s", pcap_path)
        n_packets = 0
        for packet in read_pcap(pcap_path, speed=0):  # speed=0: no real-time pacing
            n_packets += 1
            for result in self.pipeline.process_packet(packet):
                self._handle_result(result)

        if FLUSH_AFTER_EACH_FILE:
            for result in self.pipeline.flush():
                self._handle_result(result)
        else:
            self._maybe_check_idle()

        logger.info("Finished %s (%d packets)", pcap_path, n_packets)

    def _maybe_check_idle(self):
        now = time.time()
        if now - self._last_idle_check >= IDLE_CHECK_INTERVAL_SEC:
            for result in self.pipeline.check_idle_timeouts(now):
                self._handle_result(result)
            self._last_idle_check = now

    def shutdown(self):
        logger.info("Shutting down -- flushing remaining active flows")
        for result in self.pipeline.flush():
            self._handle_result(result)
        self._out_f.close()
        stats = self.pipeline.stats()
        logger.info("Totals: flows=%d predictions=%d alerts=%d",
                     stats["total_flows_finalized"], self.n_results, self.n_alerts)


# ── Mode 1: watch a folder ───────────────────────────────────────────────────

def run_watch_dir(intake: PcapIntake, watch_dir: Path, poll_interval: float = 2.0):
    processed_dir = watch_dir / "_processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Watching %s for new .pcap files (Ctrl+C to stop)", watch_dir)

    seen = set()
    try:
        while True:
            for f in sorted(watch_dir.glob("*.pcap")):
                if f in seen:
                    continue
                seen.add(f)
                try:
                    intake.process_file(str(f))
                    shutil.move(str(f), str(processed_dir / f.name))
                except Exception:
                    logger.exception("Failed processing %s -- leaving it in place", f)
            intake._maybe_check_idle()
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        pass
    finally:
        intake.shutdown()


# ── Mode 2: minimal HTTP receiver ────────────────────────────────────────────

def run_serve(intake: PcapIntake, port: int, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # quiet; we log via `logger` in process_file instead

        def do_POST(self):
            if self.path != "/pcap":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)

            fname = f"capture_{int(time.time() * 1000)}.pcap"
            fpath = save_dir / fname
            fpath.write_bytes(body)

            try:
                intake.process_file(str(fpath))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok\n")
            except Exception as e:
                logger.exception("Failed processing uploaded pcap")
                self.send_response(500)
                self.end_headers()
                self.wfile.write(f"error: {e}\n".encode())

    server = http.server.HTTPServer(("0.0.0.0", port), Handler)
    logger.info("Listening on 0.0.0.0:%d -- POST pcap bytes to /pcap (Ctrl+C to stop)", port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        intake.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch-dir", type=str, default=None,
                         help="folder to poll for new .pcap files")
    parser.add_argument("--serve", type=int, default=None,
                         help="port to run an HTTP receiver on instead")
    parser.add_argument("--save-dir", type=str, default="incoming_pcaps",
                         help="where uploaded pcaps land in --serve mode")
    parser.add_argument("--out", type=str, default=None,
                         help="JSONL output path (default: reports/pcap_results.jsonl)")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else (ROOT / "reports" / "pcap_results.jsonl")
    intake = PcapIntake(out_path)

    if args.watch_dir:
        run_watch_dir(intake, Path(args.watch_dir))
    elif args.serve:
        run_serve(intake, args.serve, Path(args.save_dir))
    else:
        parser.error("provide --watch-dir DIR or --serve PORT")


if __name__ == "__main__":
    main()
