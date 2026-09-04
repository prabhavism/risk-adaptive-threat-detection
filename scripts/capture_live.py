"""
Passive live capture through the full ingestion + AI pipeline.
Requires root/CAP_NET_RAW on Linux to open a raw socket.

Usage:
    sudo python scripts/capture_live.py --interface eth0
    sudo python scripts/capture_live.py --interface eth0 --flow-timeout 30

STRICTLY PASSIVE: this script only receives packets (scapy sniff()).
It never sends, injects, or probes anything -- see ingest/README.md.
Stop with Ctrl+C; remaining active flows are flushed and scored before exit.
"""
import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.config import IngestConfig
from ingest.live_capture import capture_live
from ingest.pipeline import DetectionPipeline

logger = logging.getLogger("capture_live")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", required=True)
    parser.add_argument("--flow-timeout", type=float, default=30.0)
    parser.add_argument("--filter", type=str, default=None, help="optional BPF filter, e.g. 'ip'")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = IngestConfig(interface=args.interface, general_flow_timeout=args.flow_timeout)
    pipeline = DetectionPipeline(config)

    stop_event = threading.Event()

    def _idle_timeout_sweeper():
        # Live capture has no "next packet" to piggyback timeout checks
        # on during quiet periods, so a background sweep uses
        # wall-clock time to expire genuinely idle flows.
        while not stop_event.is_set():
            time.sleep(5.0)
            for result in pipeline.check_idle_timeouts(time.time()):
                logger.info("Idle-timeout flow scored: %s -> %s",
                            result["flow_id"], result["prediction"]["dl_verdict"])

    sweeper = threading.Thread(target=_idle_timeout_sweeper, daemon=True)
    sweeper.start()

    def _on_packet(packet):
        for result in pipeline.process_packet(packet):
            logger.info(
                "Flow %s -> %s (confidence=%.2f, model=%s)",
                result["flow_id"], result["prediction"]["dl_verdict"],
                result["prediction"]["dl_confidence"], result["prediction"]["model_used"],
            )

    def _handle_sigint(sig, frame):
        logger.info("Stopping capture, flushing remaining flows...")
        stop_event.set()
        for result in pipeline.flush():
            logger.info("Flushed flow %s -> %s", result["flow_id"], result["prediction"]["dl_verdict"])
        logger.info("Final stats: %s", pipeline.stats())
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_sigint)

    logger.info("Listening on %s (Ctrl+C to stop)...", args.interface)
    capture_live(args.interface, _on_packet, bpf_filter=args.filter)


if __name__ == "__main__":
    main()
