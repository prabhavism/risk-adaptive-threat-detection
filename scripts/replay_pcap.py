"""
Replays a PCAP file through the full ingestion + AI pipeline:

    PCAP -> packet parser -> flow builder -> feature extractor
         -> predict(flow) -> calibrated XGBoost -> routing
         -> Light or Heavy GRU -> SHAP evidence -> alert (if malicious)

Reads the PCAP incrementally (ingest.pcap_reader.read_pcap uses
scapy's PcapReader, which never loads the whole file into memory) and
writes one JSON object per finalized flow to a JSONL file as results
arrive, not batched at the end.

Usage:
    python scripts/replay_pcap.py --pcap data/demo.pcap
    python scripts/replay_pcap.py --pcap data/demo.pcap --speed 10
    python scripts/replay_pcap.py --pcap data/demo.pcap --speed 0 --out reports/pcap_results.jsonl

Read-only / passive: only reads the given PCAP file from disk and
writes local output files. Never opens a network connection -- see
ingest/README.md for the full passive-only audit.
"""
import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.config import IngestConfig
from ingest.pcap_reader import read_pcap
from ingest.pipeline import DetectionPipeline
from ml_dl.alerts import build_alert
from ml_dl.config import ROOT


def _result_to_record(result: dict) -> dict:
    """
    One JSONL record per finalized flow, with exactly the fields Part 4
    of the ingestion-integration brief asks for. `prediction` carries
    the untouched predict_interface.predict() output (locked contract,
    docs/interfaces.md); the top-level fields are a flattened
    convenience view of the same data for quick grepping/jq'ing.
    """
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


def _print_summary_line(record: dict):
    tag = "ALERT" if record["alert"] is not None else "     "
    print(f"[{tag}] {record['flow_id']}  {record['source_ip']} -> "
          f"{record['destination_ip']}:{record['destination_port'] or ''}  "
          f"xgb={record['xgboost_class']}({record['xgboost_confidence']:.2f})  "
          f"dl={record['dl_class']}({record['dl_confidence']:.2f})  "
          f"model={record['model_used']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcap", required=True)
    parser.add_argument("--speed", type=float, default=1.0,
                         help="1.0 = real-time, 10 = 10x, 0 = max speed")
    parser.add_argument("--flow-timeout", type=float, default=None,
                         help="override general flow timeout (seconds)")
    parser.add_argument("--out", type=str, default=None,
                         help="JSONL output path (default: reports/pcap_results.jsonl)")
    parser.add_argument("--quiet", action="store_true", help="suppress per-flow console lines")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    config = IngestConfig(pcap_path=args.pcap, replay_speed=args.speed)
    if args.flow_timeout is not None:
        config.general_flow_timeout = args.flow_timeout

    out_path = args.out or str(ROOT / "reports" / "pcap_results.jsonl")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    pipeline = DetectionPipeline(config)
    n_results = 0
    n_alerts = 0

    with open(out_path, "w") as out_f:
        def _handle(result):
            nonlocal n_results, n_alerts
            n_results += 1
            record = _result_to_record(result)
            out_f.write(json.dumps(record) + "\n")
            if record["alert"] is not None:
                n_alerts += 1
            if not args.quiet:
                _print_summary_line(record)

        for packet in read_pcap(args.pcap, speed=args.speed):
            for result in pipeline.process_packet(packet):
                _handle(result)

        for result in pipeline.flush():
            _handle(result)

    stats = pipeline.stats()
    print("\n=== Replay complete ===")
    print(f"Flows created:      {stats['total_flows_created']}")
    print(f"Flows finalized:    {stats['total_flows_finalized']}")
    print(f"Predictions:        {stats['total_predictions']}")
    print(f"Alerts (non-benign): {n_alerts}")
    print(f"Capacity evictions: {stats['capacity_evictions']}")
    print(f"JSONL output: {out_path}")


if __name__ == "__main__":
    main()
