"""
One-command demo of the full passive detection chain.

    python demo.py --pcap data/demo.pcap --speed 1
    python demo.py --interface eth0          # requires sudo, Linux

Prints the passive/no-decryption banner required for the demo, then
streams live detections as flows finalize. This is a thin wrapper
around scripts/replay_pcap.py and scripts/capture_live.py -- it adds
no detection logic of its own.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingest.config import IngestConfig
from ingest.pipeline import DetectionPipeline

BANNER = """\
========================================
 PASSIVE AI THREAT DETECTION
========================================

Mode: PASSIVE
Payload inspection: DISABLED
TLS decryption: DISABLED
"""


def _print_latest(result, n_packets, n_flows):
    pred = result["prediction"]
    print(f"\nPackets processed: {n_packets}")
    print(f"Flows finalized: {n_flows}\n")
    print("Latest Detection")
    print("----------------")
    print(f"Flow ID: {result['flow_id']}")
    print(f"Threat: {pred['dl_verdict'].upper()}")
    print(f"Confidence: {pred['dl_confidence'] * 100:.1f}%")
    print(f"Model: {pred['model_used'].upper()}")
    evidence = pred.get("shap_evidence", [])
    if evidence:
        print("\nEvidence:")
        for item in evidence:
            print(f"  {item['feature']}")


def run_pcap(pcap_path: str, speed: float):
    from ingest.pcap_reader import read_pcap

    print(BANNER)
    print(f"Input: {pcap_path}\n")

    pipeline = DetectionPipeline(IngestConfig(pcap_path=pcap_path, replay_speed=speed))
    n_packets = 0
    n_flows = 0
    for packet in read_pcap(pcap_path, speed=speed):
        n_packets += 1
        for result in pipeline.process_packet(packet):
            n_flows += 1
            _print_latest(result, n_packets, n_flows)
    for result in pipeline.flush():
        n_flows += 1
        _print_latest(result, n_packets, n_flows)


def run_live(interface: str):
    from ingest.live_capture import capture_live

    print(BANNER)
    print(f"Input: live interface {interface}\n")

    pipeline = DetectionPipeline(IngestConfig(interface=interface))
    counters = {"packets": 0, "flows": 0}

    def _on_packet(packet):
        counters["packets"] += 1
        for result in pipeline.process_packet(packet):
            counters["flows"] += 1
            _print_latest(result, counters["packets"], counters["flows"])

    try:
        capture_live(interface, _on_packet)
    except KeyboardInterrupt:
        for result in pipeline.flush():
            counters["flows"] += 1
            _print_latest(result, counters["packets"], counters["flows"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcap", type=str, default=None)
    parser.add_argument("--interface", type=str, default=None)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    if args.pcap:
        run_pcap(args.pcap, args.speed)
    elif args.interface:
        run_live(args.interface)
    else:
        parser.error("provide --pcap PATH or --interface NAME")


if __name__ == "__main__":
    main()
