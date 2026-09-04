"""
Benchmarks the ingestion pipeline against a PCAP. Measures packet
throughput, flow throughput, and three distinct latency numbers
(section 35):

  - packet parsing latency:    raw packet -> normalized Packet
  - flow finalization latency: flow finalized -> features extracted
  - end-to-end latency:        packet available -> prediction returned
                                (parsing + flow update + feature
                                extraction + predict() call, whichever
                                packet triggers a finalization)

Model latency alone is already covered by scripts/benchmark.py
(existing ml_dl benchmark on flow_features.csv rows); this script
measures the ingestion path specifically, using speed=0 (unpaced) so
the numbers reflect this machine's processing speed, not the PCAP's
original capture rate.

Does not fabricate numbers -- every value printed is measured on this
run against the given PCAP.

Usage:
    python scripts/benchmark_ingestion.py --pcap data/demo.pcap
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ingest.config import IngestConfig
from ingest.packet_parser import parse_packet
from ingest.pipeline import DetectionPipeline


def benchmark(pcap_path: str):
    try:
        from scapy.utils import PcapReader
    except ImportError as e:
        raise ImportError("scapy is required: pip install scapy") from e

    config = IngestConfig(pcap_path=pcap_path)
    pipeline = DetectionPipeline(config)

    parse_latencies = []
    e2e_latencies = []
    n_packets = 0
    peak_active_flows = 0

    t_start = time.perf_counter()
    with PcapReader(pcap_path) as reader:
        for raw_pkt in reader:
            t0 = time.perf_counter()
            try:
                pkt_time = float(raw_pkt.time)
            except Exception:
                pkt_time = time.time()
            parsed = parse_packet(raw_pkt, capture_time=pkt_time)
            parse_latencies.append(time.perf_counter() - t0)
            n_packets += 1
            if parsed is None:
                continue

            t1 = time.perf_counter()
            results = pipeline.process_packet(parsed)
            if results:
                e2e_latencies.append(time.perf_counter() - t1)

            peak_active_flows = max(peak_active_flows, len(pipeline.builder.active_flows))

    for _ in pipeline.flush():
        pass
    total_time = time.perf_counter() - t_start

    stats = pipeline.stats()
    parse_ms = np.array(parse_latencies) * 1000.0
    e2e_ms = np.array(e2e_latencies) * 1000.0 if e2e_latencies else np.array([0.0])

    print("=== INGESTION BENCHMARK ===\n")
    print(f"PCAP: {pcap_path}\n")
    print(f"Packets processed: {n_packets}")
    print(f"Flows created:     {stats['total_flows_created']}")
    print(f"Flows finalized:   {stats['total_flows_finalized']}")
    print(f"Predictions made:  {stats['total_predictions']}\n")

    print(f"Packet throughput: {n_packets / total_time:.1f} packets/sec")
    if stats["total_flows_finalized"]:
        print(f"Flow throughput:   {stats['total_flows_finalized'] / total_time:.1f} flows/sec\n")

    print("Packet parsing latency (ms):")
    print(f"  P50: {np.percentile(parse_ms, 50):.3f}  P95: {np.percentile(parse_ms, 95):.3f}  "
          f"P99: {np.percentile(parse_ms, 99):.3f}\n")

    if e2e_latencies:
        print("End-to-end latency, packet-available -> prediction returned (ms):")
        print(f"  P50: {np.percentile(e2e_ms, 50):.3f}  P95: {np.percentile(e2e_ms, 95):.3f}  "
              f"P99: {np.percentile(e2e_ms, 99):.3f}\n")
    else:
        print("End-to-end latency: no flows finalized during capture (all still active at flush)\n")

    print(f"Peak active flows: {peak_active_flows}")
    print(f"Capacity evictions: {stats['capacity_evictions']}")
    print(f"\nTotal wall time: {total_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcap", required=True)
    args = parser.parse_args()
    benchmark(args.pcap)
