"""
Section 36: verifies active flow count and per-flow packet history
never grow past the configured bounds, no matter how many flows or
packets are pushed through.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.config import IngestConfig
from ingest.flow_builder import FlowBuilder
from ingest.schemas import Packet


def test_active_flows_bounded_under_flow_burst():
    config = IngestConfig(max_active_flows=100)
    b = FlowBuilder(config)
    for i in range(5000):
        pkt = Packet(timestamp=float(i), src_ip="10.0.0.1", dst_ip=f"192.0.2.{i % 255}",
                     src_port=40000 + i, dst_port=443, protocol="TCP", packet_length=64)
        b.add_packet(pkt)
        assert len(b.active_flows) <= config.max_active_flows
    assert b.capacity_evictions >= 5000 - config.max_active_flows


def test_per_flow_packet_history_bounded():
    config = IngestConfig(max_history_per_flow=50)
    b = FlowBuilder(config)
    for i in range(1000):
        pkt = Packet(timestamp=float(i) * 0.01, src_ip="10.0.0.1", dst_ip="10.0.0.2",
                     src_port=1, dst_port=2, protocol="TCP", packet_length=64)
        b.add_packet(pkt)
    flow = next(iter(b.active_flows.values()))
    assert len(flow.packet_sizes) <= config.max_history_per_flow
    assert len(flow.packet_timestamps) <= config.max_history_per_flow
    # But the *counters* still reflect the true totals, not just the
    # capped history -- history capping is an approximation for
    # IAT/entropy on very long flows, not a loss of the flow's real
    # packet/byte counts.
    assert flow.packet_count == 1000


def test_source_history_bounded_and_pruned():
    config = IngestConfig(source_history_window_sec=10.0, max_source_history_entries=200)
    b = FlowBuilder(config)
    for i in range(500):
        pkt = Packet(timestamp=float(i), src_ip="10.0.0.1", dst_ip=f"192.0.2.{i % 255}",
                     src_port=40000 + i, dst_port=443, protocol="TCP", packet_length=64)
        b.add_packet(pkt)
    # window is 10s; only the last ~10 flows should count as "recent"
    stats = b.source_stats("10.0.0.1", now=499.0)
    assert stats["flow_count"] <= 15
