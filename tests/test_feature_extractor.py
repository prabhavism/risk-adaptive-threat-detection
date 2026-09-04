import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.config import IngestConfig
from ingest.flow_builder import FlowBuilder
from ingest.feature_extractor import FeatureExtractor, ALL_OUTPUT_COLUMNS
from ingest.schemas import Packet


def make_packet(t, src_ip="10.0.0.1", dst_ip="10.0.0.2", src_port=50000,
                 dst_port=443, protocol="TCP", length=100):
    return Packet(timestamp=t, src_ip=src_ip, dst_ip=dst_ip, src_port=src_port,
                  dst_port=dst_port, protocol=protocol, packet_length=length)


def build_flow(packets):
    b = FlowBuilder(IngestConfig())
    for p in packets:
        b.add_packet(p)
    flows = b.flush()
    return b, flows[0]


def test_basic_counters_and_rates():
    packets = [make_packet(0.0, length=100), make_packet(1.0, length=200)]
    b, flow = build_flow(packets)
    row = FeatureExtractor(b).extract(flow)
    assert row["packet_count"] == 2
    assert row["byte_count"] == 300
    assert row["duration"] == 1.0
    assert row["packet_rate"] == 2.0
    assert row["avg_packet_size"] == 150.0


def test_byte_ratio_reflects_asymmetry():
    packets = [
        make_packet(0.0, src_ip="10.0.0.1", dst_ip="10.0.0.2", src_port=1, dst_port=2, length=1000),
        make_packet(0.1, src_ip="10.0.0.2", dst_ip="10.0.0.1", src_port=2, dst_port=1, length=50),
    ]
    b, flow = build_flow(packets)
    row = FeatureExtractor(b).extract(flow)
    assert row["byte_ratio"] == 20.0  # 1000 forward / 50 reverse


def test_entropy_zero_for_uniform_sizes():
    packets = [make_packet(float(i), length=100) for i in range(5)]
    b, flow = build_flow(packets)
    row = FeatureExtractor(b).extract(flow)
    assert row["entropy"] == 0.0


def test_entropy_positive_for_varied_sizes():
    packets = [make_packet(float(i), length=100 + i * 37) for i in range(5)]
    b, flow = build_flow(packets)
    row = FeatureExtractor(b).extract(flow)
    assert row["entropy"] > 0.0


def test_periodicity_high_for_regular_timing():
    packets = [make_packet(float(i) * 2.0) for i in range(6)]  # exactly 2s apart
    b, flow = build_flow(packets)
    row = FeatureExtractor(b).extract(flow)
    assert row["periodicity_score"] > 0.95


def test_periodicity_low_for_irregular_timing():
    times = [0.0, 0.1, 5.0, 5.2, 30.0, 30.1]
    packets = [make_packet(t) for t in times]
    b, flow = build_flow(packets)
    row = FeatureExtractor(b).extract(flow)
    assert row["periodicity_score"] < 0.5


def test_output_columns_match_locked_schema():
    packets = [make_packet(0.0)]
    b, flow = build_flow(packets)
    row = FeatureExtractor(b).extract(flow)
    assert set(row.keys()) == set(ALL_OUTPUT_COLUMNS)
