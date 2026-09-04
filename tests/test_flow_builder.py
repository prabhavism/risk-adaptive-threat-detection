"""
Pure-Python unit tests for ingest/flow_builder.py -- no scapy, no
trained models required. Run with: pytest tests/test_flow_builder.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.config import IngestConfig
from ingest.flow_builder import FlowBuilder, normalize_flow_key
from ingest.schemas import Packet


def make_packet(t, src_ip="10.0.0.1", dst_ip="10.0.0.2", src_port=50000,
                 dst_port=443, protocol="TCP", length=100, tcp_flags=None):
    return Packet(timestamp=t, src_ip=src_ip, dst_ip=dst_ip, src_port=src_port,
                  dst_port=dst_port, protocol=protocol, packet_length=length,
                  tcp_flags=tcp_flags)


def test_same_5_tuple_same_flow():
    b = FlowBuilder(IngestConfig())
    b.add_packet(make_packet(0.0))
    b.add_packet(make_packet(0.1))
    assert len(b.active_flows) == 1


def test_different_5_tuple_different_flow():
    b = FlowBuilder(IngestConfig())
    b.add_packet(make_packet(0.0, dst_port=443))
    b.add_packet(make_packet(0.1, dst_port=80))
    assert len(b.active_flows) == 2


def test_reverse_direction_same_bidirectional_flow():
    b = FlowBuilder(IngestConfig())
    b.add_packet(make_packet(0.0, src_ip="10.0.0.1", dst_ip="10.0.0.2",
                              src_port=50000, dst_port=443))
    b.add_packet(make_packet(0.1, src_ip="10.0.0.2", dst_ip="10.0.0.1",
                              src_port=443, dst_port=50000))
    assert len(b.active_flows) == 1
    flow = next(iter(b.active_flows.values()))
    assert flow.forward_packet_count == 1
    assert flow.reverse_packet_count == 1
    assert flow.initiator_ip == "10.0.0.1"
    assert flow.responder_ip == "10.0.0.2"


def test_icmp_flow_key_has_no_ports():
    key, _ = normalize_flow_key("10.0.0.1", "10.0.0.2", None, None, "ICMP")
    assert key[1] is None and key[3] is None


def test_tcp_fin_finalizes_flow():
    b = FlowBuilder(IngestConfig())
    b.add_packet(make_packet(0.0, tcp_flags="S"))
    finalized = b.add_packet(make_packet(0.2, tcp_flags="FA"))
    assert len(finalized) == 1
    assert finalized[0].finalize_reason == "tcp_close"
    assert len(b.active_flows) == 0


def test_inactivity_timeout_finalizes_flow():
    config = IngestConfig(general_flow_timeout=5.0, udp_flow_timeout=5.0)
    b = FlowBuilder(config)
    b.add_packet(make_packet(0.0, protocol="UDP", tcp_flags=None))
    expired = b.check_timeouts(now=10.0)  # 10s later, > 5s timeout
    assert len(expired) == 1
    assert expired[0].finalize_reason == "timeout"


def test_flush_finalizes_everything():
    b = FlowBuilder(IngestConfig())
    b.add_packet(make_packet(0.0, dst_port=443))
    b.add_packet(make_packet(0.0, dst_port=80))
    remaining = b.flush()
    assert len(remaining) == 2
    assert len(b.active_flows) == 0


def test_capacity_eviction_bounds_active_flows():
    config = IngestConfig(max_active_flows=3)
    b = FlowBuilder(config)
    for i in range(10):
        b.add_packet(make_packet(float(i), dst_port=1000 + i))
    assert len(b.active_flows) <= 3
    assert b.capacity_evictions == 7


def test_source_stats_track_fanout():
    b = FlowBuilder(IngestConfig())
    for i in range(5):
        b.add_packet(make_packet(float(i), dst_ip=f"10.0.0.{i+10}", dst_port=443))
    stats = b.source_stats("10.0.0.1", now=5.0)
    assert stats["dest_fanout"] == 5
    assert stats["flow_count"] == 5
