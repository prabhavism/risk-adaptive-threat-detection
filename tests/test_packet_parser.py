"""
Tests packet_parser.py against synthetically constructed scapy packets
(no real PCAP needed). Requires scapy -- skipped entirely if it isn't
installed, since packet_parser.py is the only ingest/ module with a
hard scapy dependency (everything else works on the normalized Packet
dataclass and is tested without it in the other test files).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import scapy  # noqa: F401
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

import pytest

pytestmark = pytest.mark.skipif(not SCAPY_AVAILABLE, reason="scapy not installed")


def test_parse_tcp_packet():
    from scapy.layers.inet import IP, TCP
    from scapy.layers.l2 import Ether
    from ingest.packet_parser import parse_packet

    pkt = Ether() / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=50000, dport=443, flags="S")
    parsed = parse_packet(pkt, capture_time=123.0)

    assert parsed is not None
    assert parsed.protocol == "TCP"
    assert parsed.src_ip == "10.0.0.1"
    assert parsed.dst_ip == "10.0.0.2"
    assert parsed.src_port == 50000
    assert parsed.dst_port == 443
    assert "S" in parsed.tcp_flags


def test_parse_udp_packet():
    from scapy.layers.inet import IP, UDP
    from ingest.packet_parser import parse_packet

    pkt = IP(src="10.0.0.1", dst="8.8.8.8") / UDP(sport=51000, dport=53)
    parsed = parse_packet(pkt, capture_time=1.0)

    assert parsed is not None
    assert parsed.protocol == "UDP"


def test_parse_dns_query():
    from scapy.layers.inet import IP, UDP
    from scapy.layers.dns import DNS, DNSQR
    from ingest.packet_parser import parse_packet

    pkt = (IP(src="10.0.0.5", dst="8.8.8.8") / UDP(sport=51000, dport=53) /
           DNS(rd=1, qd=DNSQR(qname="example.com", qtype="A")))
    parsed = parse_packet(pkt, capture_time=1.0)

    assert parsed is not None
    assert parsed.dns_query_name == "example.com"
    assert parsed.dns_record_type == "A"


def test_parse_non_ip_packet_returns_none():
    from scapy.layers.l2 import Ether, ARP
    from ingest.packet_parser import parse_packet

    pkt = Ether() / ARP()
    parsed = parse_packet(pkt, capture_time=1.0)
    assert parsed is None


def test_malformed_packet_does_not_raise():
    from ingest.packet_parser import parse_packet

    class Broken:
        time = "not-a-number"

        def haslayer(self, _):
            raise RuntimeError("boom")

    # Must return None, never raise.
    assert parse_packet(Broken()) is None
