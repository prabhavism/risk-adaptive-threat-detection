"""
Generates a tiny, deterministic, harmless PCAP fixture for demoing
PCAP -> packets -> flows -> features -> predict() without needing a
real capture (Part 13). Contains only synthetic, non-malicious traffic:

  - one TCP connection (SYN, SYN-ACK, ACK, small HTTP-like request/
    response, FIN, FIN-ACK)
  - one DNS query + response over UDP
  - one ICMP echo request + reply

Nothing here attacks anything: all addresses are RFC 5737/1918
documentation/private ranges (192.0.2.0/24, 198.51.100.0/24,
10.0.0.0/8), and every packet is synthetically constructed, not routed
anywhere.

Deliberately implemented with only Python's struct module (no scapy
dependency) so this fixture can be regenerated even in an environment
that only has the ingestion layer's *reading* dependency (scapy)
installed later, or none at all -- generation and consumption are
decoupled.

Usage:
    python scripts/generate_demo_pcap.py --out data/demo.pcap
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- classic (not pcapng) pcap file format -------------------------------
PCAP_MAGIC = 0xA1B2C3D4
PCAP_VERSION_MAJOR = 2
PCAP_VERSION_MINOR = 4
LINKTYPE_ETHERNET = 1


def _ip_checksum(data: bytes) -> int:
    if len(data) % 2:
        data += b"\x00"
    total = sum(struct.unpack(f"!{len(data) // 2}H", data))
    total = (total >> 16) + (total & 0xFFFF)
    total += total >> 16
    return (~total) & 0xFFFF


def _mac(s: str) -> bytes:
    return bytes(int(b, 16) for b in s.split(":"))


def _eth_header(src_mac: str, dst_mac: str, ethertype: int = 0x0800) -> bytes:
    return _mac(dst_mac) + _mac(src_mac) + struct.pack("!H", ethertype)


def _ipv4_header(src_ip: str, dst_ip: str, proto: int, payload_len: int, ident: int) -> bytes:
    version_ihl = (4 << 4) | 5
    total_len = 20 + payload_len
    flags_frag = 0x4000  # don't fragment
    ttl = 64
    header = struct.pack(
        "!BBHHHBBH4s4s",
        version_ihl, 0, total_len, ident, flags_frag, ttl, proto, 0,
        _ipv4_to_bytes(src_ip), _ipv4_to_bytes(dst_ip),
    )
    checksum = _ip_checksum(header)
    return header[:10] + struct.pack("!H", checksum) + header[12:]


def _ipv4_to_bytes(ip: str) -> bytes:
    return bytes(int(o) for o in ip.split("."))


def _tcp_segment(src_ip, dst_ip, sport, dport, seq, ack, flags, payload: bytes) -> bytes:
    offset_reserved = (5 << 4)
    window = 64240
    header = struct.pack(
        "!HHIIBBHHH",
        sport, dport, seq, ack, offset_reserved, flags, window, 0, 0,
    )
    pseudo = _ipv4_to_bytes(src_ip) + _ipv4_to_bytes(dst_ip) + struct.pack(
        "!BBH", 0, 6, len(header) + len(payload)
    )
    checksum = _ip_checksum(pseudo + header + payload)
    header = header[:16] + struct.pack("!H", checksum) + header[18:]
    return header + payload


def _udp_segment(src_ip, dst_ip, sport, dport, payload: bytes) -> bytes:
    length = 8 + len(payload)
    header = struct.pack("!HHHH", sport, dport, length, 0)
    pseudo = _ipv4_to_bytes(src_ip) + _ipv4_to_bytes(dst_ip) + struct.pack("!BBH", 0, 17, length)
    checksum = _ip_checksum(pseudo + header + payload)
    if checksum == 0:
        checksum = 0xFFFF
    header = header[:6] + struct.pack("!H", checksum) + header[8:]
    return header + payload


def _icmp_echo(icmp_type: int, ident: int, seq: int, payload: bytes) -> bytes:
    header = struct.pack("!BBHHH", icmp_type, 0, 0, ident, seq)
    checksum = _ip_checksum(header + payload)
    header = struct.pack("!BBHHH", icmp_type, 0, checksum, ident, seq)
    return header + payload


def _dns_query(qname: str, qtype: int = 1) -> bytes:
    header = struct.pack("!HHHHHH", 0x1234, 0x0100, 1, 0, 0, 0)
    qparts = b"".join(bytes([len(p)]) + p.encode() for p in qname.split("."))
    question = qparts + b"\x00" + struct.pack("!HH", qtype, 1)
    return header + question


def _dns_response(qname: str, answer_ip: str, qtype: int = 1) -> bytes:
    header = struct.pack("!HHHHHH", 0x1234, 0x8180, 1, 1, 0, 0)
    qparts = b"".join(bytes([len(p)]) + p.encode() for p in qname.split("."))
    question = qparts + b"\x00" + struct.pack("!HH", qtype, 1)
    answer = (
        b"\xc0\x0c"  # pointer to qname
        + struct.pack("!HHIH", qtype, 1, 60, 4)
        + _ipv4_to_bytes(answer_ip)
    )
    return header + question + answer


class _PcapWriter:
    def __init__(self, path: str):
        self.f = open(path, "wb")
        self.f.write(struct.pack(
            "<IHHiIII", PCAP_MAGIC, PCAP_VERSION_MAJOR, PCAP_VERSION_MINOR,
            0, 0, 65535, LINKTYPE_ETHERNET,
        ))
        self._t = 1_700_000_000.0  # fixed base timestamp -> deterministic file

    def write(self, eth_payload: bytes, dt: float = 0.05):
        self._t += dt
        sec = int(self._t)
        usec = int((self._t - sec) * 1_000_000)
        self.f.write(struct.pack("<IIII", sec, usec, len(eth_payload), len(eth_payload)))
        self.f.write(eth_payload)

    def close(self):
        self.f.close()


CLIENT_MAC, SERVER_MAC = "02:00:00:00:00:01", "02:00:00:00:00:02"
CLIENT_IP, SERVER_IP, DNS_IP = "192.0.2.20", "192.0.2.10", "192.0.2.53"
RESOLVED_IP = "198.51.100.5"


def _eth_ip(src_mac, dst_mac, src_ip, dst_ip, proto, payload, ident):
    ip_hdr = _ipv4_header(src_ip, dst_ip, proto, len(payload), ident)
    return _eth_header(src_mac, dst_mac) + ip_hdr + payload


def generate(out_path: str):
    w = _PcapWriter(out_path)
    ident = 1000

    # --- DNS query + response (UDP, port 53) ---
    dns_q = _dns_query("demo.example.com")
    w.write(_eth_ip(CLIENT_MAC, SERVER_MAC, CLIENT_IP, DNS_IP, 17,
                     _udp_segment(CLIENT_IP, DNS_IP, 51000, 53, dns_q), ident)); ident += 1
    dns_r = _dns_response("demo.example.com", RESOLVED_IP)
    w.write(_eth_ip(SERVER_MAC, CLIENT_MAC, DNS_IP, CLIENT_IP, 17,
                     _udp_segment(DNS_IP, CLIENT_IP, 53, 51000, dns_r), ident)); ident += 1

    # --- TCP handshake + tiny HTTP-like request/response + close ---
    seq_c, seq_s = 1000, 5000
    w.write(_eth_ip(CLIENT_MAC, SERVER_MAC, CLIENT_IP, SERVER_IP, 6,
                     _tcp_segment(CLIENT_IP, SERVER_IP, 50000, 80, seq_c, 0, 0x02, b""), ident)); ident += 1
    seq_c += 1
    w.write(_eth_ip(SERVER_MAC, CLIENT_MAC, SERVER_IP, CLIENT_IP, 6,
                     _tcp_segment(SERVER_IP, CLIENT_IP, 80, 50000, seq_s, seq_c, 0x12, b""), ident)); ident += 1
    seq_s += 1
    w.write(_eth_ip(CLIENT_MAC, SERVER_MAC, CLIENT_IP, SERVER_IP, 6,
                     _tcp_segment(CLIENT_IP, SERVER_IP, 50000, 80, seq_c, seq_s, 0x10, b""), ident)); ident += 1

    http_req = b"GET /demo HTTP/1.1\r\nHost: demo.example.com\r\n\r\n"
    w.write(_eth_ip(CLIENT_MAC, SERVER_MAC, CLIENT_IP, SERVER_IP, 6,
                     _tcp_segment(CLIENT_IP, SERVER_IP, 50000, 80, seq_c, seq_s, 0x18, http_req), ident)); ident += 1
    seq_c += len(http_req)
    w.write(_eth_ip(SERVER_MAC, CLIENT_MAC, SERVER_IP, CLIENT_IP, 6,
                     _tcp_segment(SERVER_IP, CLIENT_IP, 80, 50000, seq_s, seq_c, 0x10, b""), ident)); ident += 1

    http_resp = b"HTTP/1.1 200 OK\r\nContent-Length: 13\r\n\r\nHello, demo!\n"
    w.write(_eth_ip(SERVER_MAC, CLIENT_MAC, SERVER_IP, CLIENT_IP, 6,
                     _tcp_segment(SERVER_IP, CLIENT_IP, 80, 50000, seq_s, seq_c, 0x18, http_resp), ident)); ident += 1
    seq_s += len(http_resp)
    w.write(_eth_ip(CLIENT_MAC, SERVER_MAC, CLIENT_IP, SERVER_IP, 6,
                     _tcp_segment(CLIENT_IP, SERVER_IP, 50000, 80, seq_c, seq_s, 0x10, b""), ident)); ident += 1

    w.write(_eth_ip(CLIENT_MAC, SERVER_MAC, CLIENT_IP, SERVER_IP, 6,
                     _tcp_segment(CLIENT_IP, SERVER_IP, 50000, 80, seq_c, seq_s, 0x11, b""), ident)); ident += 1
    seq_c += 1
    w.write(_eth_ip(SERVER_MAC, CLIENT_MAC, SERVER_IP, CLIENT_IP, 6,
                     _tcp_segment(SERVER_IP, CLIENT_IP, 80, 50000, seq_s, seq_c, 0x11, b""), ident)); ident += 1
    seq_s += 1
    w.write(_eth_ip(CLIENT_MAC, SERVER_MAC, CLIENT_IP, SERVER_IP, 6,
                     _tcp_segment(CLIENT_IP, SERVER_IP, 50000, 80, seq_c, seq_s, 0x10, b""), ident)); ident += 1

    # --- ICMP echo request + reply ---
    w.write(_eth_ip(CLIENT_MAC, SERVER_MAC, CLIENT_IP, SERVER_IP, 1,
                     _icmp_echo(8, 1, 1, b"demo-ping"), ident)); ident += 1
    w.write(_eth_ip(SERVER_MAC, CLIENT_MAC, SERVER_IP, CLIENT_IP, 1,
                     _icmp_echo(0, 1, 1, b"demo-ping"), ident)); ident += 1

    w.close()
    print(f"Wrote demo PCAP to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="data/demo.pcap")
    args = parser.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    generate(args.out)
