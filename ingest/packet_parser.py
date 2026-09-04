"""
Turns a raw scapy packet (from a PCAP or a live NIC) into a normalized
ingest.schemas.Packet. This is the only file that touches packet
internals -- everything downstream (flow_builder, feature_extractor)
works on the normalized dataclass, so a future switch to PyShark/dpkt
only needs a new function here.

Supports: Ethernet, IPv4, IPv6, TCP, UDP, ICMP, DNS (via scapy), and a
lightweight hand-rolled TLS ClientHello parser for the SNI extension
(see _extract_tls_sni). QUIC's payload is encrypted end-to-end
including SNI (unless ECH is absent and QUIC-TLS CRYPTO frames are
manually decoded, which this prototype does not attempt) -- QUIC flows
are still captured and featurized, just without tls_sni populated.

Nothing here decrypts anything. TLS SNI is read from the plaintext
ClientHello record, which is metadata sent unencrypted by design in
both TLS 1.2 and TLS 1.3 (absent Encrypted Client Hello) -- this is
not decryption, it's reading a plaintext header field, the same thing
any passive network monitor or firewall does.

Unsupported/malformed packets never raise out of parse_packet(); they
log a warning and return None, and the caller skips them.
"""
from __future__ import annotations

import logging
import struct
from typing import Optional

from ingest.schemas import Packet

logger = logging.getLogger("ingest.packet_parser")

_DNS_QTYPES = {1: "A", 28: "AAAA", 5: "CNAME", 15: "MX", 16: "TXT", 2: "NS", 6: "SOA", 33: "SRV"}

_TLS_HANDSHAKE_CONTENT_TYPE = 0x16
_TLS_CLIENT_HELLO_TYPE = 0x01
_TLS_EXT_SERVER_NAME = 0x0000


def _dns_qtype_name(qtype: int) -> str:
    return _DNS_QTYPES.get(int(qtype), "OTHER")


def _extract_tls_sni(raw_payload: bytes) -> Optional[str]:
    """
    Minimal, defensive TLS ClientHello parser that reads only the SNI
    extension from an unencrypted ClientHello record. Returns None for
    anything that isn't a well-formed ClientHello -- this must never
    raise, since it runs on arbitrary/possibly non-TLS TCP payloads.
    """
    try:
        if len(raw_payload) < 6:
            return None
        if raw_payload[0] != _TLS_HANDSHAKE_CONTENT_TYPE:
            return None
        # record: [type(1)][version(2)][length(2)] then handshake body
        body = raw_payload[5:]
        if len(body) < 4 or body[0] != _TLS_CLIENT_HELLO_TYPE:
            return None

        pos = 4  # skip handshake type(1) + length(3)
        pos += 2  # client_version
        pos += 32  # random
        if pos >= len(body):
            return None

        session_id_len = body[pos]
        pos += 1 + session_id_len
        if pos + 2 > len(body):
            return None

        cipher_suites_len = struct.unpack("!H", body[pos:pos + 2])[0]
        pos += 2 + cipher_suites_len
        if pos >= len(body):
            return None

        compression_len = body[pos]
        pos += 1 + compression_len
        if pos + 2 > len(body):
            return None

        extensions_len = struct.unpack("!H", body[pos:pos + 2])[0]
        pos += 2
        extensions_end = pos + extensions_len

        while pos + 4 <= min(extensions_end, len(body)):
            ext_type = struct.unpack("!H", body[pos:pos + 2])[0]
            ext_len = struct.unpack("!H", body[pos + 2:pos + 4])[0]
            ext_body = body[pos + 4:pos + 4 + ext_len]
            if ext_type == _TLS_EXT_SERVER_NAME and len(ext_body) > 5:
                # server_name_list: [list_len(2)][type(1)][name_len(2)][name]
                name_len = struct.unpack("!H", ext_body[3:5])[0]
                name = ext_body[5:5 + name_len]
                return name.decode("ascii", errors="ignore")
            pos += 4 + ext_len

        return None
    except Exception:
        return None


def parse_packet(scapy_pkt, capture_time: Optional[float] = None) -> Optional[Packet]:
    """
    scapy_pkt: a scapy packet object (from PcapReader or sniff()).
    Returns a normalized Packet, or None if the packet is non-IP,
    malformed, or otherwise unsupported (never raises).
    """
    try:
        from scapy.layers.inet import IP, TCP, UDP, ICMP
        from scapy.layers.inet6 import IPv6

        ts = float(capture_time if capture_time is not None else getattr(scapy_pkt, "time", 0.0))

        if scapy_pkt.haslayer(IP):
            ip_layer = scapy_pkt[IP]
        elif scapy_pkt.haslayer(IPv6):
            ip_layer = scapy_pkt[IPv6]
        else:
            return None  # non-IP (ARP, etc.) -- not a flow, skip

        src_ip = ip_layer.src
        dst_ip = ip_layer.dst

        protocol = "OTHER"
        src_port = dst_port = None
        tcp_flags = None
        payload = b""

        if scapy_pkt.haslayer(TCP):
            protocol = "TCP"
            tcp_layer = scapy_pkt[TCP]
            src_port = int(tcp_layer.sport)
            dst_port = int(tcp_layer.dport)
            tcp_flags = str(tcp_layer.flags)
            payload = bytes(tcp_layer.payload)
        elif scapy_pkt.haslayer(UDP):
            protocol = "UDP"
            udp_layer = scapy_pkt[UDP]
            src_port = int(udp_layer.sport)
            dst_port = int(udp_layer.dport)
            payload = bytes(udp_layer.payload)
        elif scapy_pkt.haslayer(ICMP):
            protocol = "ICMP"

        pkt = Packet(
            timestamp=ts,
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            dst_port=dst_port,
            protocol=protocol,
            packet_length=len(scapy_pkt),
            tcp_flags=tcp_flags,
            payload_length=len(payload),
        )

        # --- DNS metadata (scapy already parses this for us) ---
        try:
            from scapy.layers.dns import DNS
            if scapy_pkt.haslayer(DNS):
                dns_layer = scapy_pkt[DNS]
                if dns_layer.qdcount and dns_layer.qd is not None:
                    qname = dns_layer.qd.qname
                    if isinstance(qname, bytes):
                        qname = qname.decode("utf-8", errors="ignore")
                    qname = qname.rstrip(".")
                    pkt.dns_query_name = qname
                    pkt.dns_query_length = len(qname)
                    pkt.dns_record_type = _dns_qtype_name(dns_layer.qd.qtype)
        except Exception as e:
            logger.debug("DNS parse skipped: %s", e)

        # --- TLS SNI (plaintext ClientHello metadata only) ---
        if protocol == "TCP" and payload:
            sni = _extract_tls_sni(payload)
            if sni:
                pkt.tls_handshake = True
                pkt.tls_sni = sni

        return pkt

    except Exception as e:
        logger.warning("Failed to parse packet, skipping: %s", e)
        return None
