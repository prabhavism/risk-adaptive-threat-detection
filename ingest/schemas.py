"""
Internal data model for the ingestion layer.

Packet is the normalized output of packet_parser.py -- the same shape
regardless of whether it came from a PCAP or a live NIC. FlowState is
the mutable, bounded, in-memory record flow_builder.py maintains while
a flow is active. Neither of these is the AI model's input -- that's
feature_extractor.py's job, which turns a *finalized* FlowState into
the raw flow dict ml_dl.predict_interface.predict() expects.

Metadata-only by construction: nothing here stores payload bytes.
`payload_length` is a count, not content.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Packet:
    """One parsed packet. See ingest/packet_parser.py for construction."""
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: Optional[int]
    dst_port: Optional[int]
    protocol: str  # "TCP" / "UDP" / "ICMP" / "OTHER"
    packet_length: int
    tcp_flags: Optional[str] = None
    payload_length: int = 0

    # Optional metadata-only fields, populated only when detectable.
    dns_query_name: Optional[str] = None
    dns_query_length: Optional[int] = None
    dns_record_type: Optional[str] = None
    tls_handshake: bool = False
    tls_sni: Optional[str] = None


@dataclass
class FlowState:
    """
    Mutable state for one bidirectional flow while it's active.

    "forward" == the direction the flow's initiator (first packet seen)
    sent in; "reverse" == the response direction. This is what makes
    byte_ratio/packet_ratio meaningful instead of just "total bytes".

    Bounded by construction: packet_sizes/timestamps are capped at
    MAX_HISTORY_PER_FLOW (see ingest/config.py) so a single long-lived
    flow can't grow memory unboundedly -- IAT/entropy/periodicity are
    computed from the most recent MAX_HISTORY_PER_FLOW packets, which
    is a documented approximation for very long flows, not the full
    history.
    """
    flow_id: str
    canonical_key: tuple
    protocol: str

    initiator_ip: str
    initiator_port: Optional[int]
    responder_ip: str
    responder_port: Optional[int]

    start_time: float
    last_seen: float

    packet_count: int = 0
    byte_count: int = 0
    forward_packet_count: int = 0
    reverse_packet_count: int = 0
    forward_byte_count: int = 0
    reverse_byte_count: int = 0

    packet_sizes: list = field(default_factory=list)
    packet_timestamps: list = field(default_factory=list)

    fin_seen: bool = False
    rst_seen: bool = False

    dns_query_lengths: list = field(default_factory=list)
    dns_query_names: list = field(default_factory=list)
    dns_record_types: list = field(default_factory=list)
    tls_seen: bool = False
    tls_sni: Optional[str] = None

    finalized: bool = False
    finalize_reason: Optional[str] = None  # "timeout" | "tcp_close" | "flush" | "capacity_evict"

    def duration(self) -> float:
        return max(self.last_seen - self.start_time, 0.0)
