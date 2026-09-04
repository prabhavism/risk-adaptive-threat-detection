"""
Turns a stream of ingest.schemas.Packet into ingest.schemas.FlowState
records, incrementally (section 19/20 of the ingestion brief): one
packet updates one flow's state, in O(1), never re-scanning history.

Bidirectional handling: a flow's canonical key is direction-independent
(the lower-sorted endpoint is always "A"), so a response packet updates
the SAME FlowState as its request instead of creating a second,
unrelated flow. forward_* counters always mean "packets/bytes sent by
the flow's initiator" and reverse_* means "sent by the responder",
regardless of which physical direction scapy handed us the packet in.

Finalization policy (documented per section 20):
  - TCP: finalized when a FIN or RST is observed (flow-finalization
    inference is preferred over per-packet inference for stability).
  - Any protocol: finalized after `timeout_for(protocol)` seconds of
    inactivity (checked incrementally on every add_packet() call using
    the current packet's timestamp -- no separate background thread
    needed for PCAP replay; a live-capture caller should also call
    check_timeouts() periodically using wall-clock time so idle flows
    still expire when no new packets are arriving at all).
  - flush(): finalizes everything still active, used at PCAP/replay end.

Memory bounds (section 8): active_flows is capped at
config.max_active_flows. When full, the oldest-by-last_seen flows are
expired first (finalize_reason="capacity_evict") before any new flow
is admitted -- the system never grows an unbounded dict and never
crashes on a flow burst. Per-flow packet history is capped at
max_history_per_flow. Per-source behavioral history is capped at
max_source_history_entries and pruned by
source_history_window_sec, so a spraying/scanning source can't grow
that state unboundedly either.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from typing import Optional

from ingest.config import IngestConfig, DEFAULT_CONFIG
from ingest.schemas import Packet, FlowState

logger = logging.getLogger("ingest.flow_builder")


def normalize_flow_key(src_ip: str, dst_ip: str, src_port: Optional[int],
                        dst_port: Optional[int], protocol: str):
    """
    Direction-independent flow key (section 4/5). TCP/UDP use the full
    5-tuple; ICMP/other protocols (no ports) fall back to src+dst+proto.
    Returns (canonical_key, is_forward) where is_forward tells the
    caller whether THIS packet's src matches the canonical "A" endpoint.
    """
    protocol = (protocol or "OTHER").upper()
    if protocol in ("TCP", "UDP") and src_port is not None and dst_port is not None:
        a = (src_ip, src_port)
        b = (dst_ip, dst_port)
    else:
        a = (src_ip, None)
        b = (dst_ip, None)

    if a <= b:
        endpoint_a, endpoint_b = a, b
        is_forward = True
    else:
        endpoint_a, endpoint_b = b, a
        is_forward = False

    key = (endpoint_a[0], endpoint_a[1], endpoint_b[0], endpoint_b[1], protocol)
    return key, is_forward


class _SourceHistory:
    """
    Bounded, TTL-pruned per-source-IP behavior tracker, used for
    dest_fanout / port_fanout / flow_count (section 18). Each entry is
    one *new flow* seen from that source, not every packet.
    """

    def __init__(self, window_sec: float, max_entries: int):
        self.window_sec = window_sec
        self.max_entries = max_entries
        self._entries: dict[str, deque] = {}

    def record_new_flow(self, src_ip: str, dst_ip: str, dst_port: Optional[int], ts: float):
        buf = self._entries.setdefault(src_ip, deque())
        buf.append((ts, dst_ip, dst_port))
        if len(buf) > self.max_entries:
            buf.popleft()

    def _prune(self, src_ip: str, now: float):
        buf = self._entries.get(src_ip)
        if not buf:
            return
        cutoff = now - self.window_sec
        while buf and buf[0][0] < cutoff:
            buf.popleft()

    def stats(self, src_ip: str, now: float) -> dict:
        self._prune(src_ip, now)
        buf = self._entries.get(src_ip, deque())
        return {
            "dest_fanout": len({e[1] for e in buf}),
            "port_fanout": len({e[2] for e in buf if e[2] is not None}),
            "flow_count": len(buf),
        }


class FlowBuilder:
    def __init__(self, config: IngestConfig = DEFAULT_CONFIG):
        self.config = config
        self.active_flows: dict[tuple, FlowState] = {}
        self._flow_counter = 0
        self.source_history = _SourceHistory(
            config.source_history_window_sec, config.max_source_history_entries
        )
        # stats for logging/benchmarking (section 8/34)
        self.total_flows_created = 0
        self.total_flows_finalized = 0
        self.capacity_evictions = 0

    def _new_flow_id(self) -> str:
        self._flow_counter += 1
        return f"flow-{self._flow_counter:06d}"

    def _enforce_capacity(self):
        if len(self.active_flows) < self.config.max_active_flows:
            return
        # Evict the single oldest-by-last_seen flow to make room. Doing
        # this one at a time (rather than a big batch) keeps the policy
        # simple and deterministic and is called before every new-flow
        # admission, so it self-corrects under sustained load.
        oldest_key = min(self.active_flows, key=lambda k: self.active_flows[k].last_seen)
        evicted = self.active_flows.pop(oldest_key)
        evicted.finalized = True
        evicted.finalize_reason = "capacity_evict"
        self.capacity_evictions += 1
        logger.warning(
            "max_active_flows (%d) reached, evicting oldest flow %s (last_seen=%s)",
            self.config.max_active_flows, evicted.flow_id, evicted.last_seen,
        )
        return evicted

    def add_packet(self, packet: Packet) -> list[FlowState]:
        """
        Update flow state with one packet. Returns a list of
        FlowState objects that were finalized as a *result* of this
        packet (TCP close, or capacity eviction to make room for a new
        flow) -- usually empty. Timeout-based finalization is handled
        separately by check_timeouts(), so it also works for idle
        periods with no incoming packets to piggyback on.
        """
        finalized: list[FlowState] = []
        key, is_forward = normalize_flow_key(
            packet.src_ip, packet.dst_ip, packet.src_port, packet.dst_port, packet.protocol
        )

        flow = self.active_flows.get(key)
        if flow is None:
            evicted = self._enforce_capacity()
            if evicted is not None:
                finalized.append(evicted)

            flow_id = self._new_flow_id()
            if is_forward:
                init_ip, init_port = packet.src_ip, packet.src_port
                resp_ip, resp_port = packet.dst_ip, packet.dst_port
            else:
                init_ip, init_port = packet.dst_ip, packet.dst_port
                resp_ip, resp_port = packet.src_ip, packet.src_port

            flow = FlowState(
                flow_id=flow_id,
                canonical_key=key,
                protocol=packet.protocol,
                initiator_ip=init_ip,
                initiator_port=init_port,
                responder_ip=resp_ip,
                responder_port=resp_port,
                start_time=packet.timestamp,
                last_seen=packet.timestamp,
            )
            self.active_flows[key] = flow
            self.total_flows_created += 1
            self.source_history.record_new_flow(init_ip, resp_ip, resp_port, packet.timestamp)

        self._apply_packet(flow, packet, is_forward)

        if packet.protocol == "TCP" and packet.tcp_flags:
            if "F" in packet.tcp_flags:
                flow.fin_seen = True
            if "R" in packet.tcp_flags:
                flow.rst_seen = True
            if flow.fin_seen or flow.rst_seen:
                self.active_flows.pop(key, None)
                flow.finalized = True
                flow.finalize_reason = "tcp_close"
                self.total_flows_finalized += 1
                finalized.append(flow)

        return finalized

    def _apply_packet(self, flow: FlowState, packet: Packet, is_forward: bool):
        flow.last_seen = packet.timestamp
        flow.packet_count += 1
        flow.byte_count += packet.packet_length

        if is_forward:
            flow.forward_packet_count += 1
            flow.forward_byte_count += packet.packet_length
        else:
            flow.reverse_packet_count += 1
            flow.reverse_byte_count += packet.packet_length

        cap = self.config.max_history_per_flow
        flow.packet_sizes.append(packet.packet_length)
        flow.packet_timestamps.append(packet.timestamp)
        if len(flow.packet_sizes) > cap:
            flow.packet_sizes.pop(0)
            flow.packet_timestamps.pop(0)

        if packet.dns_query_length is not None:
            flow.dns_query_lengths.append(packet.dns_query_length)
            flow.dns_query_names.append(packet.dns_query_name or "")
            flow.dns_record_types.append(packet.dns_record_type or "")
        if packet.tls_handshake:
            flow.tls_seen = True
            if packet.tls_sni and not flow.tls_sni:
                flow.tls_sni = packet.tls_sni

    def check_timeouts(self, now: float) -> list[FlowState]:
        """
        Finalize any active flow that's been idle longer than its
        protocol's timeout, relative to `now` (the current packet's
        timestamp in PCAP/replay mode, or wall-clock time in live mode).
        """
        expired = []
        for key, flow in list(self.active_flows.items()):
            timeout = self.config.timeout_for(flow.protocol)
            if now - flow.last_seen >= timeout:
                self.active_flows.pop(key, None)
                flow.finalized = True
                flow.finalize_reason = "timeout"
                self.total_flows_finalized += 1
                expired.append(flow)
        return expired

    def flush(self) -> list[FlowState]:
        """Finalize every remaining active flow (PCAP/replay end, or shutdown)."""
        remaining = list(self.active_flows.values())
        self.active_flows.clear()
        for flow in remaining:
            flow.finalized = True
            flow.finalize_reason = flow.finalize_reason or "flush"
            self.total_flows_finalized += 1
        return remaining

    def source_stats(self, src_ip: str, now: float) -> dict:
        return self.source_history.stats(src_ip, now)
