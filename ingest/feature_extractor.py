"""
Converts a finalized ingest.schemas.FlowState into the raw flow dict
ml_dl.predict_interface.predict() expects -- i.e. this produces exactly
the columns Person 1's flow_features.csv would have had, computed from
observed packets instead of a CSV row.

Reuses ml_dl.config.ORIGINAL_FEATURE_COLUMNS as the canonical numeric
schema (section 13: "reuse the existing schema... rather than
duplicating it") plus the raw identity/categorical columns
(src_ip/dst_ip/ports/protocol, dns_record_type, tls_ja3, tls_sni) that
ml_dl.data_utils.add_engineered_features derives has_dns/has_tls/etc.
from. predict() does that derivation internally -- this module does
NOT compute has_dns/has_tls/tls_sni_length itself, to avoid a second,
possibly-inconsistent copy of that logic (see docs/interfaces.md).

No future information is ever used (section 15): every value here is
computed only from packets already observed for this flow plus this
source's own recent history (via FlowBuilder.source_stats), never from
other flows' future behavior or from any label.
"""
from __future__ import annotations

import math
from collections import Counter

from ml_dl.config import ORIGINAL_FEATURE_COLUMNS
from ingest.schemas import FlowState
from ingest.flow_builder import FlowBuilder

# Raw non-numeric / identity columns predict()'s engineering step reads
# in addition to ORIGINAL_FEATURE_COLUMNS (see ml_dl/data_utils.py
# add_engineered_features and docs/interfaces.md section 1).
IDENTITY_COLUMNS = ["flow_id", "src_ip", "dst_ip", "src_port", "dst_port", "protocol"]
RAW_CATEGORICAL_COLUMNS = ["dns_record_type", "tls_ja3", "tls_ja3s", "tls_ja4", "tls_sni"]

ALL_OUTPUT_COLUMNS = IDENTITY_COLUMNS + ORIGINAL_FEATURE_COLUMNS + RAW_CATEGORICAL_COLUMNS


def _shannon_entropy(values: list) -> float:
    """
    Shannon entropy (bits) over a discrete sample of *observed packet
    sizes* -- documented explicitly per section 14: this is entropy of
    packet-length metadata, never of payload/encrypted content. A flow
    with wildly varying packet sizes has higher entropy than one with
    uniform sizes (e.g. a DDoS flood of near-identical packets).
    """
    n = len(values)
    if n == 0:
        return 0.0
    counts = Counter(values)
    entropy = 0.0
    for c in counts.values():
        p = c / n
        entropy -= p * math.log2(p)
    return float(entropy)


def _iat_stats(timestamps: list) -> tuple[float, float]:
    if len(timestamps) < 2:
        return 0.0, 0.0
    iats = [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
    mean = sum(iats) / len(iats)
    if len(iats) < 2:
        return float(mean), 0.0
    variance = sum((x - mean) ** 2 for x in iats) / len(iats)
    return float(mean), float(math.sqrt(variance))


def _periodicity_score(iat_mean: float, iat_std: float) -> float:
    """
    Deterministic periodicity measure from observed inter-arrival
    timing only (section 14) -- no labels involved. Uses the
    coefficient of variation (CV = std/mean): low CV -> very regular
    spacing -> classic C2 beaconing signature -> score near 1.0. High
    CV -> irregular/bursty spacing -> score near 0.0.
    """
    if iat_mean <= 0:
        return 0.0
    cv = iat_std / iat_mean
    return float(1.0 / (1.0 + cv))


class FeatureExtractor:
    def __init__(self, builder: FlowBuilder):
        # Needs the same FlowBuilder instance so it can read
        # dest_fanout/port_fanout/flow_count from that source's
        # already-bounded, already-pruned recent-flow history rather
        # than recomputing/duplicating it (section 18).
        self.builder = builder

    def extract(self, flow: FlowState) -> dict:
        duration = flow.duration()
        packet_rate = flow.packet_count / duration if duration > 0 else 0.0
        byte_rate = flow.byte_count / duration if duration > 0 else 0.0
        avg_packet_size = flow.byte_count / flow.packet_count if flow.packet_count > 0 else 0.0

        byte_ratio = flow.forward_byte_count / max(flow.reverse_byte_count, 1)
        packet_ratio = flow.forward_packet_count / max(flow.reverse_packet_count, 1)

        iat_mean, iat_std = _iat_stats(flow.packet_timestamps)
        periodicity_score = _periodicity_score(iat_mean, iat_std)
        entropy = _shannon_entropy(flow.packet_sizes)

        src_stats = self.builder.source_stats(flow.initiator_ip, flow.last_seen)

        dns_query_length = (
            sum(flow.dns_query_lengths) / len(flow.dns_query_lengths)
            if flow.dns_query_lengths else 0.0
        )
        dns_entropy = (
            sum(_shannon_entropy(list(name)) for name in flow.dns_query_names) / len(flow.dns_query_names)
            if flow.dns_query_names else 0.0
        )
        dns_record_type = flow.dns_record_types[-1] if flow.dns_record_types else ""

        row = {
            "flow_id": flow.flow_id,
            "src_ip": flow.initiator_ip,
            "dst_ip": flow.responder_ip,
            "src_port": flow.initiator_port,
            "dst_port": flow.responder_port,
            "protocol": flow.protocol.lower(),

            "duration": duration,
            "packet_count": flow.packet_count,
            "byte_count": flow.byte_count,
            "packet_rate": packet_rate,
            "byte_rate": byte_rate,
            "avg_packet_size": avg_packet_size,
            "byte_ratio": byte_ratio,
            "packet_ratio": packet_ratio,
            "dest_fanout": src_stats["dest_fanout"],
            "port_fanout": src_stats["port_fanout"],
            "flow_count": src_stats["flow_count"],
            "entropy": entropy,
            "dns_query_length": dns_query_length,
            "dns_entropy": dns_entropy,
            "iat_mean": iat_mean,
            "iat_std": iat_std,
            "periodicity_score": periodicity_score,

            "dns_record_type": dns_record_type,
            # Real JA3/JA3S/JA4 hashing requires a cipher-suite/extension
            # fingerprint table this prototype doesn't implement; we only
            # know a TLS handshake was *observed*, which is enough for
            # predict()'s has_tls presence signal (see ml_dl/data_utils.py).
            "tls_ja3": "observed" if flow.tls_seen else "",
            "tls_ja3s": "",
            "tls_ja4": "",
            "tls_sni": flow.tls_sni or "",
        }

        assert set(row.keys()) == set(ALL_OUTPUT_COLUMNS), (
            "feature_extractor output columns drifted from the locked schema "
            f"-- got {sorted(row.keys())}, expected {sorted(ALL_OUTPUT_COLUMNS)}"
        )
        return row
