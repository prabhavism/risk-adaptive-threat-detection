"""
The glue layer (section 23/40): packets in, existing predict(flow)
results out. This file is the ONLY place that calls ml_dl's predict();
nothing in packet_parser/flow_builder/feature_extractor imports ml_dl,
so the ingestion layer can be developed/tested independently of the AI
engine (and vice versa).

    packet -> FlowBuilder -> (finalized) FlowState
           -> FeatureExtractor -> flow dict
           -> ml_dl.predict_interface.predict(flow) -- UNCHANGED
           -> result dict for Person 3
"""
from __future__ import annotations

import datetime
import logging
from typing import Callable, Optional

from ingest.config import IngestConfig, DEFAULT_CONFIG
from ingest.flow_builder import FlowBuilder
from ingest.feature_extractor import FeatureExtractor
from ingest.schemas import Packet, FlowState

logger = logging.getLogger("ingest.pipeline")


def _default_predict_fn():
    """
    Lazily imported so the ingestion layer alone (packet parsing, flow
    building, feature extraction) can be developed and unit-tested
    without xgboost/tensorflow/shap installed or trained model files
    present -- only actually calling predict() requires them.
    """
    from ml_dl.predict_interface import predict
    return predict


class DetectionPipeline:
    def __init__(self, config: IngestConfig = DEFAULT_CONFIG,
                 predict_fn: Optional[Callable[[dict], dict]] = None):
        self.config = config
        self.builder = FlowBuilder(config)
        self.extractor = FeatureExtractor(self.builder)
        self._predict_fn = predict_fn  # resolved lazily if None
        self.total_predictions = 0

    def _predict(self, features: dict) -> dict:
        if self._predict_fn is None:
            self._predict_fn = _default_predict_fn()
        return self._predict_fn(features)

    def _finalize_result(self, flow: FlowState) -> dict:
        features = self.extractor.extract(flow)
        prediction = self._predict(features)
        self.total_predictions += 1
        logger.info(
            "Flow %s finalized (%s): ml=%s dl=%s conf=%.3f model=%s",
            flow.flow_id, flow.finalize_reason,
            prediction.get("ml_verdict"), prediction.get("dl_verdict"),
            prediction.get("dl_confidence", 0.0), prediction.get("model_used"),
        )
        return {
            "flow_id": flow.flow_id,
            "timestamp": datetime.datetime.utcfromtimestamp(flow.last_seen).isoformat() + "Z",
            "flow": {
                "src_ip": flow.initiator_ip,
                "dst_ip": flow.responder_ip,
                "src_port": flow.initiator_port,
                "dst_port": flow.responder_port,
                "protocol": flow.protocol,
                "packet_count": flow.packet_count,
                "byte_count": flow.byte_count,
                "duration": flow.duration(),
                "finalize_reason": flow.finalize_reason,
            },
            "features": features,
            "prediction": prediction,  # untouched ml_dl.predict_interface.predict() output
        }

    def process_packet(self, packet: Packet) -> list[dict]:
        """Feed one packet in. Returns results for any flows finalized
        as a direct/immediate consequence (TCP close, capacity
        eviction, or timeout relative to this packet's timestamp)."""
        results = []
        for flow in self.builder.add_packet(packet):
            results.append(self._finalize_result(flow))
        for flow in self.builder.check_timeouts(packet.timestamp):
            results.append(self._finalize_result(flow))
        return results

    def check_idle_timeouts(self, now: float) -> list[dict]:
        """For live capture: call periodically with wall-clock time so
        flows that go idle (no more packets at all) still expire even
        though nothing is arriving to piggyback the check on."""
        return [self._finalize_result(f) for f in self.builder.check_timeouts(now)]

    def flush(self) -> list[dict]:
        """Finalize every remaining active flow. Call at PCAP/replay end
        or on shutdown so no in-progress flow is silently dropped."""
        return [self._finalize_result(f) for f in self.builder.flush()]

    def close(self):
        self.flush()

    def stats(self) -> dict:
        return {
            "active_flows": len(self.builder.active_flows),
            "total_flows_created": self.builder.total_flows_created,
            "total_flows_finalized": self.builder.total_flows_finalized,
            "capacity_evictions": self.builder.capacity_evictions,
            "total_predictions": self.total_predictions,
        }
