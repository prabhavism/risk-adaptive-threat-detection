"""
Integration test for the full ingestion chain (section 38/39):

    packet -> FlowBuilder -> FeatureExtractor -> predict(flow)

Uses an injected stub predict_fn instead of the real
ml_dl.predict_interface.predict, so this test runs without trained
model artifacts (model.pkl / *.keras / threshold.pkl) present -- it
verifies the ingestion layer's wiring and the shape of the contract,
not model accuracy. For the real end-to-end test against the actual
trained model, see the "final end-to-end" checklist in ingest/README.md
-- run scripts/replay_pcap.py against a real PCAP after `python -m
ml_dl.train_all` has produced the model files.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.config import IngestConfig
from ingest.pipeline import DetectionPipeline
from ingest.schemas import Packet
from ingest.feature_extractor import ALL_OUTPUT_COLUMNS


def _stub_predict(flow_dict: dict) -> dict:
    """Mimics ml_dl.predict_interface.predict()'s exact return shape
    (docs/interfaces.md section 2) without loading any real model."""
    return {
        "ml_verdict": "benign",
        "ml_confidence": 0.99,
        "dl_verdict": "benign",
        "dl_confidence": 0.98,
        "model_used": "light",
        "shap_evidence": [{"feature": "packet_rate", "value": 0.01}],
    }


def make_packet(t, **kwargs):
    defaults = dict(src_ip="10.0.0.1", dst_ip="10.0.0.2", src_port=50000,
                     dst_port=443, protocol="TCP", packet_length=100, tcp_flags=None)
    defaults.update(kwargs)
    return Packet(timestamp=t, **defaults)


def test_pipeline_produces_result_on_tcp_close():
    pipeline = DetectionPipeline(IngestConfig(), predict_fn=_stub_predict)
    pipeline.process_packet(make_packet(0.0, tcp_flags="S"))
    results = pipeline.process_packet(make_packet(0.2, tcp_flags="FA"))

    assert len(results) == 1
    result = results[0]
    assert set(result.keys()) == {"flow_id", "timestamp", "flow", "features", "prediction"}
    assert set(result["features"].keys()) == set(ALL_OUTPUT_COLUMNS)
    # predict()'s return dict must pass through completely untouched
    assert result["prediction"] == _stub_predict(result["features"])


def test_pipeline_flush_scores_remaining_flows():
    pipeline = DetectionPipeline(IngestConfig(), predict_fn=_stub_predict)
    pipeline.process_packet(make_packet(0.0, dst_port=443))
    pipeline.process_packet(make_packet(0.0, dst_port=80))
    results = pipeline.flush()
    assert len(results) == 2
    assert pipeline.stats()["active_flows"] == 0


def test_pipeline_never_calls_real_predict_when_stub_given():
    # Sanity check that the injected predict_fn is actually what's used
    # (i.e. lazy-loading ml_dl.predict_interface didn't happen).
    calls = []

    def counting_predict(flow):
        calls.append(flow)
        return _stub_predict(flow)

    pipeline = DetectionPipeline(IngestConfig(), predict_fn=counting_predict)
    pipeline.process_packet(make_packet(0.0, tcp_flags="S"))
    pipeline.process_packet(make_packet(0.1, tcp_flags="FA"))
    assert len(calls) == 1
