"""
Unit tests for the Part 2/3/7/8/9 fixes that don't require
xgboost/tensorflow/a trained model -- run standalone with just
pandas/numpy, so they execute even in environments where the heavier
ML deps aren't installed yet (see tests/test_pipeline.py for the full
trained-model integration tests, which do need those deps).

Run with: pytest tests/test_person2_fixes.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ml_dl.config import FEATURE_COLUMNS, ORIGINAL_FEATURE_COLUMNS
from ml_dl.data_utils import add_engineered_features
from ml_dl.alerts import build_alert
from ml_dl.host_history import _HostHistory


# --- Part 3: feature schema / ordering ------------------------------------

def test_feature_column_count_and_no_duplicates():
    # 17 original + 7 engineered = 24 (the brief's header calls this
    # "25 features" but its own itemized list totals 24 -- verified
    # count is 24; documented in docs/person2_guide.md).
    assert len(FEATURE_COLUMNS) == 24
    assert len(set(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS), "duplicate feature column"


def test_original_17_are_all_present_in_full_schema():
    assert len(ORIGINAL_FEATURE_COLUMNS) == 17
    for col in ORIGINAL_FEATURE_COLUMNS:
        assert col in FEATURE_COLUMNS


def test_engineered_columns_present_with_expected_names():
    expected_engineered = {
        "has_dns", "dns_is_A", "dns_is_AAAA", "dns_is_TXT", "dns_is_NS",
        "has_tls", "tls_sni_length",
    }
    assert expected_engineered.issubset(set(FEATURE_COLUMNS))


def test_add_engineered_features_produces_exact_locked_order():
    row = {
        "duration": 1.0, "packet_count": 5, "byte_count": 500,
        "packet_rate": 5.0, "byte_rate": 500.0, "avg_packet_size": 100.0,
        "byte_ratio": 1.0, "packet_ratio": 1.0, "dest_fanout": 1,
        "port_fanout": 1, "flow_count": 1, "entropy": 0.5,
        "dns_query_length": 0, "dns_entropy": 0.0, "iat_mean": 0.1,
        "iat_std": 0.05, "periodicity_score": 0.0,
        "dns_record_type": "A", "tls_ja3": "", "tls_sni": "",
    }
    import pandas as pd
    engineered = add_engineered_features(pd.DataFrame([row]))
    sliced = engineered[FEATURE_COLUMNS]
    assert list(sliced.columns) == FEATURE_COLUMNS


# --- Part 7: SHAP evidence_source -----------------------------------------
# (top_features() itself needs a real xgboost model -- see
# tests/test_pipeline.py for that. This just checks the contract shape
# alerts.py produces from a stub shap_evidence list.)

def test_alert_evidence_carries_evidence_source():
    flow = {
        "flow_id": "flow-1", "src_ip": "10.0.0.1", "dst_ip": "10.0.0.2",
        "protocol": "tcp", "packet_rate": 500.0,
        "dns_record_type": "", "tls_ja3": "", "tls_sni": "",
    }
    predict_result = {
        "ml_verdict": "ddos", "ml_confidence": 0.93,
        "dl_verdict": "ddos", "dl_confidence": 0.97,
        "model_used": "light",
        "shap_evidence": [
            {"feature": "packet_rate", "value": 0.41, "evidence_source": "xgboost_triage"},
        ],
    }
    alert = build_alert(flow, predict_result)
    assert alert["evidence"][0]["evidence_source"] == "xgboost_triage"
    assert alert["evidence"][0]["contribution"] == 0.41
    assert alert["evidence"][0]["value"] == 500.0  # actual feature value, not the shap number


# --- Part 8: alert model_agreement / xgb_verdict / dl_verdict -------------

def test_alert_records_model_agreement_true():
    flow = {"flow_id": "f1", "src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "protocol": "tcp"}
    predict_result = {
        "ml_verdict": "benign", "ml_confidence": 0.99,
        "dl_verdict": "benign", "dl_confidence": 0.98,
        "model_used": "light", "shap_evidence": [],
    }
    alert = build_alert(flow, predict_result)
    assert alert["model_agreement"] is True
    assert alert["xgb_verdict"] == "benign"
    assert alert["dl_verdict"] == "benign"


def test_alert_records_model_disagreement():
    flow = {"flow_id": "f2", "src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "protocol": "tcp"}
    predict_result = {
        "ml_verdict": "benign", "ml_confidence": 0.60,
        "dl_verdict": "c2_beaconing", "dl_confidence": 0.91,
        "model_used": "heavy", "shap_evidence": [],
    }
    alert = build_alert(flow, predict_result)
    assert alert["model_agreement"] is False
    assert alert["xgb_verdict"] == "benign"
    assert alert["dl_verdict"] == "c2_beaconing"
    assert alert["threat_class"] == "c2_beaconing"  # final verdict uses DL, not XGBoost
    # disagreement should be reflected as a lower risk_score than a
    # same-confidence agreeing case
    assert alert["risk_score"] < 0.91


# --- Part 9: bounded per-host history (predict_interface._HostHistory) ---

def test_host_history_bounded_lru_eviction():
    h = _HostHistory(max_hosts=3, seq_len=5)
    for i in range(10):
        h.append(f"host-{i}", np.zeros(4))
    assert len(h) == 3
    # only the last 3 hosts should still be tracked
    assert h.window("host-9") != []
    assert h.window("host-0") == []


def test_host_history_touch_refreshes_lru_order():
    h = _HostHistory(max_hosts=2, seq_len=5)
    h.append("a", np.zeros(4))
    h.append("b", np.zeros(4))
    h.window("a")            # touch "a" -> now most-recently-used
    h.append("c", np.zeros(4))  # should evict "b", not "a"
    assert h.window("a") != []
    assert h.window("b") == []
    assert h.window("c") != []


def test_host_history_per_host_seq_len_bounded():
    h = _HostHistory(max_hosts=10, seq_len=5)
    for i in range(20):
        h.append("host-x", np.array([float(i)]))
    window = h.window("host-x")
    assert len(window) == 5
    assert window[-1][0] == 19.0  # most recent kept
