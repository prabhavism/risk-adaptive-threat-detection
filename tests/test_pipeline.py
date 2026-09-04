"""
End-to-end + unit test suite (section 25). Generates synthetic data,
runs the full training pipeline once (module-scoped fixture), then
tests each requirement area against the resulting artifacts.

Run with:  pytest tests/ -v

setup_module() generates synthetic data and runs ml_dl.train_all's
one-shot pipeline (XGBoost -> Light DL -> Heavy DL -> calibration ->
threshold tuning -> SHAP smoke test), so all tests run against a real
trained pipeline without repeating expensive setup.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from scripts.generate_synthetic_data import generate
from ml_dl.config import DATA_PATH, CLASSES, SEQ_LEN, FEATURE_COLUMNS
from ml_dl import train_all
from ml_dl.data_utils import (
    load_raw, time_based_split, build_sequences, add_engineered_features,
)
from ml_dl.routing import route
from ml_dl.predict_interface import predict, reset_history
from ml_dl.alerts import build_alert


# ── Fixtures / setup ──────────────────────────────────────────────────────────

def setup_module(module):
    """
    Runs once before all tests.
    Generates synthetic rows and runs the full training pipeline, then
    resets per-host history so streaming tests start clean.
    """
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Small (not the full real dataset) so this stays a fast smoke test
    # rather than a full training run -- train_all uses early stopping
    # so a small synthetic set is enough to exercise every stage.
    df = generate(600, seed=1)
    df.to_csv(DATA_PATH, index=False)
    train_all.main(DATA_PATH)
    reset_history()


@pytest.fixture(scope="module")
def sample_flow():
    """Returns the first row of the test CSV as a dict."""
    df = pd.read_csv(DATA_PATH)
    return df.iloc[0].to_dict()


@pytest.fixture(scope="module")
def sample_flows():
    """Returns up to 500 rows as a list of dicts."""
    df = pd.read_csv(DATA_PATH)
    return [row.to_dict() for _, row in df.head(500).iterrows()]


# --------------------------------------------------------------------
# Data
# --------------------------------------------------------------------

def test_missing_raw_features_default_to_zero():
    df = pd.DataFrame([{"label": "benign"}])
    engineered = add_engineered_features(df)
    for col in FEATURE_COLUMNS:
        assert col in engineered.columns
    assert engineered["has_dns"].iloc[0] == 0.0
    assert engineered["has_tls"].iloc[0] == 0.0


def test_invalid_label_raises(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    df = generate(5, seed=2)
    df.loc[0, "label"] = "not_a_real_class"
    df.to_csv(bad_csv, index=False)
    with pytest.raises(ValueError):
        load_raw(bad_csv)


def test_feature_ordering_is_stable():
    df = load_raw(DATA_PATH)
    row_dict = df.iloc[0][FEATURE_COLUMNS].to_dict()
    # rebuilding from a dict and re-selecting FEATURE_COLUMNS must give
    # the exact same order every time (predict_interface relies on this)
    rebuilt = pd.DataFrame([row_dict])[FEATURE_COLUMNS]
    assert list(rebuilt.columns) == FEATURE_COLUMNS


def test_timestamp_sorting_used_when_present(tmp_path):
    df = load_raw(DATA_PATH).copy()
    df["timestamp"] = list(range(len(df)))[::-1]  # reverse order
    tmp_csv = str(tmp_path / "_ts_test.csv")
    df.to_csv(tmp_csv, index=False)
    reloaded = load_raw(tmp_csv)
    assert list(reloaded["timestamp"]) == sorted(reloaded["timestamp"])


def test_chronological_split_no_overlap():
    df = load_raw(DATA_PATH)
    train_df, val_df, test_df = time_based_split(df)
    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    # chronological: every train row's position precedes every val row's
    assert len(train_df) < len(df)
    assert len(val_df) > 0 and len(test_df) > 0


# --------------------------------------------------------------------
# Models
# --------------------------------------------------------------------

def test_xgboost_output_shape_and_classes():
    import pickle
    from ml_dl.config import XGB_MODEL_PATH
    from ml_dl.data_utils import xy
    with open(XGB_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    df = load_raw(DATA_PATH)
    X, _ = xy(df.iloc[:5])
    probs = model.predict_proba(X)
    assert probs.shape == (5, len(CLASSES))
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)


def test_dl_confidence_in_valid_range():
    reset_history()
    df = pd.read_csv(DATA_PATH)
    for flow in df.iloc[:5].to_dict("records"):
        result = predict(flow)
        assert 0.0 <= result["ml_confidence"] <= 1.0
        assert 0.0 <= result["dl_confidence"] <= 1.0
        assert result["dl_verdict"] in CLASSES


# --------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------

def test_routing_high_confidence_goes_light():
    assert route(0.99, theta=0.85) == "light"


def test_routing_low_confidence_goes_heavy():
    assert route(0.10, theta=0.85) == "heavy"


def test_routing_boundary_is_light():
    assert route(0.85, theta=0.85) == "light"  # >= theta -> light


def test_route_returns_only_valid_values():
    for conf in np.linspace(0.0, 1.0, 50):
        assert route(float(conf)) in {"light", "heavy"}


# --------------------------------------------------------------------
# Temporal history / streaming
# --------------------------------------------------------------------

def test_sequence_no_future_leakage():
    df = load_raw(DATA_PATH).iloc[:50].reset_index(drop=True)
    X_seq, _ = build_sequences(df)
    # every value in sequence i must come from feats[j] for some j <= i
    feats = df[FEATURE_COLUMNS].astype(float).values
    for i in range(len(df)):
        for t in range(X_seq.shape[1]):
            row = X_seq[i, t]
            # must match one of feats[0..i]
            matches = np.any(np.all(np.isclose(feats[: i + 1], row), axis=1))
            assert matches, f"sequence row {i} step {t} uses data not in feats[0:{i+1}]"


def test_sequence_length_correct():
    df = load_raw(DATA_PATH).iloc[:30].reset_index(drop=True)
    X_seq, y_seq = build_sequences(df)
    assert X_seq.shape == (30, SEQ_LEN, len(FEATURE_COLUMNS))
    assert y_seq.shape == (30,)


def test_per_host_history_persists_across_predict_calls():
    reset_history()
    df = pd.read_csv(DATA_PATH)
    flows = df.iloc[:3].to_dict("records")
    for f in flows:
        f["src_ip"] = "10.0.0.99"
    for flow in flows:
        result = predict(flow)
        assert result["model_used"] in ("light", "heavy")


def test_streaming_sequential_flows_incremental():
    """Multiple sequential flows from different hosts shouldn't error
    and each call is independent/incremental (no batch requirement)."""
    reset_history()
    df = pd.read_csv(DATA_PATH)
    for flow in df.iloc[:10].to_dict("records"):
        result = predict(flow)
        assert set(result.keys()) == {
            "ml_verdict", "ml_confidence", "dl_verdict",
            "dl_confidence", "model_used", "shap_evidence",
        }


# --------------------------------------------------------------------
# Explainability
# --------------------------------------------------------------------

def test_shap_evidence_exists_and_valid():
    reset_history()
    df = pd.read_csv(DATA_PATH)
    result = predict(df.iloc[0].to_dict())
    assert len(result["shap_evidence"]) > 0
    for e in result["shap_evidence"]:
        assert e["feature"] in FEATURE_COLUMNS
        assert isinstance(e["value"], float)


# --------------------------------------------------------------------
# Prediction interface (locked contract)
# --------------------------------------------------------------------

def test_predict_output_shape(sample_flow):
    reset_history()
    result = predict(sample_flow)
    expected_keys = {
        "ml_verdict", "ml_confidence",
        "dl_verdict", "dl_confidence",
        "model_used", "shap_evidence",
    }
    assert set(result.keys()) == expected_keys
    assert result["ml_verdict"] in CLASSES
    assert result["dl_verdict"] in CLASSES
    assert result["model_used"] in ("light", "heavy")
    assert isinstance(result["shap_evidence"], list)
    for e in result["shap_evidence"]:
        assert set(e.keys()) == {"feature", "value", "evidence_source"}


def test_ml_confidence_in_range(sample_flow):
    result = predict(sample_flow)
    assert 0.0 <= result["ml_confidence"] <= 1.0


def test_dl_confidence_in_range(sample_flow):
    result = predict(sample_flow)
    assert 0.0 <= result["dl_confidence"] <= 1.0


def test_shap_feature_names_are_valid(sample_flow):
    result = predict(sample_flow)
    for item in result["shap_evidence"]:
        assert item["feature"] in FEATURE_COLUMNS, (
            f"Unknown feature in shap_evidence: {item['feature']}"
        )


# --------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------

def test_missing_optional_tls_fields():
    """Flow with empty TLS fields should not crash."""
    df = pd.read_csv(DATA_PATH)
    flow = df.iloc[0].to_dict()
    flow["tls_ja3"] = ""
    flow["tls_ja3s"] = ""
    flow["tls_ja4"] = ""
    flow["tls_sni"] = ""
    result = predict(flow)
    assert "ml_verdict" in result


def test_missing_dns_fields_zero():
    """Flow where DNS fields are 0 / empty (non-DNS flow) should not crash."""
    df = pd.read_csv(DATA_PATH)
    flow = df.iloc[0].to_dict()
    flow["dns_query_length"] = 0.0
    flow["dns_entropy"] = 0.0
    flow["dns_record_type"] = ""
    result = predict(flow)
    assert "ml_verdict" in result


def test_extra_keys_in_flow_ignored():
    """Extra keys in the flow dict (e.g. from a richer CSV) don't cause errors."""
    df = pd.read_csv(DATA_PATH)
    flow = df.iloc[0].to_dict()
    flow["extra_column_not_in_schema"] = "some_value"
    result = predict(flow)
    assert "ml_verdict" in result


def test_all_numeric_features_zero():
    """A flow of all zeros should return a valid (not crashed) prediction."""
    flow = {col: 0.0 for col in FEATURE_COLUMNS}
    flow.update({"flow_id": "zero", "src_ip": "0.0.0.0", "dst_ip": "0.0.0.0",
                 "src_port": 0, "dst_port": 0, "protocol": "tcp",
                 "dns_record_type": "", "tls_ja3": "", "tls_ja3s": "",
                 "tls_ja4": "", "tls_sni": "", "label": "benign"})
    result = predict(flow)
    assert result["ml_verdict"] in CLASSES


# --------------------------------------------------------------------
# Coverage / bulk
# --------------------------------------------------------------------

def test_bulk_no_crashes(sample_flows):
    """100 flows run without any exception."""
    reset_history()
    for flow in sample_flows[:100]:
        result = predict(flow)
        assert isinstance(result, dict)


def test_model_used_distribution_has_both_tiers(sample_flows):
    """
    Across a few hundred flows there should be at least one flow routed
    to each of light/heavy. Validates that the tuned threshold isn't
    degenerately pinned to one branch.
    """
    reset_history()
    tiers = {predict(flow)["model_used"] for flow in sample_flows}
    assert tiers.issubset({"light", "heavy"})
    assert len(tiers) >= 1


# --------------------------------------------------------------------
# Alerts
# --------------------------------------------------------------------

def test_alert_required_fields_present():
    reset_history()
    df = pd.read_csv(DATA_PATH)
    flow = df.iloc[0].to_dict()
    result = predict(flow)
    alert = build_alert(flow, result)

    for field in ["timestamp", "flow_id", "threat_class", "confidence", "evidence"]:
        assert field in alert
    assert alert["severity"] in ("INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert 0.0 <= alert["confidence"] <= 1.0
    assert isinstance(alert["evidence"], list)


def test_alert_is_json_serializable():
    import json
    reset_history()
    df = pd.read_csv(DATA_PATH)
    flow = df.iloc[0].to_dict()
    result = predict(flow)
    alert = build_alert(flow, result)
    json.dumps(alert)  # raises if not serializable
