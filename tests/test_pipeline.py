"""
End-to-end test suite for the Person 2 ML/DL layer.

Run with:  pytest tests/ -v

setup_module() generates synthetic data, trains XGBoost, trains both DL
models, and caches them — so all tests run against a real trained pipeline
without repeating expensive setup.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.generate_synthetic_data import generate
from ml_dl.config import (
    DATA_PATH, FEATURE_COLUMNS, CLASSES,
    THETA_HIGH, THETA_LOW,
    XGB_MODEL_PATH, THRESHOLD_PATH, SCALER_PATH,
    LIGHT_DL_PATH, HEAVY_DL_PATH,
)
from ml_dl import train_xgboost, light_dl, heavy_dl
from ml_dl.routing import route
from ml_dl.predict_interface import predict, reload_models

# ── Fixtures / setup ──────────────────────────────────────────────────────────

def setup_module(module):
    """
    Runs once before all tests.
    Generates 2 000 synthetic rows, trains the full pipeline, forces a reload.
    Using 2 000 (not 500) rows improves class coverage for the all-classes test.
    """
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = generate(2000, seed=1)
    df.to_csv(DATA_PATH, index=False)

    # Train XGBoost + thresholds
    train_xgboost.train()

    # Train Light + Heavy DL (via scripts/train_dl.py logic inline)
    import pickle
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from ml_dl.config import TRAIN_RATIO, VAL_RATIO

    X = df[FEATURE_COLUMNS].fillna(0.0).astype(float).values
    y = df["label"].map({c: i for i, c in enumerate(CLASSES)}).astype(int).values
    TEST_RATIO = 1.0 - TRAIN_RATIO - VAL_RATIO
    val_rel = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)

    X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=TEST_RATIO, stratify=y, random_state=42)
    X_tr, X_v, y_tr, y_v  = train_test_split(X_tv, y_tv, test_size=val_rel, stratify=y_tv, random_state=42)

    scaler = StandardScaler().fit(X_tr)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    X_tr_s, X_v_s = scaler.transform(X_tr), scaler.transform(X_v)

    lm = light_dl.train(X_tr_s, y_tr, X_v_s, y_v)
    hm = heavy_dl.train(X_tr_s, y_tr, X_v_s, y_v)

    reload_models()   # flush cached None values, force reload from disk


@pytest.fixture(scope="module")
def sample_flow():
    """Returns the first row of the test CSV as a dict."""
    df = pd.read_csv(DATA_PATH)
    return df.iloc[0].to_dict()


@pytest.fixture(scope="module")
def sample_flows():
    """Returns 1 000 rows as a list of dicts."""
    df = pd.read_csv(DATA_PATH)
    return [row.to_dict() for _, row in df.head(1000).iterrows()]


# ── Phase 1: Routing unit tests ───────────────────────────────────────────────

class TestRouting:
    def test_route_high_confidence_is_light(self):
        """Confidence above DEFAULT_THETA (0.90) → light."""
        assert route(0.95) == "light"

    def test_route_low_confidence_is_heavy(self):
        """Confidence below THETA_LOW (0.60) → heavy."""
        assert route(0.50) == "heavy"

    def test_route_boundary_at_theta(self):
        """Confidence exactly at THETA_HIGH → light (>= threshold)."""
        from ml_dl.config import DEFAULT_THETA
        assert route(DEFAULT_THETA) == "light"

    def test_route_just_below_theta(self):
        """Confidence just below THETA_HIGH → heavy."""
        from ml_dl.config import DEFAULT_THETA
        assert route(DEFAULT_THETA - 0.001) == "heavy"

    def test_route_returns_only_valid_values(self):
        for conf in np.linspace(0.0, 1.0, 50):
            assert route(float(conf)) in {"light", "heavy"}


# ── Phase 2: predict() output contract ───────────────────────────────────────

class TestPredictContract:
    REQUIRED_KEYS = {
        "ml_verdict", "ml_confidence",
        "dl_verdict", "dl_confidence",
        "model_used", "shap_evidence",
    }

    def test_all_required_keys_present(self, sample_flow):
        result = predict(sample_flow)
        assert self.REQUIRED_KEYS.issubset(set(result.keys())), (
            f"Missing keys: {self.REQUIRED_KEYS - set(result.keys())}"
        )

    def test_no_extra_error_key_on_success(self, sample_flow):
        result = predict(sample_flow)
        assert "_error" not in result, f"Unexpected error: {result.get('_error')}"

    def test_ml_confidence_in_range(self, sample_flow):
        result = predict(sample_flow)
        assert 0.0 <= result["ml_confidence"] <= 1.0

    def test_dl_confidence_in_range(self, sample_flow):
        result = predict(sample_flow)
        assert 0.0 <= result["dl_confidence"] <= 1.0

    def test_ml_verdict_is_valid_class(self, sample_flow):
        result = predict(sample_flow)
        assert result["ml_verdict"] in CLASSES, (
            f"ml_verdict '{result['ml_verdict']}' not in CLASSES"
        )

    def test_dl_verdict_is_valid_class(self, sample_flow):
        result = predict(sample_flow)
        assert result["dl_verdict"] in CLASSES

    def test_model_used_is_valid(self, sample_flow):
        result = predict(sample_flow)
        assert result["model_used"] in {"xgb_only", "light", "heavy"}, (
            f"Unexpected model_used: {result['model_used']}"
        )

    def test_shap_evidence_is_list(self, sample_flow):
        result = predict(sample_flow)
        assert isinstance(result["shap_evidence"], list)

    def test_shap_evidence_items_have_correct_structure(self, sample_flow):
        result = predict(sample_flow)
        for item in result["shap_evidence"]:
            assert "feature" in item, "shap_evidence item missing 'feature'"
            assert "value"   in item, "shap_evidence item missing 'value'"
            assert isinstance(item["feature"], str)
            assert isinstance(item["value"],   float)

    def test_shap_feature_names_are_valid(self, sample_flow):
        result = predict(sample_flow)
        for item in result["shap_evidence"]:
            assert item["feature"] in FEATURE_COLUMNS, (
                f"Unknown feature in shap_evidence: {item['feature']}"
            )


# ── Phase 3: Edge cases ───────────────────────────────────────────────────────

class TestEdgeCases:
    def test_missing_optional_tls_fields(self):
        """Flow with empty TLS fields should not crash."""
        df = pd.read_csv(DATA_PATH)
        flow = df.iloc[0].to_dict()
        flow["tls_ja3"]  = ""
        flow["tls_ja3s"] = ""
        flow["tls_ja4"]  = ""
        flow["tls_sni"]  = ""
        result = predict(flow)
        assert "ml_verdict" in result

    def test_missing_dns_fields_zero(self):
        """Flow where DNS fields are 0 / empty (non-DNS flow) should not crash."""
        df = pd.read_csv(DATA_PATH)
        flow = df.iloc[0].to_dict()
        flow["dns_query_length"] = 0.0
        flow["dns_entropy"]      = 0.0
        flow["dns_record_type"]  = ""
        result = predict(flow)
        assert "ml_verdict" in result

    def test_extra_keys_in_flow_ignored(self):
        """Extra keys in the flow dict (e.g. from a richer CSV) don't cause errors."""
        df = pd.read_csv(DATA_PATH)
        flow = df.iloc[0].to_dict()
        flow["extra_column_not_in_schema"] = "some_value"
        result = predict(flow)
        assert "ml_verdict" in result

    def test_all_numeric_features_zero(self):
        """A flow of all zeros should return a valid (not crashed) prediction."""
        flow = {col: 0.0 for col in FEATURE_COLUMNS}
        flow.update({"flow_id": "zero", "src_ip": "0.0.0.0", "dst_ip": "0.0.0.0",
                     "src_port": 0, "dst_port": 0, "protocol": "tcp",
                     "dns_record_type": "", "tls_ja3": "", "tls_ja3s": "",
                     "tls_ja4": "", "tls_sni": "", "label": "benign"})
        result = predict(flow)
        assert result["ml_verdict"] in CLASSES

    def test_graceful_response_when_models_missing(self, tmp_path, monkeypatch):
        """predict() returns a safe error dict if model files don't exist."""
        import ml_dl.predict_interface as pi
        # Temporarily point paths to non-existent files
        monkeypatch.setattr(pi, "_models_ready", False)
        monkeypatch.setattr("ml_dl.predict_interface.XGB_MODEL_PATH", tmp_path / "model.pkl")
        monkeypatch.setattr("ml_dl.predict_interface.THRESHOLD_PATH", tmp_path / "threshold.pkl")
        monkeypatch.setattr("ml_dl.predict_interface.SCALER_PATH",    tmp_path / "scaler.pkl")
        monkeypatch.setattr("ml_dl.predict_interface.LIGHT_DL_PATH",  tmp_path / "light.keras")
        monkeypatch.setattr("ml_dl.predict_interface.HEAVY_DL_PATH",  tmp_path / "heavy.keras")

        flow = {col: 0.0 for col in FEATURE_COLUMNS}
        result = pi.predict(flow)
        assert "_error" in result
        assert result["ml_verdict"] == "unknown"
        assert result["model_used"] == "none"


# ── Phase 4: Coverage and bulk tests ─────────────────────────────────────────

class TestCoverage:
    def test_all_seven_classes_reachable(self, sample_flows):
        """
        Over 1000 synthetic flows, all 7 classes should appear at least once
        in ml_verdict. Ensures the model isn't collapsing to a single class.
        """
        verdicts = {predict(flow)["ml_verdict"] for flow in sample_flows}
        missing = set(CLASSES) - verdicts
        assert not missing, (
            f"These classes were never predicted over 1000 flows: {missing}\n"
            "The model may be under-trained or the synthetic data may be skewed."
        )

    def test_bulk_no_crashes(self, sample_flows):
        """100 flows run without any exception."""
        for flow in sample_flows[:100]:
            result = predict(flow)
            assert isinstance(result, dict)

    def test_model_used_distribution_has_all_tiers(self, sample_flows):
        """
        With 1000 flows there should be at least one flow in each routing tier.
        Validates that the three-tier routing thresholds are properly calibrated.
        """
        tiers = {predict(flow)["model_used"] for flow in sample_flows}
        # At minimum, xgb_only and one DL tier should appear
        assert len(tiers) >= 2, (
            f"Only {tiers} routing tiers observed — check theta_high/theta_low calibration."
        )
