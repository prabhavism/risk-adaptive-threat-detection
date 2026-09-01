"""
The single entry point Person 3's pipeline orchestrator calls per flow.
See docs/interfaces.md §2 for the exact contract.

Three-tier risk-adaptive routing:
    ml_confidence >= theta_high  →  XGBoost verdict is final (skip DL)
    theta_low <= conf < theta_high  →  Light DL verifies
    conf < theta_low                →  Heavy DL for hard / ambiguous flows

All models are lazy-loaded on the first call and cached for the process lifetime.
If models haven't been trained yet, predict() returns a safe error dict rather
than crashing — this lets Person 3 develop and test the pipeline before training
is complete.

Usage:
    from ml_dl.predict_interface import predict
    result = predict(flow_dict)   # flow_dict = one row of flow_features.csv as dict
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from ml_dl.config import (
    CLASSES, FEATURE_COLUMNS,
    XGB_MODEL_PATH, THRESHOLD_PATH, SCALER_PATH,
    LIGHT_DL_PATH, HEAVY_DL_PATH,
    THETA_HIGH, THETA_LOW,
)
from ml_dl.explainability import SHAPExplainer

# ── Module-level model cache (loaded once per process) ───────────────────────
_xgb_payload   = None   # dict {"model": calibrated, "label_encoder_classes": [...]}
_thresholds    = None   # dict {"theta_high": float, "theta_low": float}
_scaler        = None   # StandardScaler
_light_model   = None   # tf.keras.Model
_heavy_model   = None   # tf.keras.Model
_shap_explainer = None  # SHAPExplainer (built once)
_models_ready  = False


def _models_exist() -> bool:
    return (
        XGB_MODEL_PATH.exists()
        and THRESHOLD_PATH.exists()
        and SCALER_PATH.exists()
        and LIGHT_DL_PATH.exists()
        and HEAVY_DL_PATH.exists()
    )


def _lazy_load():
    global _xgb_payload, _thresholds, _scaler
    global _light_model, _heavy_model, _shap_explainer, _models_ready

    if _models_ready:
        return

    if not _models_exist():
        return  # caller checks _models_ready

    with open(XGB_MODEL_PATH, "rb") as f:
        _xgb_payload = pickle.load(f)

    with open(THRESHOLD_PATH, "rb") as f:
        _thresholds = pickle.load(f)

    with open(SCALER_PATH, "rb") as f:
        _scaler = pickle.load(f)

    _light_model = tf.keras.models.load_model(str(LIGHT_DL_PATH))
    _heavy_model = tf.keras.models.load_model(str(HEAVY_DL_PATH))

    _shap_explainer = SHAPExplainer(_xgb_payload["model"])

    _models_ready = True


# ── Public API ────────────────────────────────────────────────────────────────

def predict(flow: dict) -> dict:
    """
    Parameters
    ----------
    flow : dict
        A single row from flow_features.csv as a Python dict.
        Missing FEATURE_COLUMNS values are filled with 0.0.

    Returns
    -------
    dict matching docs/interfaces.md §2:
    {
        "ml_verdict":    str,    # one of the 7 CLASSES
        "ml_confidence": float,  # 0.0–1.0 (calibrated)
        "dl_verdict":    str,    # one of the 7 CLASSES (or same as ml if xgb_only)
        "dl_confidence": float,  # 0.0–1.0 (or same as ml if xgb_only)
        "model_used":    str,    # "xgb_only" | "light" | "heavy"
        "shap_evidence": list[{"feature": str, "value": float}]
    }
    On error (models not trained yet):
    {
        ... same keys with safe defaults ...,
        "_error": str
    }
    """
    _lazy_load()

    if not _models_ready:
        return _not_ready_response()

    # ── Build feature row ─────────────────────────────────────────────────────
    raw = {col: flow.get(col, 0.0) for col in FEATURE_COLUMNS}
    row_df = pd.DataFrame([raw])
    row_scaled = _scaler.transform(row_df.values.astype(float))
    row_scaled_df = pd.DataFrame(row_scaled, columns=FEATURE_COLUMNS)

    # ── XGBoost inference ─────────────────────────────────────────────────────
    xgb_model = _xgb_payload["model"]
    ml_probs      = xgb_model.predict_proba(row_df)[0]   # calibrated, uses raw features
    ml_class_idx  = int(np.argmax(ml_probs))
    ml_verdict    = CLASSES[ml_class_idx]
    ml_confidence = float(ml_probs[ml_class_idx])

    # ── Three-tier routing ────────────────────────────────────────────────────
    theta_high = _thresholds.get("theta_high", THETA_HIGH)
    theta_low  = _thresholds.get("theta_low",  THETA_LOW)

    if ml_confidence >= theta_high:
        # High confidence — skip DL entirely
        model_used    = "xgb_only"
        dl_verdict    = ml_verdict
        dl_confidence = ml_confidence
    else:
        # Route to Light or Heavy DL
        if ml_confidence >= theta_low:
            dl_model   = _light_model
            model_used = "light"
        else:
            dl_model   = _heavy_model
            model_used = "heavy"

        dl_probs      = dl_model.predict(row_scaled, verbose=0)[0]
        dl_class_idx  = int(np.argmax(dl_probs))
        dl_verdict    = CLASSES[dl_class_idx]
        dl_confidence = float(dl_probs[dl_class_idx])

    # ── SHAP evidence (always from XGBoost) ──────────────────────────────────
    shap_evidence = _shap_explainer.top_features(row_df, k=5)

    return {
        "ml_verdict":    ml_verdict,
        "ml_confidence": round(ml_confidence, 4),
        "dl_verdict":    dl_verdict,
        "dl_confidence": round(dl_confidence, 4),
        "model_used":    model_used,
        "shap_evidence": shap_evidence,
    }


def reload_models():
    """Force a model reload (e.g. after retraining). Call from tests or CLI."""
    global _models_ready
    _models_ready = False
    _lazy_load()


# ── Error helper ──────────────────────────────────────────────────────────────

def _not_ready_response() -> dict:
    missing = [
        str(p) for p in [
            XGB_MODEL_PATH, THRESHOLD_PATH, SCALER_PATH,
            LIGHT_DL_PATH, HEAVY_DL_PATH,
        ]
        if not Path(p).exists()
    ]
    return {
        "ml_verdict":    "unknown",
        "ml_confidence": 0.0,
        "dl_verdict":    "unknown",
        "dl_confidence": 0.0,
        "model_used":    "none",
        "shap_evidence": [],
        "_error": (
            "Models not trained yet. Run:\n"
            "  python -m ml_dl.train_xgboost\n"
            "  python scripts/train_dl.py\n"
            f"Missing files: {missing}"
        ),
    }
