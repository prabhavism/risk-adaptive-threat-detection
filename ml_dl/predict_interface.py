"""
The single entry point Person 3's pipeline orchestrator calls per flow.
See docs/interfaces.md section 2 for the exact contract.
"""
import pickle
import pandas as pd

from ml_dl.config import (
    CLASSES, FEATURE_COLUMNS, XGB_MODEL_PATH, THRESHOLD_PATH,
)
from ml_dl.routing import route
from ml_dl import light_dl, heavy_dl
from ml_dl.explainability import top_features

_xgb_model = None
_theta = None
_light_model = None
_heavy_model = None


def _lazy_load():
    global _xgb_model, _theta, _light_model, _heavy_model
    if _xgb_model is None:
        with open(XGB_MODEL_PATH, "rb") as f:
            _xgb_model = pickle.load(f)
        with open(THRESHOLD_PATH, "rb") as f:
            _theta = pickle.load(f)
        _light_model = light_dl.load()
        _heavy_model = heavy_dl.load()


def predict(flow: dict) -> dict:
    _lazy_load()

    row = pd.DataFrame([{c: flow[c] for c in FEATURE_COLUMNS}])

    ml_probs = _xgb_model.predict_proba(row)[0]
    ml_class_idx = int(ml_probs.argmax())
    ml_verdict = CLASSES[ml_class_idx]
    ml_confidence = float(ml_probs.max())

    model_used = route(ml_confidence, _theta)
    dl_model = _light_model if model_used == "light" else _heavy_model

    dl_probs = dl_model.predict(row.values, verbose=0)[0]
    dl_class_idx = int(dl_probs.argmax())
    dl_verdict = CLASSES[dl_class_idx]
    dl_confidence = float(dl_probs.max())

    shap_evidence = top_features(_xgb_model, row)

    return {
        "ml_verdict": ml_verdict,
        "ml_confidence": ml_confidence,
        "dl_verdict": dl_verdict,
        "dl_confidence": dl_confidence,
        "model_used": model_used,
        "shap_evidence": shap_evidence,
    }
