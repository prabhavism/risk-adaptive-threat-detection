"""
SHAP-based explainability for the XGBoost triage model. Returns the
top-k contributing features for a single flow's prediction -- this is
the `shap_evidence` field in docs/interfaces.md's predict() contract,
which Person 3's dashboard renders when an alert is clicked.
"""
import numpy as np
import pandas as pd
import shap

from ml_dl.config import FEATURE_COLUMNS

_explainer_cache = {}


def _get_explainer(model):
    # TreeExplainer build cost is non-trivial; reuse it across calls
    # for the same model object instead of rebuilding per-flow.
    key = id(model)
    if key not in _explainer_cache:
        _explainer_cache[key] = shap.TreeExplainer(model)
    return _explainer_cache[key]


def top_features(model, flow_row, k: int = 5):
    """
    model: trained xgboost.XGBClassifier
    flow_row: 1-row DataFrame (or array) with FEATURE_COLUMNS, in the
              same column order train_xgboost.py used.
    Returns: list of {"feature": name, "value": shap_value}, sorted by
             |value| descending -- the features that most pushed the
             prediction toward (positive) or away from (negative) the
             predicted class.
    """
    if not isinstance(flow_row, pd.DataFrame):
        flow_row = pd.DataFrame([flow_row], columns=FEATURE_COLUMNS)

    explainer = _get_explainer(model)
    shap_values = explainer.shap_values(flow_row)

    # shap_values shape varies across shap/xgboost versions:
    #   shap < 0.44  + xgb 1.x  -> list of (n_samples, n_features) arrays, len = n_classes
    #   shap >= 0.44 + xgb 2.x  -> (n_samples, n_features)  [already summed/selected]
    #   some builds              -> (n_classes, n_samples, n_features) ndarray
    # Normalise everything to a single 1-D row of length n_features.
    n_features = len(FEATURE_COLUMNS)

    if isinstance(shap_values, list):
        # Old-style list: one (n_samples, n_features) array per class.
        predicted_class = int(np.argmax(model.predict_proba(flow_row)[0]))
        predicted_class = min(predicted_class, len(shap_values) - 1)
        row = np.array(shap_values[predicted_class])[0]
    else:
        values = np.array(shap_values)
        if values.ndim == 3:
            # (n_classes, n_samples, n_features)
            predicted_class = int(np.argmax(model.predict_proba(flow_row)[0]))
            predicted_class = min(predicted_class, values.shape[0] - 1)
            row = values[predicted_class][0]
        elif values.ndim == 2 and values.shape[1] == n_features:
            # (n_samples, n_features) — newer shap, already for the predicted class
            row = values[0]
        elif values.ndim == 2 and values.shape[0] == n_features:
            # (n_features, n_samples) transposed edge case
            row = values[:, 0]
        else:
            # Fallback: just take the first flat row we can find
            row = values.ravel()[:n_features]

    ranked = sorted(
        zip(FEATURE_COLUMNS, row), key=lambda x: abs(x[1]), reverse=True
    )
    # evidence_source is explicit and load-bearing (see docs/interfaces.md
    # section 2 / Part 7 of the ingestion-integration brief): SHAP here
    # explains the XGBoost TRIAGE model only. The final verdict returned
    # to Person 3 (`dl_verdict`) can come from Light or Heavy DL instead,
    # and neither has a SHAP explainer wired up -- so this evidence must
    # never be presented as if it explains the DL verdict, especially
    # when ml_verdict != dl_verdict. predict_interface.py and alerts.py
    # both surface this field so a consumer can't lose track of it.
    return [
        {"feature": f, "value": float(v), "evidence_source": "xgboost_triage"}
        for f, v in ranked[:k]
    ]


def test_explainability(model, X_sample: pd.DataFrame, n: int = 3):
    """
    Smoke test used by tests/ and manual runs: make sure SHAP returns
    sane, non-degenerate evidence for a handful of real rows rather
    than silently throwing or returning all-zero contributions.
    """
    results = []
    for i in range(min(n, len(X_sample))):
        row = X_sample.iloc[[i]]
        evidence = top_features(model, row)
        assert len(evidence) > 0, "SHAP returned no evidence"
        assert any(abs(e["value"]) > 1e-9 for e in evidence), (
            "SHAP evidence is degenerate (all ~0) -- check feature scaling "
            "or that the model was actually fit."
        )
        results.append(evidence)
    return results
