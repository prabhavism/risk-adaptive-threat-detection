"""
Standardized alert schema (section 14). Turns one predict_interface.predict()
result + the originating flow dict into the JSON-serializable alert
Person 3's correlation/dashboard layer consumes.

This sits ON TOP OF predict_interface.predict() -- it does not change
that function's locked return contract (docs/interfaces.md). Person 3
is free to build their own alert layer instead; this is provided so
the streaming replay demo (scripts/replay_stream.py) has something
concrete to emit.
"""
from __future__ import annotations

import datetime as _dt

import pandas as pd

from ml_dl.config import SEVERITY_THRESHOLDS, SEVERITY_DEFAULT
from ml_dl.data_utils import add_engineered_features


def _severity_for(confidence: float, verdict: str) -> str:
    if verdict == "benign":
        return "INFO"
    for threshold, label in SEVERITY_THRESHOLDS:
        if confidence >= threshold:
            return label
    return SEVERITY_DEFAULT


def build_alert(flow: dict, predict_result: dict, received_at: str | None = None) -> dict:
    """
    flow: the raw flow dict passed to predict()
    predict_result: the dict returned by predict_interface.predict(flow)

    Returns a standardized alert dict. `evidence` reshapes
    predict_result["shap_evidence"] to include both the flow's actual
    feature value and its SHAP contribution, matching section 14's
    example schema -- this doesn't fabricate new information, it's
    just `flow[feature]` joined with the contribution already computed
    by ml_dl/explainability.py.
    """
    verdict = predict_result["dl_verdict"]
    confidence = predict_result["dl_confidence"]

    # Re-derive the engineered (has_dns / has_tls / ...) values too, so
    # evidence for those features shows a real value, not None.
    engineered_row = add_engineered_features(pd.DataFrame([flow])).iloc[0].to_dict()

    evidence = [
        {
            "feature": e["feature"],
            "value": engineered_row.get(e["feature"]),
            "contribution": e["value"],
            "evidence_source": e.get("evidence_source", "xgboost_triage"),
        }
        for e in predict_result.get("shap_evidence", [])
    ]

    xgb_verdict = predict_result["ml_verdict"]
    dl_verdict = predict_result["dl_verdict"]
    model_agreement = xgb_verdict == dl_verdict

    return {
        "timestamp": received_at or _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "flow_id": flow.get("flow_id"),
        "source_ip": flow.get("src_ip"),
        "destination_ip": flow.get("dst_ip"),
        "protocol": flow.get("protocol"),
        "threat_class": verdict,
        "confidence": confidence,
        "severity": _severity_for(confidence, verdict),
        "model_used": predict_result["model_used"],
        # Explicit disagreement tracking (Part 8): never hide it when
        # the fast triage model and the DL verifier land on different
        # classes -- the alert consumer (Person 3 / an analyst) should
        # see that directly rather than only the final verdict.
        "model_agreement": model_agreement,
        "xgb_verdict": xgb_verdict,
        "dl_verdict": dl_verdict,
        # Kept for backward compatibility with earlier alert consumers;
        # ml_verdict/ml_confidence mean the same thing as xgb_verdict/
        # predict_result["ml_confidence"] above.
        "ml_verdict": xgb_verdict,
        "ml_confidence": predict_result["ml_confidence"],
        # Simple, transparent risk score: confidence scaled down when
        # the two models disagree (more uncertainty -> lower trust),
        # not scaled by class severity. This is a deliberately simple
        # placeholder -- Person 3 owns real host-level risk scoring /
        # correlation across multiple alerts.
        "risk_score": round(confidence * (1.0 if model_agreement else 0.85), 4),
        "evidence": evidence,
        # See ml_dl/explainability.py: evidence explains the XGBoost
        # triage stage, not necessarily the final dl_verdict above.
        "evidence_source": "xgboost_triage",
    }
