"""
The single entry point Person 3's pipeline orchestrator calls per flow.
See docs/interfaces.md section 2 for the exact contract -- the return
dict's keys/types are locked; don't rename anything here without
telling Person 3 first.

Internally this keeps a small rolling per-src_ip history buffer so
Heavy DL (a sequence model) can be fed real recent context even though
the public contract only takes one flow dict at a time. Person 3 can
build the same kind of buffer upstream if they'd rather manage host
state themselves -- this module works either way, since a host with no
prior history just gets its single flow front-padded (see
data_utils.build_sequences for the same padding rule used at training
time).

State retention policy (Part 9): each host's own sequence buffer is
already bounded (a maxlen=SEQ_LEN deque -- old flows fall off
automatically). What was NOT bounded before is the *number of hosts*
tracked at all -- a long-running deployment seeing many distinct
src_ips would grow _history without limit. _HostHistory below adds an
LRU cap (MAX_TRACKED_HOSTS) on top of the existing deque bound: the
least-recently-updated host is evicted first once the cap is hit. This
only affects Heavy DL's context for a host that hasn't been seen in a
very long time (it starts fresh, same as a brand-new host) -- it does
not affect ml_verdict/dl_verdict for the current flow.

Read-only / passive by construction (section 23): this module only
reads the `flow` dict it's given and returns a verdict. It never opens
a socket, sends a packet, or calls out anywhere -- there is nothing in
here that could touch the monitored network, active-probe a host, or
depend on a return path.
"""
from collections import deque
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

from ml_dl.config import (
    CLASSES, FEATURE_COLUMNS, XGB_MODEL_PATH, SCALER_PATH, THRESHOLD_PATH,
    CALIBRATOR_PATH, SEQ_LEN,
)
from ml_dl.routing import route
from ml_dl import light_dl, heavy_dl
from ml_dl.explainability import top_features
from ml_dl.data_utils import add_engineered_features
from ml_dl.host_history import _HostHistory, DEFAULT_MAX_HOSTS

_xgb_model = None
_calibrator = None  # optional; None if ml_dl.calibration hasn't been run yet
_scaler = None
_theta = None
_light_model = None
_heavy_model = None

# Cap on distinct src_ips tracked at once (Part 9: "do not maintain
# unbounded dictionaries"). Well above SEQ_LEN*expected concurrent
# hosts for a prototype/demo scale; tune for a real deployment.
MAX_TRACKED_HOSTS = DEFAULT_MAX_HOSTS

_history = _HostHistory(max_hosts=MAX_TRACKED_HOSTS, seq_len=SEQ_LEN)


def _lazy_load():
    global _xgb_model, _calibrator, _scaler, _theta, _light_model, _heavy_model
    if _xgb_model is not None:
        return
    try:
        with open(XGB_MODEL_PATH, "rb") as f:
            _xgb_model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            _scaler = pickle.load(f)
        with open(THRESHOLD_PATH, "rb") as f:
            _theta = pickle.load(f)
        _light_model = light_dl.load()
        _heavy_model = heavy_dl.load()
        if CALIBRATOR_PATH.exists():
            with open(CALIBRATOR_PATH, "rb") as f:
                _calibrator = pickle.load(f)
    except FileNotFoundError as e:
        _xgb_model = None  # allow a retry after the missing file is fixed
        raise FileNotFoundError(
            f"predict() cannot load a required model artifact: {e}. "
            "Run `python -m ml_dl.train_all` first to train and save "
            "model.pkl / scaler.pkl / threshold.pkl / the DL weights "
            "(calibrator.pkl is optional -- predict() falls back to "
            "uncalibrated XGBoost confidence if it's missing)."
        ) from e


def reset_history():
    """Clear per-host sequence buffers. Mainly useful for tests."""
    _history.clear()


def _sequence_for(host: str) -> np.ndarray:
    """Build the (SEQ_LEN, n_features) window for Heavy DL from this
    host's rolling buffer (front-padded if history is short -- same
    rule used at training time in data_utils.build_sequences)."""
    window = _history.window(host)
    if len(window) < SEQ_LEN:
        window = [window[0]] * (SEQ_LEN - len(window)) + window
    return np.stack(window)


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
        "dl_verdict":    str,    # one of the 7 CLASSES
        "dl_confidence": float,  # 0.0–1.0
        "model_used":    str,    # "light" | "heavy"
        "shap_evidence": list[{"feature": str, "value": float}]
    }
    """
    _lazy_load()

    # Engineer the same DNS/TLS-derived features used at training time
    # from this flow's raw columns before slicing FEATURE_COLUMNS.
    row = add_engineered_features(pd.DataFrame([flow]))[FEATURE_COLUMNS]

    # --- Stage 1: XGBoost triage (unscaled -- trees don't need it) ---
    if _calibrator is not None:
        ml_probs = _calibrator.predict_proba(row)[0]
    else:
        ml_probs = _xgb_model.predict_proba(row)[0]
    ml_class_idx = int(ml_probs.argmax())
    ml_verdict = CLASSES[ml_class_idx]
    ml_confidence = float(ml_probs.max())

    # --- Stage 2: risk-adaptive routing ---
    model_used = route(ml_confidence, _theta)

    scaled_row = _scaler.transform(row.values)[0]
    host = flow.get("src_ip", "unknown")
    # Every flow updates history (regardless of routing), so that if a
    # later flow from this host *is* routed to Heavy DL, its sequence
    # reflects real recent traffic instead of only the heavy-routed
    # subset. State is updated in the order predict() is called, i.e.
    # streaming order -- there is no batching or shuffling here.
    _history.append(host, scaled_row)

    if model_used == "light":
        dl_probs = _light_model.predict(scaled_row.reshape(1, -1), verbose=0)[0]
    else:
        seq = _sequence_for(host)
        dl_probs = _heavy_model.predict(seq[np.newaxis, ...], verbose=0)[0]

    dl_class_idx = int(dl_probs.argmax())
    dl_verdict = CLASSES[dl_class_idx]
    dl_confidence = float(dl_probs.max())

    # --- Stage 3: explainability (always against the raw XGBoost model;
    # SHAP needs the underlying tree structure, not the calibrator).
    # This explains the XGBoost TRIAGE verdict, not necessarily
    # dl_verdict above -- see each item's "evidence_source" field and
    # ml_dl/explainability.py's docstring (Part 7). ---
    shap_evidence = top_features(_xgb_model, row)

    return {
        "ml_verdict":    ml_verdict,
        "ml_confidence": round(ml_confidence, 4),
        "dl_verdict":    dl_verdict,
        "dl_confidence": round(dl_confidence, 4),
        "model_used":    model_used,
        "shap_evidence": shap_evidence,
    }
