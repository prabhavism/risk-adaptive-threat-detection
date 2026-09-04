"""
Baseline comparison (section 12): objectively checks whether the
risk-adaptive Light/Heavy routing is actually worth it, versus three
simpler alternatives, on the same held-out test split:

    Baseline 1: XGBoost only (no DL verification at all)
    Baseline 2: Always Light DL (ignore XGBoost confidence, always fast path)
    Baseline 3: Always Heavy DL (ignore XGBoost confidence, always slow path)
    Proposed:   Risk-Adaptive (XGBoost confidence decides Light vs Heavy)

Reports accuracy, macro F1, malicious recall, malicious->benign rate,
mean per-flow latency, and total inference count for each, so the
trade-off is visible rather than asserted. Numbers are only ever
measured, never invented (section 30) -- run this script to produce
them.

Run (after ml_dl/train_all.py has produced models/*):
    python -m ml_dl.baselines
"""
import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from ml_dl.config import (
    CLASSES, DATA_PATH, XGB_MODEL_PATH, SCALER_PATH, THRESHOLD_PATH,
    REPORT_DIR,
)
from ml_dl.data_utils import load_raw, time_based_split, xy, build_sequences
from ml_dl.routing import route
from ml_dl import light_dl, heavy_dl


def _malicious_metrics(y_true, y_pred):
    benign_idx = CLASSES.index("benign")
    mal_mask = y_true != benign_idx
    if mal_mask.sum() == 0:
        return None, None
    to_benign = ((y_pred == benign_idx) & mal_mask).sum() / mal_mask.sum()
    return float(1.0 - to_benign), float(to_benign)


def _timed_predict(fn, X, n_timing_samples=200):
    """Runs fn over all of X (for accuracy) and separately times a
    subsample one-row-at-a-time (for realistic per-flow latency, not
    batched throughput)."""
    preds = fn(X)
    n = min(n_timing_samples, len(X))
    latencies = []
    for i in range(n):
        row = X[i:i + 1] if isinstance(X, np.ndarray) else X.iloc[i:i + 1]
        t0 = time.perf_counter()
        fn(row)
        latencies.append(time.perf_counter() - t0)
    return preds, latencies


def run_baselines(data_path=DATA_PATH, verbose=True):
    with open(XGB_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(THRESHOLD_PATH, "rb") as f:
        theta = pickle.load(f)
    light_model = light_dl.load()
    heavy_model = heavy_dl.load()

    df = load_raw(data_path)
    _, _, test_df = time_based_split(df)
    X_test, y_test = xy(test_df)
    X_scaled = scaler.transform(X_test)
    X_seq, y_seq_check = build_sequences(test_df)
    assert (y_seq_check == y_test).all()
    n_feat = X_seq.shape[2]
    X_seq_scaled = scaler.transform(X_seq.reshape(-1, n_feat)).reshape(X_seq.shape)

    results = {}

    # --- Baseline 1: XGBoost only ---
    xgb_preds, xgb_lat = _timed_predict(lambda x: model.predict_proba(x).argmax(axis=1), X_test)
    results["xgboost_only"] = _score("XGBoost only", y_test, xgb_preds, xgb_lat, n_dl_calls=0)

    # --- Baseline 2: Always Light DL ---
    light_preds, light_lat = _timed_predict(
        lambda x: light_model.predict(x, verbose=0).argmax(axis=1), X_scaled
    )
    results["always_light"] = _score("Always Light DL", y_test, light_preds, light_lat, n_dl_calls=len(X_test))

    # --- Baseline 3: Always Heavy DL ---
    heavy_preds, heavy_lat = _timed_predict(
        lambda x: heavy_model.predict(x, verbose=0).argmax(axis=1), X_seq_scaled
    )
    results["always_heavy"] = _score("Always Heavy DL", y_test, heavy_preds, heavy_lat, n_dl_calls=len(X_test))

    # --- Proposed: Risk-Adaptive ---
    ml_probs = model.predict_proba(X_test)
    ml_conf = ml_probs.max(axis=1)
    routed = np.array([route(c, theta) for c in ml_conf])
    light_mask = routed == "light"
    adaptive_preds = np.where(light_mask, light_preds, heavy_preds)
    # End-to-end latency per flow = XGBoost triage + whichever DL branch
    # that flow was routed to (weighted by the actual routing mix).
    mean_xgb_lat = np.mean(xgb_lat)
    adaptive_lat = [
        mean_xgb_lat + (np.random.choice(light_lat) if is_light else np.random.choice(heavy_lat))
        for is_light in light_mask[:200]
    ]
    results["risk_adaptive"] = _score(
        "Risk-Adaptive (proposed)", y_test, adaptive_preds, adaptive_lat,
        n_dl_calls=len(X_test),  # every flow still gets exactly one DL call
        pct_light=float(light_mask.mean()), pct_heavy=float((~light_mask).mean()),
    )

    if verbose:
        print(f"\n{'Model':<28}{'Accuracy':>10}{'MacroF1':>10}{'MalRecall':>11}{'Mal->Ben':>10}{'MeanLat(ms)':>13}")
        for r in results.values():
            print(f"{r['name']:<28}{r['accuracy']:>10.4f}{r['macro_f1']:>10.4f}"
                  f"{(r['malicious_recall'] or 0):>11.4f}{(r['malicious_to_benign_rate'] or 0):>10.4f}"
                  f"{r['mean_latency_ms']:>13.3f}")
        heavy_saving = 1 - (results["risk_adaptive"]["pct_heavy"])
        print(f"\nRisk-adaptive routes {results['risk_adaptive']['pct_light']:.1%} of "
              f"test flows to Light DL and {results['risk_adaptive']['pct_heavy']:.1%} to "
              f"Heavy DL -- i.e. {heavy_saving:.1%} fewer Heavy DL calls than "
              f"'Always Heavy', for "
              f"{results['risk_adaptive']['accuracy'] - results['xgboost_only']['accuracy']:+.4f} "
              f"accuracy vs. XGBoost-only.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results.values()).to_csv(REPORT_DIR / "baseline_comparison.csv", index=False)
    with open(REPORT_DIR / "baseline_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def _score(name, y_true, y_pred, latencies, n_dl_calls, pct_light=None, pct_heavy=None):
    malicious_recall, malicious_to_benign_rate = _malicious_metrics(y_true, y_pred)
    return {
        "name": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "malicious_recall": malicious_recall,
        "malicious_to_benign_rate": malicious_to_benign_rate,
        "mean_latency_ms": float(np.mean(latencies) * 1000.0),
        "n_dl_calls": n_dl_calls,
        "pct_light": pct_light,
        "pct_heavy": pct_heavy,
    }


if __name__ == "__main__":
    run_baselines()
