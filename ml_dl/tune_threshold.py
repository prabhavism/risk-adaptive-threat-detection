"""
Threshold tuning: sweep candidate theta values on the validation split
and measure, for each one:
  - what fraction of flows get routed light vs heavy (throughput impact)
  - XGBoost-alone accuracy on each branch (why the branches differ)
  - final accuracy after DL verification on each branch (does Light DL
    on the "easy" branch hold up, and does Heavy DL actually rescue the
    "hard" branch XGBoost was unsure about?)

Must run AFTER train_xgboost.py, light_dl training and heavy_dl
training have all produced saved artifacts, since this needs real
Light/Heavy predictions, not just XGBoost confidence, to pick a theta
that reflects what the deployed system will actually do.

Picks the theta that maximises overall (light+heavy combined) DL
accuracy on validation, subject to a max-heavy-fraction cap so the
"risk-adaptive" part of the design still means something -- routing
everything to Heavy DL would trivially maximise accuracy but defeat
the throughput goal.
"""
import pickle

import numpy as np
import pandas as pd

from ml_dl.config import (
    CLASSES, THRESHOLD_PATH, ROUTING_STATS_PATH,
)
from ml_dl.routing import routing_stats
from ml_dl.data_utils import build_sequences


MAX_HEAVY_FRACTION = 0.40  # throughput guardrail; tune to your target

# Section 2 of the calibration/threshold brief asks for routing stats
# reported at these specific reference thresholds; make sure they're
# always present in the sweep (not just landed on by the 0.02 step
# grid, which skips e.g. 0.75/0.95) in addition to the finer grid used
# to actually pick theta.
REFERENCE_THETAS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def _dl_predict_all(light_model, heavy_model, X_scaled, X_seq):
    light_preds = light_model.predict(X_scaled, verbose=0).argmax(axis=1)
    heavy_preds = heavy_model.predict(X_seq, verbose=0).argmax(axis=1)
    return light_preds, heavy_preds


def tune(xgb_model, scaler, light_model, heavy_model, val_df,
         thetas=None, max_heavy_fraction: float = MAX_HEAVY_FRACTION,
         verbose: bool = True, calibrator=None):
    """
    calibrator: optional, an already-fit probability calibrator (see
    ml_dl/calibration.py). If given, ALL confidence/routing decisions
    here use calibrated probabilities -- this must be the same
    calibrator predict_interface.py will use at deployment time, or
    the saved theta won't mean what it was tuned to mean (this was a
    real bug: theta used to be tuned on raw XGBoost probabilities while
    deployment routed on calibrated ones). If calibrator is None,
    falls back to raw probabilities (e.g. for a quick uncalibrated
    sanity check) -- callers doing a real run should always pass one.
    """
    if thetas is None:
        thetas = sorted(set(np.round(np.arange(0.50, 0.99, 0.02), 2).tolist()) | set(REFERENCE_THETAS))

    from ml_dl.data_utils import xy
    X_val, y_val = xy(val_df)
    X_val_scaled = scaler.transform(X_val)
    X_val_seq, y_seq_check = build_sequences(val_df)
    assert (y_seq_check == y_val).all(), "sequence labels out of order with xy()"

    ml_probs = (calibrator if calibrator is not None else xgb_model).predict_proba(X_val)
    ml_confidences = ml_probs.max(axis=1)

    # Pre-compute DL predictions ONCE for every val row under both
    # models, then just re-slice per theta -- avoids re-running
    # inference len(thetas) times.
    light_preds, heavy_preds = _dl_predict_all(
        light_model, heavy_model, X_val_scaled, X_val_seq
    )

    rows = []
    for theta in thetas:
        stats = routing_stats(ml_confidences, theta)
        light_mask = ml_confidences >= theta
        heavy_mask = ~light_mask

        final_preds = np.where(light_mask, light_preds, heavy_preds)
        overall_acc = (final_preds == y_val).mean()

        light_acc = (light_preds[light_mask] == y_val[light_mask]).mean() \
            if light_mask.sum() else float("nan")
        heavy_acc = (heavy_preds[heavy_mask] == y_val[heavy_mask]).mean() \
            if heavy_mask.sum() else float("nan")
        xgb_only_acc_heavy_branch = (
            ml_probs[heavy_mask].argmax(axis=1) == y_val[heavy_mask]
        ).mean() if heavy_mask.sum() else float("nan")

        rows.append({
            "theta": theta,
            "pct_light": stats["pct_light"],
            "pct_heavy": stats["pct_heavy"],
            "overall_dl_accuracy": overall_acc,
            "light_branch_accuracy": light_acc,
            "heavy_branch_accuracy": heavy_acc,
            "heavy_branch_xgb_only_accuracy": xgb_only_acc_heavy_branch,
        })

    results = pd.DataFrame(rows)

    # Fixed-threshold comparison baselines (Part 2 requirement): 100%
    # Light (theta so low everything qualifies) and 100% Heavy (theta
    # so high nothing qualifies), computed directly rather than reading
    # them off the sweep's edge rows.
    all_light_acc = (light_preds == y_val).mean()
    all_heavy_acc = (heavy_preds == y_val).mean()

    feasible = results[results["pct_heavy"] <= max_heavy_fraction]
    candidates = feasible if len(feasible) else results
    best_row = candidates.loc[candidates["overall_dl_accuracy"].idxmax()]
    best_theta = float(best_row["theta"])

    summary = {
        "selected_theta": best_theta,
        "validation_overall_dl_accuracy": float(best_row["overall_dl_accuracy"]),
        "light_routing_fraction": float(best_row["pct_light"]),
        "heavy_routing_fraction": float(best_row["pct_heavy"]),
        "light_branch_accuracy": float(best_row["light_branch_accuracy"]),
        "heavy_branch_accuracy": float(best_row["heavy_branch_accuracy"]),
        "fixed_100pct_light_accuracy": float(all_light_acc),
        "fixed_100pct_heavy_accuracy": float(all_heavy_acc),
        "used_calibrated_probabilities": calibrator is not None,
    }

    if verbose:
        pd.set_option("display.width", 120)
        print("\n=== Threshold sweep (validation split"
              f"{', CALIBRATED probabilities' if calibrator is not None else ', RAW XGBoost probabilities -- pass a calibrator for a deployment-accurate sweep'}) ===")
        print(results.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print(f"\nSelected theta = {best_theta:.2f} "
              f"(heavy-branch cap: {max_heavy_fraction:.0%})")
        print(f"  Validation overall DL accuracy:  {summary['validation_overall_dl_accuracy']:.3f}")
        print(f"  Light routing fraction:          {summary['light_routing_fraction']:.3f}")
        print(f"  Heavy routing fraction:          {summary['heavy_routing_fraction']:.3f}")
        print(f"  Light branch accuracy:           {summary['light_branch_accuracy']:.3f}")
        print(f"  Heavy branch accuracy:           {summary['heavy_branch_accuracy']:.3f}")
        print(f"  vs. fixed 100% Light accuracy:   {all_light_acc:.3f}")
        print(f"  vs. fixed 100% Heavy accuracy:   {all_heavy_acc:.3f}")

    with open(THRESHOLD_PATH, "wb") as f:
        pickle.dump(best_theta, f)
    with open(ROUTING_STATS_PATH, "wb") as f:
        pickle.dump(results, f)

    return best_theta, results, summary


if __name__ == "__main__":
    import pickle as _pickle
    from ml_dl.config import XGB_MODEL_PATH, SCALER_PATH, CALIBRATOR_PATH, DATA_PATH
    from ml_dl.data_utils import load_raw, time_based_split
    from ml_dl import light_dl, heavy_dl

    with open(XGB_MODEL_PATH, "rb") as f:
        xgb_model = _pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = _pickle.load(f)
    light_model = light_dl.load()
    heavy_model = heavy_dl.load()

    calibrator = None
    if CALIBRATOR_PATH.exists():
        with open(CALIBRATOR_PATH, "rb") as f:
            calibrator = _pickle.load(f)
    else:
        print("WARNING: no calibrator found at", CALIBRATOR_PATH,
              "-- run `python -m ml_dl.calibration` first for a "
              "deployment-accurate threshold sweep. Falling back to "
              "raw XGBoost probabilities for this standalone run.")

    df = load_raw(DATA_PATH)
    _, val_df, _ = time_based_split(df)

    tune(xgb_model, scaler, light_model, heavy_model, val_df, calibrator=calibrator)
