"""
Probability calibration for the XGBoost triage confidence (section 11).

Tree ensembles' predict_proba outputs are not automatically well
calibrated -- a flow XGBoost calls "97% confident" isn't necessarily
right 97% of the time. Since ml_confidence directly drives the
risk-adaptive routing threshold (theta), miscalibration can silently
skew how much traffic goes to Light vs Heavy DL.

Uses per-class isotonic regression (monotonic, doesn't assume a
parametric shape -- reasonable default vs. Platt/sigmoid scaling for
tree ensembles) fit on the VALIDATION split only. sklearn >= 1.4
removed CalibratedClassifierCV(cv="prefit"); this _IsotonicCalibrator
wrapper does the same thing directly with IsotonicRegression and
exposes the same predict_proba() interface so predict_interface.py
and tune_threshold.py are unaffected.
"""
import pickle

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import label_binarize

from ml_dl.config import CLASSES, CALIBRATOR_PATH, REPORT_DIR


class _IsotonicCalibrator:
    """
    Multiclass isotonic calibrator for a pre-fitted classifier.

    Fits one IsotonicRegression per class (one-vs-rest) on the
    validation-set raw probabilities, then renormalises each row so
    the outputs sum to 1. Exposes predict_proba() so it is a drop-in
    replacement for CalibratedClassifierCV everywhere it is used.
    """

    def __init__(self, base_model, calibrators, n_classes):
        self._base_model = base_model
        self._calibrators = calibrators  # list of IsotonicRegression, one per class
        self._n_classes = n_classes

    def predict_proba(self, X):
        raw = self._base_model.predict_proba(X)            # (n, n_classes)
        cal = np.zeros_like(raw)
        for c, ir in enumerate(self._calibrators):
            cal[:, c] = ir.predict(raw[:, c])
        # Renormalise rows so they sum to 1 (isotonic mapping can break this)
        row_sums = cal.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return cal / row_sums


def fit_calibrator(model, X_val, y_val):
    """
    model: an already-fit xgboost.XGBClassifier.
    Returns a fitted _IsotonicCalibrator that wraps it (doesn't refit
    the base model -- only learns the isotonic mapping on X_val/y_val).
    """
    n_classes = len(CLASSES)
    raw_probs = model.predict_proba(X_val)           # (n_val, n_classes)
    y_bin = label_binarize(y_val, classes=list(range(n_classes)))
    if y_bin.shape[1] == 1:
        # Binary edge-case from label_binarize: expand to (n, 2)
        y_bin = np.hstack([1 - y_bin, y_bin])

    calibrators = []
    for c in range(n_classes):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(raw_probs[:, c], y_bin[:, c])
        calibrators.append(ir)

    return _IsotonicCalibrator(model, calibrators, n_classes)


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """
    ECE over the model's top-1 confidence: bins predictions by
    confidence, compares each bin's average confidence to its actual
    accuracy, and returns the weighted average gap. Lower is better
    (0 = perfectly calibrated).
    """
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def evaluate_calibration(model, calibrator, X_test, y_test, save_report: bool = True):
    """Compares raw XGBoost ECE vs calibrated ECE on the (untouched)
    test split, so calibration quality is itself leakage-free."""
    raw_probs = model.predict_proba(X_test)
    cal_probs = calibrator.predict_proba(X_test)

    raw_ece = expected_calibration_error(raw_probs, y_test)
    cal_ece = expected_calibration_error(cal_probs, y_test)

    result = {"raw_xgboost_ece": raw_ece, "calibrated_ece": cal_ece,
              "improved": cal_ece <= raw_ece}

    if save_report:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([result]).to_csv(REPORT_DIR / "calibration_report.csv", index=False)

    return result


def save(calibrator, path=CALIBRATOR_PATH):
    with open(path, "wb") as f:
        pickle.dump(calibrator, f)


def load(path=CALIBRATOR_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    import pickle as _pickle
    from ml_dl.config import XGB_MODEL_PATH, DATA_PATH
    from ml_dl.data_utils import load_raw, time_based_split, xy

    with open(XGB_MODEL_PATH, "rb") as f:
        model = _pickle.load(f)

    df = load_raw(DATA_PATH)
    _, val_df, test_df = time_based_split(df)
    X_val, y_val = xy(val_df)
    X_test, y_test = xy(test_df)

    calibrator = fit_calibrator(model, X_val, y_val)
    save(calibrator)
    result = evaluate_calibration(model, calibrator, X_test, y_test)
    print(f"Raw XGBoost ECE:   {result['raw_xgboost_ece']:.4f}")
    print(f"Calibrated ECE:    {result['calibrated_ece']:.4f}")
    print(f"Calibration {'improved' if result['improved'] else 'did NOT improve'} "
          f"top-1 confidence reliability on the test split.")
