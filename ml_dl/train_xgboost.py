"""
Trains the 7-class XGBoost triage classifier.

Split:  60% train / 20% val (early-stopping + threshold tuning) / 20% test (held-out report)
Output: models/model.pkl     — CalibratedXGB wrapping XGBClassifier
        models/threshold.pkl  — dict {"theta_high": float, "theta_low": float}

Works today against synthetic data from scripts/generate_synthetic_data.py.
When Person 1's real data/flow_features.csv lands, re-run unchanged.
"""
import pickle
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from ml_dl.config import (
    CLASSES, FEATURE_COLUMNS, DATA_PATH,
    XGB_MODEL_PATH, THRESHOLD_PATH, MODEL_DIR,
    TRAIN_RATIO, VAL_RATIO, THETA_HIGH, THETA_LOW,
)

warnings.filterwarnings("ignore", category=UserWarning)

TEST_RATIO = round(1.0 - TRAIN_RATIO - VAL_RATIO, 10)


# ── Probability calibration ───────────────────────────────────────────────────

class CalibratedXGB:
    """
    Thin isotonic-regression calibration wrapper around XGBClassifier.
    Replaces sklearn's CalibratedClassifierCV(cv='prefit') which was removed
    in scikit-learn >= 1.3.

    One IsotonicRegression is fitted per class on the val-set raw probabilities,
    then outputs are row-normalised so they sum to 1.
    """

    def __init__(self, xgb_model: xgb.XGBClassifier,
                 X_val: pd.DataFrame, y_val: np.ndarray):
        self.xgb_model   = xgb_model
        self.n_classes   = len(CLASSES)
        raw_probs        = xgb_model.predict_proba(X_val)        # (n, 7)

        self._calibrators = []
        for c in range(self.n_classes):
            y_binary = (y_val == c).astype(int)
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(raw_probs[:, c], y_binary)
            self._calibrators.append(ir)

    def predict_proba(self, X) -> np.ndarray:
        raw = self.xgb_model.predict_proba(X)                    # (n, 7)
        cal = np.column_stack([
            self._calibrators[c].predict(raw[:, c])
            for c in range(self.n_classes)
        ])
        # Normalise rows to sum to 1 (isotonic regression per-class doesn't guarantee this)
        row_sums = cal.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return cal / row_sums

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    X  = df[FEATURE_COLUMNS].fillna(0.0).astype(float)
    label_map = {c: i for i, c in enumerate(CLASSES)}
    y = df["label"].map(label_map)
    if y.isna().any():
        unknown = df.loc[y.isna(), "label"].unique()
        raise ValueError(f"Unknown labels in dataset: {unknown}")
    return X, y.astype(int)


def three_way_split(X, y):
    """60 / 20 / 20 split stratified by label. Same seed everywhere."""
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, stratify=y, random_state=42
    )
    val_relative = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)   # 0.20/0.80 = 0.25
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_relative, stratify=y_trainval, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Loading data …")
    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(X, y)
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # ── XGBoost base model ────────────────────────────────────────────────────
    print("\nTraining XGBoost …")
    base_model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(CLASSES),
        n_estimators=400,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        early_stopping_rounds=25,
        random_state=42,
        verbosity=0,
    )
    base_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    print(f"  Best iteration: {base_model.best_iteration}")

    # ── Probability calibration (isotonic on val set) ─────────────────────────
    print("Calibrating probabilities (isotonic regression per class) …")
    calibrated = CalibratedXGB(base_model, X_val, y_val.values)

    # ── Evaluate on held-out test set ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    test_probs = calibrated.predict_proba(X_test)
    test_preds = test_probs.argmax(axis=1)
    test_conf  = test_probs.max(axis=1)

    print(classification_report(
        y_test, test_preds, target_names=CLASSES, zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(y_test, test_preds)
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print("         " + "  ".join(f"{c[:8]:>8}" for c in CLASSES))
    for i, row in enumerate(cm):
        print(f"{CLASSES[i][:8]:>8} |" + "  ".join(f"{v:>8}" for v in row))

    # Benign false-positive rate
    benign_idx  = CLASSES.index("benign")
    benign_mask = (y_test.values == benign_idx)
    fp  = ((test_preds != benign_idx) & benign_mask).sum()
    tn  = ((test_preds == benign_idx) & benign_mask).sum()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    print(f"\nBenign False-Positive Rate (FPR): {fpr:.4f}  "
          f"({fp} misclassified / {fp+tn} benign flows)")

    # ROC-AUC (one-vs-rest)
    try:
        auc = roc_auc_score(
            y_test, test_probs, multi_class="ovr", average="macro"
        )
        print(f"Macro ROC-AUC (OvR):              {auc:.4f}")
    except Exception:
        pass

    # ── Threshold sweep ───────────────────────────────────────────────────────
    thresholds = sweep_threshold(test_conf, test_preds, y_test.values)
    theta_high = thresholds["theta_high"]
    theta_low  = thresholds["theta_low"]
    print(f"\nRouting thresholds -> theta_high={theta_high:.3f}  "
          f"theta_low={theta_low:.3f}")
    _print_routing_stats(test_conf, theta_high, theta_low)

    # ── Save ──────────────────────────────────────────────────────────────────
    payload = {
        "model": calibrated,
        "label_encoder_classes": CLASSES,
    }
    with open(XGB_MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
    with open(THRESHOLD_PATH, "wb") as f:
        pickle.dump(thresholds, f)

    print(f"\nSaved model      -> {XGB_MODEL_PATH}")
    print(f"Saved thresholds -> {THRESHOLD_PATH}")
    print("=" * 60)


# ── Threshold sweep ───────────────────────────────────────────────────────────

def sweep_threshold(confidences, preds, truth, candidates=None):
    """
    Find theta_high: smallest threshold where accuracy on the high-confidence
    slice is >= 95% AND coverage >= 20%.
    theta_low is fixed at THETA_LOW (0.60).
    """
    if candidates is None:
        candidates = np.arange(0.50, 0.99, 0.005)

    correct = (preds == truth)
    best_theta_high = THETA_HIGH  # fallback

    for t in sorted(candidates, reverse=True):
        mask = confidences >= t
        if mask.sum() < 10:
            continue
        acc = correct[mask].mean()
        cov = mask.mean()
        if acc >= 0.95 and cov >= 0.20:
            best_theta_high = float(t)
            break

    return {"theta_high": best_theta_high, "theta_low": THETA_LOW}


def _print_routing_stats(confidences, theta_high, theta_low):
    xgb_only = (confidences >= theta_high).mean()
    light    = ((confidences >= theta_low) & (confidences < theta_high)).mean()
    heavy    = (confidences < theta_low).mean()
    print(f"  XGBoost-only (skip DL): {xgb_only:.1%}")
    print(f"  -> Light DL:            {light:.1%}")
    print(f"  -> Heavy DL:            {heavy:.1%}")


if __name__ == "__main__":
    train()
