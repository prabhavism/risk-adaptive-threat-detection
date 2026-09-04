"""
Trains the 7-class XGBoost triage classifier.

Pipeline:
  1. Load flow_features.csv, time-based split (70/15/15 -- see
     data_utils.time_based_split), NOT a random split.
  2. Report class distribution, compute balanced sample weights
     (section 6 -- class imbalance).
  3. Small, bounded hyperparameter search (config.XGB_PARAM_GRID),
     model-selected on the VALIDATION split only (section 7) -- the
     test split is never touched until final reporting.
  4. Fit a StandardScaler on train features -- saved to models/scaler.pkl
     for Light/Heavy DL, which are scale-sensitive (trees aren't, so
     XGBoost itself trains on the raw, unscaled features).
  5. Evaluate the selected model on the held-out test split:
     classification report (precision/recall/F1 per class), confusion
     matrix (CSV + PNG), malicious recall, malicious->benign rate.
  6. Save model, scaler, and reports/ artifacts.

Threshold tuning is a separate, later step (ml_dl/tune_threshold.py)
because it needs Light/Heavy DL trained first to measure whether DL
verification actually helps.

Works today against scripts/generate_synthetic_data.py output. When
Person 1's real data/flow_features.csv lands, just re-run this script
unchanged -- same schema, same columns.
"""
import json
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from ml_dl.config import (
    CLASSES, MALICIOUS_CLASSES, DATA_PATH, XGB_MODEL_PATH, SCALER_PATH,
    MODEL_DIR, REPORT_DIR, SEED, CLASS_WEIGHT_STRATEGY, XGB_PARAM_GRID,
)
from ml_dl.data_utils import load_raw, time_based_split, xy, class_distribution


# ── Evaluation helpers ──────────────────────────────────────────────────────

def _malicious_metrics(y_true, y_pred):
    """
    Malicious recall: of all truly-malicious flows (any non-benign
    class), what fraction did we correctly flag as malicious (i.e. NOT
    predicted benign)? This matters more for a security system than
    plain accuracy, per section 20.

    Malicious->benign rate: the complement -- how often malicious
    traffic slips through labelled benign. This is the number an
    analyst should care about most.
    """
    benign_idx = CLASSES.index("benign")
    mal_mask = y_true != benign_idx
    if mal_mask.sum() == 0:
        return None, None
    predicted_benign = y_pred == benign_idx
    malicious_to_benign = (predicted_benign & mal_mask).sum() / mal_mask.sum()
    malicious_recall = 1.0 - malicious_to_benign
    return float(malicious_recall), float(malicious_to_benign)


def _fit_candidate(params, X_train, y_train, X_val, y_val, sample_weight):
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(CLASSES),
        eval_metric="mlogloss",
        early_stopping_rounds=25,
        random_state=SEED,
        **params,
    )
    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    val_preds = model.predict(X_val)
    # Macro F1 (not accuracy) as the model-selection criterion: with
    # balanced classes this rarely flips the winner, but it keeps
    # selection honest if the split ends up skewed.
    val_macro_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
    return model, val_macro_f1


# ── Training ─────────────────────────────────────────────────────────────────

def train(data_path=DATA_PATH, verbose: bool = True, param_grid=None):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    param_grid = param_grid if param_grid is not None else XGB_PARAM_GRID

    df = load_raw(data_path)
    train_df, val_df, test_df = time_based_split(df)

    if verbose:
        print("=== Class distribution ===")
        for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
            dist = class_distribution(split)
            print(f"\n{name} (n={len(split)}):")
            print(dist)

    X_train, y_train = xy(train_df)
    X_val, y_val = xy(val_df)
    X_test, y_test = xy(test_df)

    if CLASS_WEIGHT_STRATEGY == "balanced":
        sample_weight = compute_sample_weight("balanced", y_train)
    else:
        sample_weight = None

    # --- Bounded hyperparameter search, selected on VAL only ---
    best_model, best_f1, best_params = None, -1.0, None
    search_log = []
    for params in param_grid:
        model, val_f1 = _fit_candidate(params, X_train, y_train, X_val, y_val, sample_weight)
        search_log.append({**params, "val_macro_f1": val_f1})
        if verbose:
            print(f"candidate {params} -> val macro-F1 = {val_f1:.4f}")
        if val_f1 > best_f1:
            best_model, best_f1, best_params = model, val_f1, params

    model = best_model
    if verbose:
        print(f"\nSelected hyperparameters: {best_params} "
              f"(val macro-F1 = {best_f1:.4f})")

    # --- Evaluation on the held-out (chronologically last) test split ---
    probs = model.predict_proba(X_test)
    preds = probs.argmax(axis=1)

    report = classification_report(
        y_test, preds, target_names=CLASSES, zero_division=0, digits=3
    )
    report_dict = classification_report(
        y_test, preds, target_names=CLASSES, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_test, preds, labels=list(range(len(CLASSES))))

    malicious_recall, malicious_to_benign_rate = _malicious_metrics(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    overall_acc = float((preds == y_test).mean())

    if verbose:
        print("\n=== XGBoost test-set classification report (time-based split) ===")
        print(report)
        print("=== Confusion matrix (rows=true, cols=pred) ===")
        print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES))
        print(f"\nOverall test accuracy: {overall_acc:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        if malicious_recall is not None:
            print(f"Malicious recall (any malicious class correctly NOT called benign): "
                  f"{malicious_recall:.4f}")
            print(f"Malicious->benign rate (malicious traffic missed as benign): "
                  f"{malicious_to_benign_rate:.4f}")

    # --- Scaler for the DL models (fit on train only, no leakage) ---
    scaler = StandardScaler().fit(X_train)

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(XGB_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    # --- reports/ artifacts (section 28) ---
    (REPORT_DIR / "classification_report.txt").write_text(report)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(
        REPORT_DIR / "confusion_matrix.csv"
    )
    pd.DataFrame(search_log).to_csv(REPORT_DIR / "xgb_hyperparam_search.csv", index=False)
    with open(REPORT_DIR / "xgb_summary.json", "w") as f:
        json.dump({
            "selected_params": best_params,
            "val_macro_f1": best_f1,
            "test_accuracy": overall_acc,
            "test_macro_f1": macro_f1,
            "malicious_recall": malicious_recall,
            "malicious_to_benign_rate": malicious_to_benign_rate,
            "per_class": report_dict,
        }, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(CLASSES)))
        ax.set_yticks(range(len(CLASSES)))
        ax.set_xticklabels(CLASSES, rotation=45, ha="right")
        ax.set_yticklabels(CLASSES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("XGBoost confusion matrix (test split)")
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150)
        plt.close(fig)
    except ImportError:
        pass  # matplotlib optional; CSV confusion matrix is still written

    if verbose:
        print(f"\nSaved model to {XGB_MODEL_PATH}")
        print(f"Saved scaler to {SCALER_PATH}")
        print(f"Saved reports to {REPORT_DIR}/")

    return {
        "model": model,
        "scaler": scaler,
        "report": report,
        "confusion_matrix": cm,
        "malicious_recall": malicious_recall,
        "malicious_to_benign_rate": malicious_to_benign_rate,
        "test_accuracy": overall_acc,
        "test_macro_f1": macro_f1,
        "best_params": best_params,
        "splits": {"train": train_df, "val": val_df, "test": test_df},
    }


if __name__ == "__main__":
    train()
