"""
Feature-group ablation study (section 13): trains a (smaller, faster)
XGBoost on each cumulative feature-group stage in config.ABLATION_STAGES
and reports validation accuracy/macro-F1, so it's visible which
feature groups actually contribute to detection quality rather than
just asserting they do.

Run (needs data/flow_features.csv, no other trained artifacts required):
    python -m ml_dl.ablation
"""
import json

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

from ml_dl.config import (
    CLASSES, DATA_PATH, ABLATION_STAGES, FEATURE_GROUPS, REPORT_DIR, SEED,
)
from ml_dl.data_utils import load_raw, time_based_split, xy


def run_ablation(data_path=DATA_PATH, verbose: bool = True):
    df = load_raw(data_path)
    train_df, val_df, _ = time_based_split(df)

    rows = []
    for stage_name, group_names in ABLATION_STAGES:
        cols = [c for g in group_names for c in FEATURE_GROUPS[g]]

        X_train, y_train = xy(train_df, feature_columns=cols)
        X_val, y_val = xy(val_df, feature_columns=cols)
        sample_weight = compute_sample_weight("balanced", y_train)

        model = xgb.XGBClassifier(
            objective="multi:softprob", num_class=len(CLASSES),
            n_estimators=200, max_depth=5, learning_rate=0.1,
            eval_metric="mlogloss", random_state=SEED,
        )
        model.fit(X_train, y_train, sample_weight=sample_weight, verbose=False)
        preds = model.predict(X_val)

        row = {
            "stage": stage_name,
            "n_features": len(cols),
            "feature_groups": ",".join(group_names),
            "val_accuracy": float(accuracy_score(y_val, preds)),
            "val_macro_f1": float(f1_score(y_val, preds, average="macro", zero_division=0)),
        }
        rows.append(row)
        if verbose:
            print(f"{stage_name:<22} ({len(cols):>2} features): "
                  f"val accuracy={row['val_accuracy']:.4f}, "
                  f"macro-F1={row['val_macro_f1']:.4f}")

    results = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORT_DIR / "ablation_results.csv", index=False)
    with open(REPORT_DIR / "ablation_results.json", "w") as f:
        json.dump(rows, f, indent=2)

    return results


if __name__ == "__main__":
    run_ablation()
