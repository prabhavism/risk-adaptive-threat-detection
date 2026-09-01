"""
Trains the 7-class XGBoost triage classifier and sweeps the routing
threshold theta.

Works today against scripts/generate_synthetic_data.py output. When
Person 1's real data/flow_features.csv lands, just re-run this script
unchanged — same schema, same columns.
"""
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from ml_dl.config import (
    CLASSES, FEATURE_COLUMNS, DATA_PATH, XGB_MODEL_PATH, THRESHOLD_PATH,
    MODEL_DIR,
)


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    X = df[FEATURE_COLUMNS]
    y = df["label"].map({c: i for i, c in enumerate(CLASSES)})
    return X, y


def train():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(CLASSES),
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)
    preds = probs.argmax(axis=1)
    confidences = probs.max(axis=1)

    print(classification_report(
        y_test, preds, target_names=CLASSES, zero_division=0
    ))

    theta = sweep_threshold(confidences, preds, y_test.values)
    print(f"Suggested routing threshold theta = {theta:.3f}")

    with open(XGB_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(THRESHOLD_PATH, "wb") as f:
        pickle.dump(theta, f)

    print(f"Saved model to {XGB_MODEL_PATH}")
    print(f"Saved threshold to {THRESHOLD_PATH}")


def sweep_threshold(confidences, preds, truth, candidates=None):
    """
    Pick theta so that flows routed to Light DL (confidence >= theta) have
    high ML accuracy, keeping the Heavy DL path for genuinely ambiguous
    flows. This is a placeholder heuristic — revisit once real data and
    Light/Heavy DL accuracy numbers exist (Person 2, week 2).
    """
    if candidates is None:
        candidates = np.arange(0.5, 0.99, 0.01)
    correct = preds == truth
    best_theta, best_score = 0.85, -1
    for t in candidates:
        mask = confidences >= t
        if mask.sum() == 0:
            continue
        acc_above = correct[mask].mean()
        coverage = mask.mean()
        score = acc_above * coverage  # crude combined objective
        if score > best_score:
            best_score, best_theta = score, t
    return best_theta


if __name__ == "__main__":
    train()
