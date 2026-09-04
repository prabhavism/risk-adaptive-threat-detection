"""
Trains both the Light MLP and Heavy MLP verification networks.

Run AFTER ml_dl/train_xgboost.py (which produces the scaler and split indices).
This script independently loads the CSV and applies the same 60/20/20 split
with the same seed so the train/val/test populations are identical.

Output artefacts:
    models/light_weights.keras  — Light MLP (full model, ready to load)
    models/heavy_weights.keras  — Heavy MLP (full model, ready to load)
    models/scaler.pkl           — StandardScaler fitted on training split only

Usage:
    python scripts/train_dl.py
    python scripts/train_dl.py --rows 20000   # generate fresh synthetic data first
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Make sure root is on path when running as a script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml_dl.config import (
    CLASSES, FEATURE_COLUMNS, DATA_PATH, MODEL_DIR,
    LIGHT_DL_PATH, HEAVY_DL_PATH, SCALER_PATH,
    TRAIN_RATIO, VAL_RATIO,
)
from ml_dl import light_dl, heavy_dl


TEST_RATIO = 1.0 - TRAIN_RATIO - VAL_RATIO


# ── Data preparation ──────────────────────────────────────────────────────────

def load_and_split():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS].fillna(0.0).astype(float).values
    label_map = {c: i for i, c in enumerate(CLASSES)}
    y = df["label"].map(label_map).astype(int).values

    # ── Same split as XGBoost (same seed) ────────────────────────────────────
    val_relative = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_relative, stratify=y_trainval, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale(X_train, X_val, X_test):
    """
    Fit scaler on training data only — never on val or test.
    Save scaler to disk so predict_interface.py can use it at inference time.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved -> {SCALER_PATH}")

    return X_train_s, X_val_s, X_test_s


# ── Evaluation helper ─────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, name: str):
    print(f"\n{'='*60}")
    print(f"{name} — TEST SET EVALUATION")
    print("=" * 60)
    probs = model.predict(X_test, verbose=0)
    preds = probs.argmax(axis=1)
    print(classification_report(y_test, preds, target_names=CLASSES, zero_division=0))

    # Benign FPR
    benign_idx = CLASSES.index("benign")
    benign_mask = (y_test == benign_idx)
    fp = ((preds != benign_idx) & benign_mask).sum()
    tn = ((preds == benign_idx) & benign_mask).sum()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    print(f"Benign FPR: {fpr:.4f}  ({fp}/{fp+tn} benign flows misclassified)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Optionally regenerate synthetic data
    if args.rows:
        from scripts.generate_synthetic_data import generate
        print(f"Generating {args.rows} synthetic rows …")
        df = generate(args.rows, seed=42)
        df.to_csv(DATA_PATH, index=False)
        print(f"Written to {DATA_PATH}")

    print("\nLoading and splitting data …")
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split()
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    print("\nScaling features …")
    X_train_s, X_val_s, X_test_s = scale(X_train, X_val, X_test)

    # ── Light DL ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Training Light MLP …")
    print("=" * 60)
    light_model = light_dl.train(X_train_s, y_train, X_val_s, y_val)
    evaluate(light_model, X_test_s, y_test, "Light MLP")
    print(f"Light MLP saved -> {LIGHT_DL_PATH}")

    # ── Heavy DL ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Training Heavy MLP ...")
    print("=" * 60)
    heavy_model = heavy_dl.train(X_train_s, y_train, X_val_s, y_val)
    evaluate(heavy_model, X_test_s, y_test, "Heavy MLP")
    print(f"Heavy MLP saved -> {HEAVY_DL_PATH}")

    print(f"\n{'='*60}")
    print("All DL models trained and saved.")
    print(f"  {LIGHT_DL_PATH}")
    print(f"  {HEAVY_DL_PATH}")
    print(f"  {SCALER_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Light and Heavy DL models")
    parser.add_argument(
        "--rows", type=int, default=None,
        help="If set, regenerate N rows of synthetic data before training"
    )
    main(parser.parse_args())
