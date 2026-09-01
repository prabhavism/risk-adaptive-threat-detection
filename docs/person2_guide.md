# Person 2 — ML/DL Detection & Intelligence Engine Guide

Your branch: **`person2-ml`**  
Your input: `data/flow_features.csv` (Person 1's output)  
Your deliverable: `ml_dl/predict_interface.py` — a single `predict(flow)` function that Person 3 calls

> **Rule #1** — The return dict from `predict()` is locked in `docs/interfaces.md`.
> If you rename any field, tell Person 3 immediately — their entire alert layer breaks.

---

## 1. First-Time Setup

```bash
# Clone the repo (if you haven't already)
git clone https://github.com/PRABHAVISM/risk-adaptive-threat-detection.git
cd risk-adaptive-threat-detection

# Switch to your branch
git checkout person2-ml

# Create a Python virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Your Directory — What Exists and What Still Needs Work

```
risk-adaptive-threat-detection/
└── ml_dl/
    ├── __init__.py              ✅ done
    ├── config.py                ✅ done — FEATURE_COLUMNS, CLASSES, paths
    ├── train_xgboost.py         ⚠ skeleton — needs full training loop
    ├── routing.py               ⚠ skeleton — needs threshold tuning
    ├── light_dl.py              ⚠ skeleton — needs MLP training loop
    ├── heavy_dl.py              ⚠ skeleton — needs LSTM/GRU training loop
    ├── explainability.py        ⚠ skeleton — needs SHAP integration
    └── predict_interface.py     ⚠ skeleton — needs to wire all models together
```

Before Person 1's real data arrives, use the synthetic generator:
```bash
python scripts/generate_synthetic_data.py --rows 10000
# → writes data/flow_features.csv
```

---

## 3. Working Without Real Data

Synthetic data lets you build and test the full ML pipeline now.
When Person 1's real `data/flow_features.csv` arrives, just drop it in — same filename,
same schema — and re-run the same training commands. No code changes needed.

---

## 4. ml_dl/train_xgboost.py — What to Complete

The skeleton already exists. You need to add:

```python
# Full training loop outline:
import pandas as pd, numpy as np, pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from ml_dl.config import DATA_PATH, FEATURE_COLUMNS, CLASSES, XGB_MODEL_PATH, DEFAULT_THETA

df = pd.read_csv(DATA_PATH)
X  = df[FEATURE_COLUMNS].fillna(0)
le = LabelEncoder().fit(CLASSES)
y  = le.transform(df["label"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="mlogloss",
    early_stopping_rounds=20,
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))
print(confusion_matrix(y_test, y_pred))

# Save
with open(XGB_MODEL_PATH, "wb") as f:
    pickle.dump({"model": model, "label_encoder": le}, f)

print(f"Model saved to {XGB_MODEL_PATH}")
```

**Key metrics to report**: precision, recall, F1 per class, overall accuracy, false-positive rate for benign traffic.

---

## 5. ml_dl/routing.py — Risk-Adaptive Routing

```python
# Three-tier routing:
# XGBoost confidence ≥ THETA_HIGH  → XGBoost verdict is final (skip DL)
# THETA_LOW ≤ confidence < THETA_HIGH → Light DL verifies
# confidence < THETA_LOW              → Heavy DL for hard cases

THETA_HIGH = 0.90   # tune on validation set
THETA_LOW  = 0.60

def route(xgb_confidence: float) -> str:
    if xgb_confidence >= THETA_HIGH:
        return "xgb_only"
    elif xgb_confidence >= THETA_LOW:
        return "light"
    else:
        return "heavy"
```

Tune `THETA_HIGH` and `THETA_LOW` on validation data to balance accuracy vs. throughput.

---

## 6. ml_dl/light_dl.py — Lightweight MLP

```python
# Architecture: fast screening for moderately suspicious traffic
# Input: FEATURE_COLUMNS (17 numerical features)
# Output: softmax over 7 classes

import torch, torch.nn as nn

class LightMLP(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),         nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)
```

Training tip: use `nn.CrossEntropyLoss()` and `torch.optim.Adam(lr=1e-3)` with early stopping on validation F1.

---

## 7. ml_dl/heavy_dl.py — Temporal Sequence Model

```python
# Architecture: LSTM for capturing temporal patterns (beaconing, exfil)
# Input: sequence of feature vectors (sliding window of recent flows per host)
# Output: softmax over 7 classes

class HeavyLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int, n_classes: int, n_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, n_layers,
                            batch_first=True, dropout=0.3)
        self.fc   = nn.Linear(hidden, n_classes)

    def forward(self, x):          # x: (batch, seq_len, n_features)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])      # last hidden state
```

For the sequence, group the last `SEQ_LEN=10` flows from the same `src_ip` as one sample.

---

## 8. ml_dl/explainability.py — SHAP Wrapper

```python
import shap, numpy as np

class SHAPExplainer:
    def __init__(self, model, X_background: np.ndarray):
        self.explainer = shap.TreeExplainer(model)   # fast for XGBoost

    def top_features(self, x_row: np.ndarray, feature_names: list, top_k: int = 5):
        shap_vals = self.explainer.shap_values(x_row)  # shape: (n_classes, n_features)
        # pick the predicted class shap values
        # sort by abs value, return top_k
        idx = np.argsort(np.abs(shap_vals))[::-1][:top_k]
        return [
            {"feature": feature_names[i], "value": round(float(shap_vals[i]), 4)}
            for i in idx
        ]
```

---

## 9. ml_dl/predict_interface.py — The Final Wired Interface

This is what Person 3 calls. Make sure it matches `docs/interfaces.md` exactly.

```python
def predict(flow: dict) -> dict:
    """
    Input:  one row from flow_features.csv as a dict
    Output: {
        "ml_verdict":    str,    # one of the 7 class strings
        "ml_confidence": float,  # 0-1
        "dl_verdict":    str,
        "dl_confidence": float,
        "model_used":    str,    # "light" or "heavy"
        "shap_evidence": [{"feature": str, "value": float}, ...]
    }
    """
```

**Wire-up order**:
1. Extract `FEATURE_COLUMNS` from `flow` dict → numpy array
2. Run XGBoost → get probabilities → pick top class + confidence
3. Call `route(confidence)` → decide `"xgb_only"`, `"light"`, or `"heavy"`
4. If light/heavy → run that DL model → get dl_verdict + dl_confidence
5. Call SHAP explainer → get top 5 feature contributions
6. Return the full dict

---

## 10. Running and Testing

```bash
# Generate data
python scripts/generate_synthetic_data.py --rows 10000

# Train XGBoost
python -m ml_dl.train_xgboost

# Run unit tests
pytest tests/ -v

# Quick smoke test of predict interface
python -c "
import pandas as pd
from ml_dl.predict_interface import predict
row = pd.read_csv('data/flow_features.csv').iloc[0].to_dict()
print(predict(row))
"
```

---

## 11. Day-by-Day Plan

| Day | Focus |
|---|---|
| 1 | Repo set up, schema agreed, synthetic data generator working |
| 2–3 | Complete `train_xgboost.py` training loop; precision/recall/F1/confusion matrix |
| 4–5 | `light_dl.py` MLP training loop; validate on synthetic data |
| 6–7 | `heavy_dl.py` LSTM training loop; sequence dataset builder |
| 8 | `explainability.py` SHAP wrapper + top-feature extraction |
| 9 | `predict_interface.py` wired end-to-end; all unit tests passing |
| 10 | Docs for Person 3; dry-run integration with dummy outputs |
| — | *(real data arrives)* swap in real CSV, retrain all models, export final `models/*` |

---

## 12. Git Workflow

```bash
# Start of every session
git checkout person2-ml
git pull origin person2-ml

# After doing work
git add ml_dl/
git commit -m "feat(ml): complete XGBoost training loop with F1 per class"
git push origin person2-ml

# When predict_interface.py is stable → open a Pull Request:
#   person2-ml → main
# Tag Person 3 to review (they consume your predict interface directly)
```

> **Share model files** (`models/*.pkl`, `models/*.pt`) via the team Google Drive — they're gitignored.
