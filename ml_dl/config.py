from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── Class definitions ────────────────────────────────────────────────────────
CLASSES = [
    "benign", "ddos", "c2_beaconing", "dga_dns_tunnelling",
    "encrypted_malware", "recon_scanning", "data_exfiltration",
]

# ── Feature columns (must match docs/interfaces.md exactly) ─────────────────
FEATURE_COLUMNS = [
    "duration", "packet_count", "byte_count", "packet_rate", "byte_rate",
    "avg_packet_size", "byte_ratio", "packet_ratio", "dest_fanout",
    "port_fanout", "flow_count", "entropy", "dns_query_length",
    "dns_entropy", "iat_mean", "iat_std", "periodicity_score",
]

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH      = ROOT / "data"   / "flow_features.csv"
MODEL_DIR      = ROOT / "models"

XGB_MODEL_PATH  = MODEL_DIR / "model.pkl"
LIGHT_DL_PATH   = MODEL_DIR / "light_weights.keras"   # .keras, not .h5
HEAVY_DL_PATH   = MODEL_DIR / "heavy_weights.keras"   # .keras, not .h5
THRESHOLD_PATH  = MODEL_DIR / "threshold.pkl"
SCALER_PATH     = MODEL_DIR / "scaler.pkl"             # StandardScaler — must be saved

# ── Train / val / test split ratios ─────────────────────────────────────────
TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20
TEST_RATIO  = 0.20  # held out; never seen during training or threshold tuning

# ── DL training hyper-parameters ─────────────────────────────────────────────
LIGHT_EPOCHS = 50
HEAVY_EPOCHS = 80
BATCH_SIZE   = 256
PATIENCE     = 10   # early-stopping patience (epochs)

# ── Risk-adaptive routing thresholds ─────────────────────────────────────────
# confidence >= THETA_HIGH  → XGBoost verdict is final (skip DL entirely)
# THETA_LOW <= conf < THETA_HIGH → Light DL verifies
# conf < THETA_LOW            → Heavy DL for hard / ambiguous flows
# Both values are re-tuned by train_xgboost.sweep_threshold() and
# saved to THRESHOLD_PATH; these are fallback defaults only.
THETA_HIGH   = 0.90
THETA_LOW    = 0.60
DEFAULT_THETA = THETA_HIGH  # kept for backward compat with routing.py
