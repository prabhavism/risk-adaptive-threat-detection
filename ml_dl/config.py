from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CLASSES = [
    "benign", "ddos", "c2_beaconing", "dga_dns_tunnelling",
    "encrypted_malware", "recon_scanning", "data_exfiltration",
]

FEATURE_COLUMNS = [
    "duration", "packet_count", "byte_count", "packet_rate", "byte_rate",
    "avg_packet_size", "byte_ratio", "packet_ratio", "dest_fanout",
    "port_fanout", "flow_count", "entropy", "dns_query_length",
    "dns_entropy", "iat_mean", "iat_std", "periodicity_score",
]

DATA_PATH = ROOT / "data" / "flow_features.csv"
MODEL_DIR = ROOT / "models"

XGB_MODEL_PATH = MODEL_DIR / "model.pkl"
LIGHT_DL_PATH = MODEL_DIR / "light_weights.h5"
HEAVY_DL_PATH = MODEL_DIR / "heavy_weights.h5"
THRESHOLD_PATH = MODEL_DIR / "threshold.pkl"

# Starting point for the routing threshold theta.
# High ML confidence -> Light DL, low confidence -> Heavy DL.
# Will be tuned once real data is available (see ml_dl/train_xgboost.py).
DEFAULT_THETA = 0.85
