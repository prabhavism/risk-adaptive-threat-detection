from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── Class definitions ────────────────────────────────────────────────────────
CLASSES = [
    "benign", "ddos", "c2_beaconing", "dga_dns_tunnelling",
    "encrypted_malware", "recon_scanning", "data_exfiltration",
]
MALICIOUS_CLASSES = [c for c in CLASSES if c != "benign"]

# Reproducibility: one seed, used everywhere (numpy, xgboost, keras).
SEED = 42

# --- Feature groups -----------------------------------------------------
# Grouped so ml_dl/ablation.py can add them cumulatively and so the
# feature set stays traceable to what's *actually* observable passively
# (docs/interfaces.md / section 5 of the upgrade brief). Nothing here
# is fabricated: dns_* / tls_* groups are derived from the raw
# categorical columns Person 1 already provides (dns_record_type,
# tls_ja3/ja3s/ja4/sni) by ml_dl.data_utils.add_engineered_features --
# they don't require any new column from Person 1.
FEATURE_GROUPS = {
    # Available now: basic per-flow counters.
    "flow": [
        "duration", "packet_count", "byte_count", "packet_rate",
        "byte_rate", "avg_packet_size",
    ],
    # Available now: fan-out / connection behaviour (scanning, C2).
    "connection": ["dest_fanout", "port_fanout", "flow_count"],
    # Available now: inbound/outbound asymmetry (exfiltration signal).
    "directional": ["byte_ratio", "packet_ratio"],
    # Available now: inter-arrival timing (beaconing signal).
    "timing": ["iat_mean", "iat_std", "periodicity_score"],
    # Available now: payload/byte entropy.
    "entropy": ["entropy"],
    # Available when DNS traffic is present; derived from
    # dns_query_length/dns_entropy/dns_record_type. Rows with no DNS in
    # the flow get 0 / not-present, not a fabricated value.
    "dns": [
        "dns_query_length", "dns_entropy", "has_dns",
        "dns_is_A", "dns_is_AAAA", "dns_is_TXT", "dns_is_NS",
    ],
    # Available when TLS/QUIC metadata is present. Metadata-only by
    # construction: presence + SNI length, never payload content, and
    # the JA3/JA3S/JA4 hash strings themselves are kept out of the
    # numeric feature vector (a raw hash isn't a meaningful magnitude
    # for XGBoost/DL) -- only derived, order-preserving signals go in.
    "tls": ["has_tls", "tls_sni_length"],
}

# Cumulative order used by ml_dl/ablation.py (section 13): each stage
# adds one more group on top of the previous ones.
ABLATION_STAGES = [
    ("flow_only", ["flow"]),
    ("flow_plus_behavioral", ["flow", "connection", "directional", "timing", "entropy"]),
    ("plus_dns", ["flow", "connection", "directional", "timing", "entropy", "dns"]),
    ("plus_tls", ["flow", "connection", "directional", "timing", "entropy", "dns", "tls"]),
]

FEATURE_COLUMNS = [c for group in FEATURE_GROUPS.values() for c in group]

# The 17 purely-numeric columns Person 1 hands over directly (no
# engineering needed) -- kept around for anything that wants the
# original, pre-upgrade feature set.
ORIGINAL_FEATURE_COLUMNS = [
    "duration", "packet_count", "byte_count", "packet_rate", "byte_rate",
    "avg_packet_size", "byte_ratio", "packet_ratio", "dest_fanout",
    "port_fanout", "flow_count", "entropy", "dns_query_length",
    "dns_entropy", "iat_mean", "iat_std", "periodicity_score",
]

DATA_PATH = ROOT / "data" / "flow_features.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

XGB_MODEL_PATH = MODEL_DIR / "model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
LIGHT_DL_PATH = MODEL_DIR / "light_weights.keras"
HEAVY_DL_PATH = MODEL_DIR / "heavy_weights.keras"
THRESHOLD_PATH = MODEL_DIR / "threshold.pkl"
ROUTING_STATS_PATH = MODEL_DIR / "routing_stats.pkl"
CALIBRATOR_PATH = MODEL_DIR / "calibrator.pkl"

# Chronological split ratios (Person 1's flow_features.csv is written in
# capture order; if a `timestamp` column is ever added, data_utils will
# sort on it first, otherwise row order is treated as time order).
# There is currently no session/scenario id column in the locked schema
# (docs/interfaces.md) to group-split on, so the leakage-prevention
# strategy for now is chronological-only -- see data_utils.time_based_split
# docstring. If Person 1 adds a scenario/session id, group-aware
# splitting should be layered on top of this.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# TEST_FRAC is implicitly the remaining 0.15

# Heavy DL sequence length: number of recent flows from the same src_ip
# fed to the LSTM/GRU as one temporal sample (beaconing/exfil detection
# needs history, not a single flow). Only past + current flows are ever
# used -- see data_utils.build_sequences -- never future ones.
SEQ_LEN = 10

# Starting point for the routing threshold theta.
# High ML confidence -> Light DL, low confidence -> Heavy DL.
# Retuned by ml_dl/tune_threshold.py once Light/Heavy DL are trained.
DEFAULT_THETA = 0.85

# Class imbalance handling for XGBoost (section 6): "balanced" computes
# inverse-frequency sample weights from the TRAIN split only; None
# disables weighting.
CLASS_WEIGHT_STRATEGY = "balanced"

# Small, deliberately bounded hyperparameter grid for train_xgboost.py's
# validation-only search (section 7) -- not a full sweep, just enough to
# show tuning is principled and cheap to re-run.
XGB_PARAM_GRID = [
    {"max_depth": 4, "learning_rate": 0.10, "n_estimators": 300, "min_child_weight": 1, "gamma": 0.0},
    {"max_depth": 6, "learning_rate": 0.08, "n_estimators": 400, "min_child_weight": 1, "gamma": 0.0},
    {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 600, "min_child_weight": 3, "gamma": 0.1},
    {"max_depth": 8, "learning_rate": 0.05, "n_estimators": 500, "min_child_weight": 3, "gamma": 0.1},
]

# Severity bucketing for the standardized alert schema (section 14).
SEVERITY_THRESHOLDS = [
    (0.90, "CRITICAL"),
    (0.75, "HIGH"),
    (0.50, "MEDIUM"),
]
SEVERITY_DEFAULT = "LOW"
