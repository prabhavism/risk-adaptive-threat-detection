"""
Shared data-loading, preprocessing, splitting and sequence-building
helpers used by every training/eval script, so all of them agree on
exactly the same train/val/test rows, the same engineered features and
the same scaling.

Leakage-prevention strategy (section 4 of the upgrade brief), and its
limits, stated explicitly:

  - Split is chronological (time_based_split): train is strictly
    "before" val, val strictly "before" test, by row order. This stops
    the most obvious leak -- near-duplicate flows from the same DDoS
    burst or beaconing session ending up in both train and test.
  - The locked schema (docs/interfaces.md) has no session/scenario id
    and no real capture timestamp, so a stronger session-aware group
    split isn't possible from Person 2's side today. If Person 1 adds
    a scenario/session id column, group-splitting on it (so a whole
    attack scenario lands entirely in one split) should be added on
    top of the chronological split, not instead of it.
  - Heavy DL sequences are built walking forward through time and only
    ever look backward (build_sequences keeps a running per-host
    buffer); a sequence for row i never contains row j > i. Because
    sequences are built independently per split (train sequences only
    from train rows, val only from val rows, etc.), no test-set flow
    ever appears inside a training sequence either.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from ml_dl.config import (
    CLASSES, FEATURE_COLUMNS, DATA_PATH, TRAIN_FRAC, VAL_FRAC, SEQ_LEN,
)

LABEL_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive numeric DNS/TLS signals from the raw categorical columns
    Person 1 already provides (dns_record_type, tls_ja3/ja3s/ja4/sni).
    Nothing here inspects payloads or invents data that wasn't in the
    row -- it just turns "is there DNS/TLS metadata, and what does it
    look like" into numbers a tree/MLP/GRU can use.

    Safe to call on a DataFrame that's missing these raw columns (e.g.
    an older CSV) -- the derived columns default to 0/absent rather
    than raising, so the pipeline stays usable while Person 1's schema
    evolves.
    """
    df = df.copy()

    dns_type = df["dns_record_type"] if "dns_record_type" in df.columns else pd.Series([""] * len(df))
    dns_type = dns_type.fillna("")
    df["has_dns"] = (dns_type != "").astype(float)
    for rtype in ["A", "AAAA", "TXT", "NS"]:
        df[f"dns_is_{rtype}"] = (dns_type == rtype).astype(float)

    ja3 = df["tls_ja3"] if "tls_ja3" in df.columns else pd.Series([""] * len(df))
    sni = df["tls_sni"] if "tls_sni" in df.columns else pd.Series([""] * len(df))
    ja3 = ja3.fillna("")
    sni = sni.fillna("")
    df["has_tls"] = (ja3 != "").astype(float)
    # SNI length is a real, passively-observable, metadata-only signal
    # (long/unusual SNIs correlate with DGA-fronted C2); never touches
    # the encrypted payload.
    df["tls_sni_length"] = sni.str.len().astype(float)

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    return df


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load flow_features.csv, engineer features, basic cleanup (no split)."""
    df = pd.read_csv(path)

    # If Person 1 ever adds a real capture timestamp, honour it. Until
    # then the CSV row order *is* the chronological order it was
    # written in, which is what the time-based split relies on.
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    df = add_engineered_features(df)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0.0)

    bad_labels = set(df["label"].unique()) - set(CLASSES)
    if bad_labels:
        raise ValueError(f"Unexpected label values not in CLASSES: {bad_labels}")

    return df


def time_based_split(df: pd.DataFrame, train_frac: float = TRAIN_FRAC,
                      val_frac: float = VAL_FRAC):
    """
    Chronological split, NOT a random/stratified split:

        first  70% of rows -> train
        next   15% of rows -> validation
        last   15% of rows -> test

    This matters because with a random split, near-duplicate flows from
    the same C2 beaconing session (or the same DDoS burst) can leak
    into both train and test, making the model look better than it
    really is. A time-based split keeps test traffic strictly "in the
    future" relative to train, which is the realistic deployment
    scenario. Assumes `df` is already in time order (see load_raw). See
    the module docstring for what this does and doesn't protect against.
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)
    return train_df, val_df, test_df


def xy(df: pd.DataFrame, feature_columns=None):
    cols = feature_columns if feature_columns is not None else FEATURE_COLUMNS
    X = df[cols].astype(float)
    y = df["label"].map(LABEL_TO_IDX).astype(int).values
    return X, y


def load_split(path: Path = DATA_PATH):
    """Convenience: raw CSV -> (train_df, val_df, test_df), time-ordered."""
    df = load_raw(path)
    return time_based_split(df)


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Class counts + fractions, e.g. for reporting before/after class-
    weighting (section 6). Always includes every class in CLASSES even
    if a split happens to contain zero rows of it, so distributions are
    directly comparable across train/val/test.
    """
    counts = df["label"].value_counts().reindex(CLASSES, fill_value=0)
    fractions = counts / max(len(df), 1)
    return pd.DataFrame({"count": counts, "fraction": fractions})


def build_sequences(df: pd.DataFrame, seq_len: int = SEQ_LEN,
                     group_col: str = "src_ip", feature_columns=None):
    """
    Turn a flow-level DataFrame into per-flow sequences for the Heavy DL
    temporal model: for every row, take the last `seq_len` flows from
    the same src_ip (including itself), in time order. Hosts with less
    history than seq_len are front-padded by repeating their earliest
    available flow, so every row still produces one training/inference
    sample (no rows are dropped, which keeps sequence count == row
    count and makes labels line up 1:1).

    Only ever looks backward: the buffer for host h at row i contains
    rows <= i from that host, so a sequence never contains a flow that
    happens later in the DataFrame's row order. Call this separately
    per split (train/val/test) -- never on the concatenation of all
    three -- so no future/test information leaks into a training
    sequence via another host's shared buffer state.

    Returns:
        X_seq: np.ndarray, shape (n_rows, seq_len, n_features)
        y_seq: np.ndarray, shape (n_rows,)  -- label of the *current*
               (last) flow in each sequence, matching df's row order.
    """
    cols = feature_columns if feature_columns is not None else FEATURE_COLUMNS
    n = len(df)
    n_features = len(cols)
    X_seq = np.zeros((n, seq_len, n_features), dtype=np.float32)

    feats = df[cols].astype(float).values
    labels = df["label"].map(LABEL_TO_IDX).astype(int).values
    groups = df[group_col].values

    # Buffer of recent feature rows per host, built incrementally in
    # row order so this also works unchanged in a streaming/inference
    # setting (see ml_dl/predict_interface.py and scripts/replay_stream.py,
    # which build the same kind of buffer flow-by-flow).
    history: dict[str, list[np.ndarray]] = {}

    for i in range(n):
        host = groups[i]
        buf = history.setdefault(host, [])
        buf.append(feats[i])
        if len(buf) > seq_len:
            buf.pop(0)

        window = buf
        if len(window) < seq_len:
            pad = [window[0]] * (seq_len - len(window))
            window = pad + window

        X_seq[i] = np.stack(window)

    return X_seq, labels
