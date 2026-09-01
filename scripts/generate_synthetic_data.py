"""
Generates a synthetic flow_features.csv matching docs/interfaces.md.

IMPORTANT: Labels are NOT assigned randomly. Each class has realistic
feature distributions so the ML models can actually learn to distinguish them.
This is required for training and testing to be meaningful.

Run:
    python scripts/generate_synthetic_data.py --rows 5000
"""
import argparse
import uuid

import numpy as np
import pandas as pd

CLASSES = [
    "benign", "ddos", "c2_beaconing", "dga_dns_tunnelling",
    "encrypted_malware", "recon_scanning", "data_exfiltration",
]

PROTOCOLS = ["tcp", "udp", "icmp"]


def random_ip(rng):
    return ".".join(str(rng.integers(1, 255)) for _ in range(4))


def _row(label: str, rng: np.random.Generator) -> dict:
    """Generate one flow row with feature values correlated to the label."""

    is_dns = False
    is_tls = False

    if label == "benign":
        duration        = float(rng.exponential(5.0))
        packet_count    = int(rng.integers(5, 500))
        byte_count      = int(rng.integers(500, 500_000))
        packet_rate     = float(rng.uniform(1, 50))
        byte_rate       = float(rng.uniform(500, 50_000))
        avg_packet_size = float(rng.uniform(300, 900))
        byte_ratio      = float(rng.uniform(0.4, 0.6))
        packet_ratio    = float(rng.uniform(0.4, 0.6))
        dest_fanout     = int(rng.integers(1, 5))
        port_fanout     = int(rng.integers(1, 4))
        flow_count      = int(rng.integers(1, 20))
        entropy         = float(rng.uniform(3.0, 6.0))
        iat_mean        = float(rng.uniform(0.05, 1.0))
        iat_std         = float(rng.uniform(0.05, 0.5))
        periodicity_score = float(rng.uniform(0.0, 0.3))
        is_tls          = rng.random() < 0.4
        protocol        = rng.choice(["tcp", "udp"])

    elif label == "ddos":
        # SYN/UDP flood: very high packet rate, small packets, many flows
        duration        = float(rng.exponential(0.5))
        packet_count    = int(rng.integers(500, 50_000))
        byte_count      = int(rng.integers(30_000, 5_000_000))
        packet_rate     = float(rng.uniform(5_000, 100_000))
        byte_rate       = float(rng.uniform(500_000, 10_000_000))
        avg_packet_size = float(rng.uniform(40, 100))   # tiny SYN packets
        byte_ratio      = float(rng.uniform(0.85, 1.0))  # mostly outbound
        packet_ratio    = float(rng.uniform(0.85, 1.0))
        dest_fanout     = int(rng.integers(1, 3))         # same target
        port_fanout     = int(rng.integers(1, 3))
        flow_count      = int(rng.integers(500, 5_000))
        entropy         = float(rng.uniform(1.0, 3.5))    # low-entropy payload
        iat_mean        = float(rng.uniform(0.0001, 0.002))
        iat_std         = float(rng.uniform(0.0001, 0.001))
        periodicity_score = float(rng.uniform(0.0, 0.2))
        protocol        = rng.choice(["tcp", "udp"])

    elif label == "c2_beaconing":
        # Very periodic, low traffic, stable timing
        duration        = float(rng.uniform(60, 3600))
        packet_count    = int(rng.integers(2, 20))
        byte_count      = int(rng.integers(200, 5_000))
        packet_rate     = float(rng.uniform(0.001, 0.05))
        byte_rate       = float(rng.uniform(0.1, 10))
        avg_packet_size = float(rng.uniform(100, 400))
        byte_ratio      = float(rng.uniform(0.45, 0.55))
        packet_ratio    = float(rng.uniform(0.45, 0.55))
        dest_fanout     = int(rng.integers(1, 2))
        port_fanout     = int(rng.integers(1, 2))
        flow_count      = int(rng.integers(10, 200))
        entropy         = float(rng.uniform(4.5, 7.5))
        # Hallmark: very regular inter-arrival times (low CV = high periodicity)
        iat_mean        = float(rng.uniform(30, 300))    # ~30s–5min intervals
        iat_std         = float(rng.uniform(0.1, 2.0))   # tiny std → periodic
        periodicity_score = float(rng.uniform(0.75, 1.0))  # high!
        is_tls          = rng.random() < 0.7
        protocol        = "tcp"

    elif label == "dga_dns_tunnelling":
        # High DNS query entropy, long query names, unusual record types
        duration        = float(rng.exponential(1.0))
        packet_count    = int(rng.integers(2, 30))
        byte_count      = int(rng.integers(200, 10_000))
        packet_rate     = float(rng.uniform(0.5, 10))
        byte_rate       = float(rng.uniform(100, 5_000))
        avg_packet_size = float(rng.uniform(100, 300))
        byte_ratio      = float(rng.uniform(0.3, 0.7))
        packet_ratio    = float(rng.uniform(0.3, 0.7))
        dest_fanout     = int(rng.integers(5, 50))    # many DNS resolvers
        port_fanout     = int(rng.integers(1, 3))
        flow_count      = int(rng.integers(20, 300))
        entropy         = float(rng.uniform(5.5, 8.0))   # high entropy
        iat_mean        = float(rng.uniform(0.01, 0.5))
        iat_std         = float(rng.uniform(0.01, 0.2))
        periodicity_score = float(rng.uniform(0.0, 0.4))
        is_dns          = True
        protocol        = "udp"

    elif label == "encrypted_malware":
        # Unusual TLS fingerprint, abnormal cipher, medium traffic
        duration        = float(rng.exponential(10.0))
        packet_count    = int(rng.integers(20, 300))
        byte_count      = int(rng.integers(5_000, 500_000))
        packet_rate     = float(rng.uniform(2, 30))
        byte_rate       = float(rng.uniform(500, 50_000))
        avg_packet_size = float(rng.uniform(300, 1400))
        byte_ratio      = float(rng.uniform(0.5, 0.8))
        packet_ratio    = float(rng.uniform(0.5, 0.8))
        dest_fanout     = int(rng.integers(1, 5))
        port_fanout     = int(rng.integers(1, 3))
        flow_count      = int(rng.integers(5, 50))
        entropy         = float(rng.uniform(7.0, 8.0))   # very high entropy
        iat_mean        = float(rng.uniform(0.01, 0.2))
        iat_std         = float(rng.uniform(0.005, 0.1))
        periodicity_score = float(rng.uniform(0.1, 0.5))
        is_tls          = True
        protocol        = "tcp"

    elif label == "recon_scanning":
        # High dest/port fanout, short flows, low bytes
        duration        = float(rng.exponential(0.1))
        packet_count    = int(rng.integers(1, 5))
        byte_count      = int(rng.integers(40, 200))
        packet_rate     = float(rng.uniform(50, 500))
        byte_rate       = float(rng.uniform(1_000, 20_000))
        avg_packet_size = float(rng.uniform(40, 80))
        byte_ratio      = float(rng.uniform(0.8, 1.0))
        packet_ratio    = float(rng.uniform(0.8, 1.0))
        dest_fanout     = int(rng.integers(50, 500))   # scanning many hosts
        port_fanout     = int(rng.integers(20, 200))   # scanning many ports
        flow_count      = int(rng.integers(100, 2_000))
        entropy         = float(rng.uniform(0.5, 2.5))
        iat_mean        = float(rng.uniform(0.0001, 0.01))
        iat_std         = float(rng.uniform(0.0001, 0.005))
        periodicity_score = float(rng.uniform(0.0, 0.2))
        protocol        = rng.choice(["tcp", "icmp"])

    elif label == "data_exfiltration":
        # Large outbound transfer, high byte_ratio, high entropy
        duration        = float(rng.uniform(10, 600))
        packet_count    = int(rng.integers(100, 5_000))
        byte_count      = int(rng.integers(1_000_000, 100_000_000))
        packet_rate     = float(rng.uniform(10, 200))
        byte_rate       = float(rng.uniform(100_000, 5_000_000))
        avg_packet_size = float(rng.uniform(800, 1500))
        byte_ratio      = float(rng.uniform(0.85, 1.0))   # massive outbound asymmetry
        packet_ratio    = float(rng.uniform(0.7, 0.95))
        dest_fanout     = int(rng.integers(1, 3))
        port_fanout     = int(rng.integers(1, 2))
        flow_count      = int(rng.integers(5, 50))
        entropy         = float(rng.uniform(6.5, 8.0))
        iat_mean        = float(rng.uniform(0.001, 0.05))
        iat_std         = float(rng.uniform(0.001, 0.02))
        periodicity_score = float(rng.uniform(0.0, 0.3))
        is_tls          = rng.random() < 0.5
        protocol        = "tcp"

    # ── DNS fields ─────────────────────────────────────────────────────────
    if is_dns:
        dns_query_length = float(rng.integers(30, 120))     # long for DGA
        dns_entropy      = float(rng.uniform(3.5, 5.0))
        dns_record_type  = str(rng.choice(["A", "AAAA", "TXT", "NS", "CNAME"]))
    else:
        dns_query_length = 0.0
        dns_entropy      = 0.0
        dns_record_type  = ""

    # ── TLS fields ──────────────────────────────────────────────────────────
    if is_tls:
        import secrets
        tls_ja3  = secrets.token_hex(8)
        tls_ja3s = secrets.token_hex(8)
        tls_ja4  = secrets.token_hex(8)
        tls_sni  = f"host{rng.integers(1, 999)}.example.com"
    else:
        tls_ja3 = tls_ja3s = tls_ja4 = tls_sni = ""

    return {
        "flow_id":          str(uuid.uuid4()),
        "src_ip":           random_ip(rng),
        "dst_ip":           random_ip(rng),
        "src_port":         int(rng.integers(1024, 65535)),
        "dst_port":         int(rng.choice([80, 443, 53, 22, 3389, 8080])),
        "protocol":         str(protocol),
        "duration":         round(duration, 6),
        "packet_count":     packet_count,
        "byte_count":       byte_count,
        "packet_rate":      round(packet_rate, 4),
        "byte_rate":        round(byte_rate, 4),
        "avg_packet_size":  round(avg_packet_size, 4),
        "byte_ratio":       round(byte_ratio, 4),
        "packet_ratio":     round(packet_ratio, 4),
        "dest_fanout":      dest_fanout,
        "port_fanout":      port_fanout,
        "flow_count":       flow_count,
        "entropy":          round(entropy, 4),
        "dns_query_length": dns_query_length,
        "dns_entropy":      dns_entropy,
        "dns_record_type":  dns_record_type,
        "tls_ja3":          tls_ja3,
        "tls_ja3s":         tls_ja3s,
        "tls_ja4":          tls_ja4,
        "tls_sni":          tls_sni,
        "iat_mean":         round(iat_mean, 6),
        "iat_std":          round(iat_std, 6),
        "periodicity_score": round(periodicity_score, 4),
        "label":            label,
    }


def generate(n_rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Balanced class distribution
    labels = [CLASSES[i % len(CLASSES)] for i in range(n_rows)]
    rng.shuffle(labels)
    rows = [_row(label, rng) for label in labels]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--out",  type=str, default="data/flow_features.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate(args.rows, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} synthetic rows to {args.out}")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}")
