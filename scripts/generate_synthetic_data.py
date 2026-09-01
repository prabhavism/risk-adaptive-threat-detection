"""
Generates a synthetic flow_features.csv matching docs/interfaces.md so
Person 2 (and Person 3) can build and test the full pipeline before
Person 1's real captured/labelled data is ready.

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


def generate(n_rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_rows):
        label = rng.choice(CLASSES)
        is_dns = rng.random() < 0.15
        is_tls = rng.random() < 0.25
        rows.append({
            "flow_id": str(uuid.uuid4()),
            "src_ip": random_ip(rng),
            "dst_ip": random_ip(rng),
            "src_port": int(rng.integers(1024, 65535)),
            "dst_port": int(rng.choice([80, 443, 53, 22, 3389, 8080])),
            "protocol": rng.choice(PROTOCOLS),
            "duration": float(rng.exponential(2.0)),
            "packet_count": int(rng.integers(1, 5000)),
            "byte_count": int(rng.integers(64, 5_000_000)),
            "packet_rate": float(rng.exponential(50)),
            "byte_rate": float(rng.exponential(10000)),
            "avg_packet_size": float(rng.normal(500, 200)),
            "byte_ratio": float(rng.random()),
            "packet_ratio": float(rng.random()),
            "dest_fanout": int(rng.integers(1, 200)),
            "port_fanout": int(rng.integers(1, 100)),
            "flow_count": int(rng.integers(1, 500)),
            "entropy": float(rng.random() * 8),
            "dns_query_length": float(rng.integers(0, 60)) if is_dns else 0.0,
            "dns_entropy": float(rng.random() * 4) if is_dns else 0.0,
            "dns_record_type": rng.choice(["A", "AAAA", "TXT", "NS"]) if is_dns else "",
            "tls_ja3": uuid.uuid4().hex[:16] if is_tls else "",
            "tls_ja3s": uuid.uuid4().hex[:16] if is_tls else "",
            "tls_ja4": uuid.uuid4().hex[:16] if is_tls else "",
            "tls_sni": f"host{rng.integers(1,999)}.example.com" if is_tls else "",
            "iat_mean": float(rng.exponential(0.5)),
            "iat_std": float(rng.exponential(0.2)),
            "periodicity_score": float(rng.random()),
            "label": label,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--out", type=str, default="data/flow_features.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate(args.rows, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} synthetic rows to {args.out}")
