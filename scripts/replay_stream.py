"""
Streaming/replay demonstration (section 16): proves the detector is an
incremental, per-flow streaming pipeline, not a batch classifier that
waits for the whole file.

Reads flow_features.csv (or any CSV in the same schema) row by row,
optionally paced to a target flows/sec rate, calls
predict_interface.predict() on each row AS IT ARRIVES (host history
accumulates incrementally -- see predict_interface._history), builds a
standardized alert (ml_dl/alerts.py) for anything non-benign, and
writes alerts to a JSONL file as they're generated -- not after the
whole run finishes.

READ-ONLY / PASSIVE GUARANTEE (section 23): this script only reads a
local CSV and writes local output files. It never opens a network
socket, never sends a packet toward src_ip/dst_ip, never completes a
handshake, and has no return path to the monitored network -- "replay"
here means replaying already-captured flow records into the local
detection pipeline, exactly like a one-way tap/data-diode feed would.

Run:
    python scripts/replay_stream.py --input data/flow_features.csv --rate 200
    python scripts/replay_stream.py --input data/flow_features.csv --rate 0   # as fast as possible
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from ml_dl.predict_interface import predict, reset_history
from ml_dl.alerts import build_alert
from ml_dl.config import ROOT


def replay(input_csv: str, rate: float = 0.0, limit: int | None = None,
           out_path: str | None = None, verbose: bool = True):
    """
    rate: target flows/sec. 0 = process as fast as possible (no
    pacing/sleeping) -- used to measure max sustained throughput.
    Positive rate paces delivery to simulate a live feed; if
    processing a flow takes longer than the inter-arrival budget for
    that rate, it is counted as a soft "backlog" flow rather than
    silently dropped or silently blocking (see summary output).
    """
    reset_history()
    df = pd.read_csv(input_csv)
    if limit:
        df = df.iloc[:limit]

    out_path = out_path or str(ROOT / "reports" / "stream_alerts.jsonl")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    interval = (1.0 / rate) if rate > 0 else 0.0
    latencies = []
    n_alerts = 0
    n_backlogged = 0
    start = time.perf_counter()
    next_due = start

    with open(out_path, "w") as out_f:
        for i, flow in enumerate(df.to_dict("records")):
            if interval:
                now = time.perf_counter()
                sleep_for = next_due - now
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    n_backlogged += 1  # couldn't keep up with the target rate
                next_due += interval

            t0 = time.perf_counter()
            result = predict(flow)
            latencies.append(time.perf_counter() - t0)

            if result["dl_verdict"] != "benign":
                alert = build_alert(flow, result)
                out_f.write(json.dumps(alert) + "\n")
                n_alerts += 1

            if verbose and (i + 1) % max(1, len(df) // 10) == 0:
                print(f"  processed {i + 1}/{len(df)} flows, "
                      f"{n_alerts} alerts so far")

    total_time = time.perf_counter() - start
    achieved_rate = len(df) / total_time if total_time > 0 else float("inf")
    latencies_ms = np.array(latencies) * 1000.0

    summary = {
        "input_rows": len(df),
        "target_rate_flows_per_sec": rate or None,
        "achieved_throughput_flows_per_sec": float(achieved_rate),
        "total_time_sec": float(total_time),
        "alerts_generated": n_alerts,
        "backlogged_flows": n_backlogged,
        "latency_ms": {
            "mean": float(latencies_ms.mean()),
            "p50": float(np.percentile(latencies_ms, 50)),
            "p95": float(np.percentile(latencies_ms, 95)),
            "p99": float(np.percentile(latencies_ms, 99)),
            "max": float(latencies_ms.max()),
        },
    }

    if verbose:
        print(f"\n=== Streaming replay summary ===")
        print(f"Processed {summary['input_rows']} flows in {total_time:.2f}s "
              f"({achieved_rate:.1f} flows/sec achieved)")
        if rate:
            print(f"Target rate: {rate} flows/sec "
                  f"({'kept up' if n_backlogged == 0 else f'{n_backlogged} flows fell behind schedule'})")
        print(f"Alerts written to {out_path} ({n_alerts} non-benign flows)")
        print(f"Per-flow end-to-end latency (ms): {summary['latency_ms']}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, default=str(ROOT / "data" / "flow_features.csv"))
    parser.add_argument("--rate", type=float, default=0.0, help="target flows/sec, 0 = unpaced")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    replay(args.input, rate=args.rate, limit=args.limit, out_path=args.out)
