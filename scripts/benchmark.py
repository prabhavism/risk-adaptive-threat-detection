"""
End-to-end evaluation / benchmark script (sections 17-20, 28).

Measures, using the actual `predict()` entry point (so latency numbers
are true end-to-end: preprocessing + XGBoost + routing + DL +
explainability, not just one model's inference time):

    - Per-flow end-to-end latency (mean/p50/p95/p99/max), and split by
      XGBoost/Light DL/Heavy DL individually
    - Max sustained throughput (flows/sec), and whether that throughput
      would keep up with several target input rates (section 18)
    - Accuracy / macro-F1 / malicious recall / malicious->benign rate,
      overall and split by which path a flow was routed down
    - Per-threat-class precision/recall/F1 (section 20)
    - Actual routing percentage on held-out test data (not the
      validation data theta was picked on)
    - Compute-savings estimate vs. an "Always Heavy" policy, from
      actually-measured Light/Heavy latency (section 19)

All numbers are measured on whatever hardware this is run on -- run it
yourself and read reports/benchmark_results.json for the real figures;
nothing here is a stand-in for measurement (section 30).

Run (after ml_dl/train_all.py has produced models/*):
    python scripts/benchmark.py
"""
import json
import pickle
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report

from ml_dl.config import (
    CLASSES, DATA_PATH, XGB_MODEL_PATH, SCALER_PATH, THRESHOLD_PATH,
    CALIBRATOR_PATH, REPORT_DIR,
)
from ml_dl.data_utils import load_raw, time_based_split, xy, build_sequences
from ml_dl.predict_interface import predict, reset_history
from ml_dl.explainability import top_features
from ml_dl import light_dl, heavy_dl

TARGET_RATES = [100, 500, 1000, 5000]  # flows/sec, section 18


def _percentiles(latencies_s):
    arr = np.asarray(latencies_s) * 1000.0  # -> ms
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(np.max(arr)),
        "mean_ms": float(arr.mean()),
    }


def _malicious_metrics(y_true, y_pred):
    benign_idx = CLASSES.index("benign")
    mal_mask = y_true != benign_idx
    if mal_mask.sum() == 0:
        return None, None
    to_benign = ((y_pred == benign_idx) & mal_mask).sum() / mal_mask.sum()
    return float(1.0 - to_benign), float(to_benign)


def benchmark(data_path=DATA_PATH, n_samples: int = 500):
    with open(XGB_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(THRESHOLD_PATH, "rb") as f:
        theta = pickle.load(f)
    light_model = light_dl.load()
    heavy_model = heavy_dl.load()

    calibrator = None
    if CALIBRATOR_PATH.exists():
        with open(CALIBRATOR_PATH, "rb") as f:
            calibrator = pickle.load(f)

    df = load_raw(data_path)
    _, _, test_df = time_based_split(df)
    n = min(n_samples, len(test_df))
    bench_df = test_df.iloc[:n].reset_index(drop=True)

    X_test, y_test = xy(bench_df)
    label_idx = {c: i for i, c in enumerate(CLASSES)}

    # ------------------------------------------------------------------
    # 1. End-to-end predict() latency + verdicts, one flow at a time
    #    (this is the real deployment call, includes routing + SHAP)
    # ------------------------------------------------------------------
    reset_history()
    e2e_latencies = []
    model_used_list = []
    dl_preds = []
    for flow in bench_df.to_dict("records"):
        t0 = time.perf_counter()
        result = predict(flow)
        e2e_latencies.append(time.perf_counter() - t0)
        model_used_list.append(result["model_used"])
        dl_preds.append(label_idx[result["dl_verdict"]])

    model_used = np.array(model_used_list)
    dl_preds = np.array(dl_preds)
    light_mask = model_used == "light"
    heavy_mask = ~light_mask

    e2e_summary = _percentiles(e2e_latencies)
    light_e2e = _percentiles(np.array(e2e_latencies)[light_mask]) if light_mask.any() else None
    heavy_e2e = _percentiles(np.array(e2e_latencies)[heavy_mask]) if heavy_mask.any() else None

    # ------------------------------------------------------------------
    # 2. Component-level latency (individual model calls, single row)
    # ------------------------------------------------------------------
    X_scaled = scaler.transform(X_test)
    X_seq, y_seq_check = build_sequences(bench_df)
    assert (y_seq_check == y_test).all()
    n_feat = X_seq.shape[2]
    X_seq_scaled = scaler.transform(X_seq.reshape(-1, n_feat)).reshape(X_seq.shape)

    xgb_lat, light_lat, heavy_lat = [], [], []
    for i in range(n):
        t0 = time.perf_counter(); model.predict_proba(X_test.iloc[i:i + 1]); xgb_lat.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); light_model.predict(X_scaled[i:i + 1], verbose=0); light_lat.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); heavy_model.predict(X_seq_scaled[i:i + 1], verbose=0); heavy_lat.append(time.perf_counter() - t0)

    calibration_lat = []
    if calibrator is not None:
        for i in range(n):
            t0 = time.perf_counter()
            calibrator.predict_proba(X_test.iloc[i:i + 1])
            calibration_lat.append(time.perf_counter() - t0)

    shap_lat = []
    for i in range(n):
        t0 = time.perf_counter()
        top_features(model, X_test.iloc[i:i + 1])
        shap_lat.append(time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # 3. Accuracy / F1 / malicious metrics, overall + per branch
    # ------------------------------------------------------------------
    overall_acc = accuracy_score(y_test, dl_preds)
    overall_f1 = f1_score(y_test, dl_preds, average="macro", zero_division=0)
    mal_recall, mal_to_benign = _malicious_metrics(y_test, dl_preds)
    per_class_report = classification_report(
        y_test, dl_preds, target_names=CLASSES, output_dict=True, zero_division=0
    )

    branch_metrics = {}
    for name, mask in [("light", light_mask), ("heavy", heavy_mask)]:
        if mask.sum():
            branch_metrics[name] = {
                "n_flows": int(mask.sum()),
                "accuracy": float(accuracy_score(y_test[mask], dl_preds[mask])),
                "macro_f1": float(f1_score(y_test[mask], dl_preds[mask], average="macro",
                                            labels=list(range(len(CLASSES))), zero_division=0)),
            }

    # ------------------------------------------------------------------
    # 4. Max sustained throughput + target-rate feasibility (section 18)
    # ------------------------------------------------------------------
    total_time = sum(e2e_latencies)
    max_throughput = n / total_time if total_time > 0 else float("inf")

    throughput_rows = []
    for rate in TARGET_RATES:
        window_sec = 10.0
        offered = rate * window_sec
        capacity = max_throughput * window_sec
        processed = min(offered, capacity)
        dropped = max(0.0, offered - capacity)
        throughput_rows.append({
            "target_rate_flows_per_sec": rate,
            "offered_flows_in_10s_window": offered,
            "processed_flows": processed,
            "dropped_flows": dropped,
            "sustainable": bool(max_throughput >= rate),
            "mean_latency_ms": e2e_summary["mean_ms"],
            "p95_latency_ms": e2e_summary["p95_ms"],
            "p99_latency_ms": e2e_summary["p99_ms"],
            "pct_light": float(light_mask.mean()),
            "pct_heavy": float(heavy_mask.mean()),
        })

    # ------------------------------------------------------------------
    # 5. Compute-savings vs. Always-Heavy (section 19)
    # ------------------------------------------------------------------
    mean_heavy_ms = np.mean(heavy_lat) * 1000.0
    mean_light_ms = np.mean(light_lat) * 1000.0
    mean_xgb_ms = np.mean(xgb_lat) * 1000.0
    always_heavy_ms_per_flow = mean_xgb_ms + mean_heavy_ms
    adaptive_ms_per_flow = mean_xgb_ms + (
        light_mask.mean() * mean_light_ms + heavy_mask.mean() * mean_heavy_ms
    )
    heavy_call_reduction = float(1 - heavy_mask.mean())
    latency_reduction_pct = float(1 - adaptive_ms_per_flow / always_heavy_ms_per_flow) if always_heavy_ms_per_flow else None

    results = {
        "disclaimer": (
            "Synthetic data is used for pipeline validation only. Final "
            "model performance must be reported using realistic labeled "
            "traffic."
        ),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "python": platform.python_version(),
        },
        "n_benchmarked_flows": n,
        "theta": float(theta),
        "test_set_routing": {
            "pct_light": float(light_mask.mean()),
            "pct_heavy": float(heavy_mask.mean()),
        },
        "end_to_end_latency": {
            "overall": e2e_summary,
            "light_branch": light_e2e,
            "heavy_branch": heavy_e2e,
        },
        "component_latency": {
            "xgboost": _percentiles(xgb_lat),
            "calibration": _percentiles(calibration_lat) if calibration_lat else None,
            "shap": _percentiles(shap_lat),
            "light_dl": _percentiles(light_lat),
            "heavy_dl": _percentiles(heavy_lat),
        },
        "accuracy": {
            "overall_accuracy": float(overall_acc),
            "overall_macro_f1": float(overall_f1),
            "malicious_recall": mal_recall,
            "malicious_to_benign_rate": mal_to_benign,
            "per_branch": branch_metrics,
            "per_class": per_class_report,
        },
        "throughput": {
            "max_sustained_flows_per_sec": float(max_throughput),
            "target_rate_tests": throughput_rows,
        },
        "compute_savings_vs_always_heavy": {
            "always_heavy_ms_per_flow": float(always_heavy_ms_per_flow),
            "risk_adaptive_ms_per_flow": float(adaptive_ms_per_flow),
            "heavy_call_reduction_pct": heavy_call_reduction,
            "latency_reduction_pct": latency_reduction_pct,
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_DIR / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(throughput_rows).to_csv(REPORT_DIR / "throughput_results.csv", index=False)

    _print_summary(results)
    return results


def _print_summary(r):
    print("*** Synthetic data is used for pipeline validation only. ***")
    print("*** Final model performance must be reported using realistic labeled traffic. ***\n")
    print(f"Hardware: {r['hardware']['platform']}")
    print(f"Benchmarked {r['n_benchmarked_flows']} flows, theta={r['theta']:.3f}")
    print(f"Test-set routing: {r['test_set_routing']['pct_light']:.1%} light, "
          f"{r['test_set_routing']['pct_heavy']:.1%} heavy")

    print(f"\nEnd-to-end latency (predict(), full pipeline):")
    print(f"  overall : {r['end_to_end_latency']['overall']}")
    if r['end_to_end_latency']['light_branch']:
        print(f"  light   : {r['end_to_end_latency']['light_branch']}")
    if r['end_to_end_latency']['heavy_branch']:
        print(f"  heavy   : {r['end_to_end_latency']['heavy_branch']}")

    print(f"\nAccuracy: {r['accuracy']['overall_accuracy']:.4f}, "
          f"macro-F1: {r['accuracy']['overall_macro_f1']:.4f}")
    if r['accuracy']['malicious_recall'] is not None:
        print(f"Malicious recall: {r['accuracy']['malicious_recall']:.4f}, "
              f"malicious->benign rate: {r['accuracy']['malicious_to_benign_rate']:.4f}")

    print(f"\nMax sustained throughput: {r['throughput']['max_sustained_flows_per_sec']:.1f} flows/sec")
    for row in r['throughput']['target_rate_tests']:
        status = "OK" if row["sustainable"] else f"DROPS ~{row['dropped_flows']:.0f} flows/10s window"
        print(f"  target {row['target_rate_flows_per_sec']:>5} flows/sec -> {status}")

    cs = r['compute_savings_vs_always_heavy']
    print(f"\nAlways-Heavy: {cs['always_heavy_ms_per_flow']:.3f} ms/flow")
    print(f"Risk-Adaptive: {cs['risk_adaptive_ms_per_flow']:.3f} ms/flow "
          f"({cs['heavy_call_reduction_pct']:.1%} fewer Heavy DL calls, "
          f"{cs['latency_reduction_pct']:.1%} lower mean latency)")


if __name__ == "__main__":
    benchmark()
