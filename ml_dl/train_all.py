"""
Runs the whole Person-2 pipeline end to end, in the right order, using
the shared time-based split so every stage sees the same train/val/test
rows:

    1. Train XGBoost                      (train_xgboost.py)
    2. Train Light DL on train/val         (light_dl.py)
    3. Build sequences + train Heavy DL    (heavy_dl.py, data_utils.py)
    4. Calibrate XGBoost confidence        (calibration.py)
    5. Tune routing threshold theta        (tune_threshold.py) -- on
       CALIBRATED validation probabilities, since that's what
       deployment (predict_interface.py) actually routes on
    6. Smoke-test SHAP explainability      (explainability.py)

Ordering note (previously a bug, fixed here): threshold tuning MUST
run after calibration, using the same calibrator, or the saved theta
is tuned against probabilities deployment doesn't actually use. See
ml_dl/tune_threshold.py's `calibrator` parameter.

Baselines (ml_dl/baselines.py) and the feature-group ablation
(ml_dl/ablation.py) are deliberately separate scripts -- they're
evaluation/research artifacts for the report, not something the
deployed predict() path depends on, so they don't need to block a
retrain.

Run:
    python -m ml_dl.train_all
"""
from ml_dl.config import DATA_PATH
from ml_dl.data_utils import xy, build_sequences
from ml_dl import train_xgboost, light_dl, heavy_dl, tune_threshold, calibration
from ml_dl.explainability import test_explainability


def main(data_path=DATA_PATH):
    print("### Stage 1/6 — XGBoost triage classifier ###")
    xgb_result = train_xgboost.train(data_path)
    model = xgb_result["model"]
    scaler = xgb_result["scaler"]
    train_df = xgb_result["splits"]["train"]
    val_df = xgb_result["splits"]["val"]
    test_df = xgb_result["splits"]["test"]

    print("\n### Stage 2/6 — Light DL (MLP) ###")
    X_train, y_train = xy(train_df)
    X_val, y_val = xy(val_df)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    light_model, _, light_val_metrics = light_dl.train_light_dl(
        X_train_scaled, y_train, X_val_scaled, y_val, verbose=2
    )
    light_dl.save(light_model)
    print(f"Light DL val accuracy: {light_val_metrics['val_accuracy']:.4f}")

    print("\n### Stage 3/6 — Heavy DL (GRU, temporal) ###")
    X_train_seq, y_train_seq = build_sequences(train_df)
    X_val_seq, y_val_seq = build_sequences(val_df)
    # Sequences are built from RAW features; scale them the same way as
    # the flat vectors so both DL models see comparable input ranges.
    n_feat = X_train_seq.shape[2]
    X_train_seq_scaled = scaler.transform(
        X_train_seq.reshape(-1, n_feat)
    ).reshape(X_train_seq.shape)
    X_val_seq_scaled = scaler.transform(
        X_val_seq.reshape(-1, n_feat)
    ).reshape(X_val_seq.shape)

    heavy_model, _, heavy_val_metrics = heavy_dl.train_heavy_dl(
        X_train_seq_scaled, y_train_seq, X_val_seq_scaled, y_val_seq, verbose=2
    )
    heavy_dl.save(heavy_model)
    print(f"Heavy DL val accuracy: {heavy_val_metrics['val_accuracy']:.4f}")

    print("\n### Stage 4/6 — XGBoost confidence calibration ###")
    X_test, y_test = xy(test_df)
    calibrator = calibration.fit_calibrator(model, X_val, y_val)
    calibration.save(calibrator)
    calib_result = calibration.evaluate_calibration(model, calibrator, X_test, y_test)
    print(f"Raw ECE: {calib_result['raw_xgboost_ece']:.4f}  "
          f"Calibrated ECE: {calib_result['calibrated_ece']:.4f}")

    print("\n### Stage 5/6 — Routing threshold tuning (on CALIBRATED probabilities) ###")
    best_theta, routing_results, routing_summary = tune_threshold.tune(
        model, scaler, light_model, heavy_model, val_df, calibrator=calibrator
    )

    print("\n### Stage 6/6 — SHAP explainability smoke test ###")
    evidence = test_explainability(model, X_val.iloc[:3])
    for i, ev in enumerate(evidence):
        print(f"  sample {i}: {ev}")

    print("\nAll models saved to models/. Pipeline complete.")
    print("Run `python -m ml_dl.baselines` and `python -m ml_dl.ablation` "
          "for the research-validation reports, and `python scripts/benchmark.py` "
          "/ `python scripts/replay_stream.py` for the performance + streaming reports.")
    return {
        "xgb_result": xgb_result,
        "light_val_metrics": light_val_metrics,
        "heavy_val_metrics": heavy_val_metrics,
        "theta": best_theta,
        "routing_results": routing_results,
        "routing_summary": routing_summary,
        "calibration": calib_result,
    }


if __name__ == "__main__":
    main()
