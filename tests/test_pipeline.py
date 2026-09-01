"""
End-to-end smoke test: generate synthetic data, train everything, run
one prediction, and check the output matches docs/interfaces.md.

Run with: pytest tests/test_pipeline.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from scripts.generate_synthetic_data import generate
from ml_dl.config import DATA_PATH, FEATURE_COLUMNS
from ml_dl import train_xgboost, light_dl, heavy_dl
from ml_dl.predict_interface import predict


def setup_module(module):
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate(500, seed=1)
    df.to_csv(DATA_PATH, index=False)

    train_xgboost.train()

    light_model = light_dl.build_light_dl()
    light_dl.save(light_model)

    heavy_model = heavy_dl.build_heavy_dl()
    heavy_dl.save(heavy_model)


def test_predict_output_shape():
    df = pd.read_csv(DATA_PATH)
    flow = df.iloc[0].to_dict()

    result = predict(flow)

    expected_keys = {
        "ml_verdict", "ml_confidence", "dl_verdict",
        "dl_confidence", "model_used", "shap_evidence",
    }
    assert set(result.keys()) == expected_keys
    assert result["model_used"] in ("light", "heavy")
    assert 0.0 <= result["ml_confidence"] <= 1.0
    assert 0.0 <= result["dl_confidence"] <= 1.0
    assert isinstance(result["shap_evidence"], list)
