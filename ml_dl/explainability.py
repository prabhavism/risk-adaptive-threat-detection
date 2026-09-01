"""
SHAP-based explainability for the XGBoost triage model. Returns the
top-k contributing features for a single flow's prediction.
"""
import shap
import numpy as np

from ml_dl.config import FEATURE_COLUMNS


def top_features(model, flow_row, k: int = 3):
    """
    model: trained xgboost.XGBClassifier
    flow_row: 1-row DataFrame or array with FEATURE_COLUMNS
    Returns: list of {"feature": name, "value": shap_value}
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(flow_row)

    # shap_values shape depends on xgboost/shap version; handle multiclass
    values = np.array(shap_values)
    if values.ndim == 3:
        # (n_classes, n_samples, n_features) -> pick predicted class row
        predicted_class = int(np.argmax(model.predict_proba(flow_row)[0]))
        row = values[predicted_class][0]
    else:
        row = values[0]

    ranked = sorted(
        zip(FEATURE_COLUMNS, row), key=lambda x: abs(x[1]), reverse=True
    )
    return [{"feature": f, "value": float(v)} for f, v in ranked[:k]]
