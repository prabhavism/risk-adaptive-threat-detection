"""
SHAP-based explainability for the XGBoost triage model.

Key design: SHAPExplainer is instantiated ONCE (TreeExplainer construction
is expensive — O(n_trees)) and then .top_features() is called cheaply per
flow. Never instantiate inside a per-flow loop.

Usage:
    explainer = SHAPExplainer(calibrated_model)
    evidence  = explainer.top_features(row_df, k=5)
    importance = explainer.get_global_importance(X_background_df)
"""
import numpy as np
import pandas as pd
import shap

from ml_dl.config import FEATURE_COLUMNS


class SHAPExplainer:
    """
    Wraps a shap.TreeExplainer for repeated per-flow explanations.
    The underlying SHAP explainer is built once in __init__ and reused.
    """

    def __init__(self, model):
        """
        Parameters
        ----------
        model : CalibratedClassifierCV wrapping XGBClassifier
            The calibrated XGBoost model from models/model.pkl.
        """
        # Unwrap to get the raw XGBClassifier.
        # _raw_model is kept separately because self._explainer.model is SHAP's
        # internal TreeEnsemble object (no predict_proba), not the XGBClassifier.
        raw = self._unwrap(model)
        self._raw_model  = raw                       # XGBClassifier — has predict_proba
        self._explainer  = shap.TreeExplainer(raw)  # TreeEnsemble — SHAP internals only
        self._feature_names = list(FEATURE_COLUMNS)

    # ── Public API ────────────────────────────────────────────────────────────

    def top_features(self, flow_row: pd.DataFrame, k: int = 5) -> list[dict]:
        """
        Return the top-k SHAP contributors for a single flow prediction.

        Parameters
        ----------
        flow_row : pd.DataFrame
            Single-row DataFrame with FEATURE_COLUMNS as columns.
        k : int
            Number of top features to return.

        Returns
        -------
        list of {"feature": str, "value": float}
            Sorted by absolute SHAP value descending.
        """
        # Get predicted class from the raw XGBClassifier (has predict_proba)
        raw_probs = self._raw_model.predict_proba(flow_row)
        predicted_class = int(np.argmax(raw_probs[0]))

        shap_vals = self._explainer.shap_values(flow_row)
        values = np.array(shap_vals)
        n_features = len(self._feature_names)
        n_classes  = 7   # always 7 for this project

        # Determine row_vals: the per-feature SHAP values for the predicted class
        # SHAP returns different shapes depending on version:
        #   (n_classes, n_samples, n_features)  — old SHAP style
        #   (n_samples, n_features, n_classes)  — newer SHAP style
        #   (n_samples, n_classes, n_features)  — another variant
        #   (n_samples, n_features)             — binary / collapsed
        if values.ndim == 3:
            s = values.shape
            if s[0] == n_classes and s[1] == 1 and s[2] == n_features:
                # (n_classes, n_samples, n_features)
                row_vals = values[predicted_class, 0, :]
            elif s[0] == 1 and s[1] == n_features and s[2] == n_classes:
                # (n_samples, n_features, n_classes)
                row_vals = values[0, :, predicted_class]
            elif s[0] == 1 and s[1] == n_classes and s[2] == n_features:
                # (n_samples, n_classes, n_features)
                row_vals = values[0, predicted_class, :]
            elif s[0] == n_classes and s[2] == n_features:
                # (n_classes, any_samples, n_features)
                row_vals = values[predicted_class, 0, :]
            elif s[2] == n_classes and s[1] == n_features:
                # (any_samples, n_features, n_classes)
                row_vals = values[0, :, predicted_class]
            else:
                # Fallback: take the mean absolute SHAP across all dims except features
                row_vals = np.abs(values).reshape(-1, n_features).mean(axis=0)
        elif values.ndim == 2:
            # (n_samples, n_features) or (n_classes, n_features)
            row_vals = values[0] if values.shape[0] == 1 else values[predicted_class]
        else:
            row_vals = values.ravel()

        ranked = sorted(
            zip(self._feature_names, row_vals.tolist()),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        return [{"feature": f, "value": round(float(v), 5)} for f, v in ranked[:k]]

    def get_global_importance(
        self, X_background: pd.DataFrame, n_samples: int = 200
    ) -> list[dict]:
        """
        Compute mean absolute SHAP values across a background sample —
        useful for a dashboard 'most important features' summary panel.

        Parameters
        ----------
        X_background : pd.DataFrame
            Representative dataset (e.g. training set). Sampled to n_samples.
        n_samples : int
            Number of rows to sample for speed.

        Returns
        -------
        list of {"feature": str, "importance": float}
            Sorted by importance descending.
        """
        sample = X_background.sample(
            min(n_samples, len(X_background)), random_state=42
        )
        shap_vals = self._explainer.shap_values(sample)
        values = np.array(shap_vals)

        if values.ndim == 3:
            # Shape is (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features).
            # We want mean abs per feature: average over samples and classes.
            n_features = len(self._feature_names)
            if values.shape[1] == n_features:
                # (n_samples, n_features, n_classes) → mean over axis 0 and 2
                mean_abs = np.abs(values).mean(axis=(0, 2))
            else:
                # (n_classes, n_samples, n_features) → mean over axis 0 and 1
                mean_abs = np.abs(values).mean(axis=(0, 1))
        else:
            mean_abs = np.abs(values).mean(axis=0)

        ranked = sorted(
            zip(self._feature_names, mean_abs.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return [{"feature": f, "importance": round(float(v), 5)} for f, v in ranked]

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _unwrap(model):
        """
        Extract the raw XGBClassifier from any calibration wrapper.
        Handles:
          - CalibratedXGB  (our custom wrapper — has .xgb_model)
          - sklearn CalibratedClassifierCV  (has .estimator / .base_estimator)
          - Raw XGBClassifier (returned as-is)
        """
        # Our custom CalibratedXGB wrapper
        if hasattr(model, "xgb_model"):
            return model.xgb_model
        # sklearn >= 1.2 naming
        if hasattr(model, "estimator"):
            return model.estimator
        # older sklearn
        if hasattr(model, "base_estimator"):
            return model.base_estimator
        if hasattr(model, "calibrated_classifiers_"):
            cal = model.calibrated_classifiers_[0]
            if hasattr(cal, "estimator"):
                return cal.estimator
            if hasattr(cal, "base_estimator"):
                return cal.base_estimator
        return model  # already a raw model
