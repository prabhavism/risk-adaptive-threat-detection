"""
Risk-adaptive routing: XGBoost confidence determines which model handles a flow.

Three tiers:
    confidence >= theta_high  →  "light"  (fast verification — XGBoost is confident)
    theta_low  <= conf < theta_high  →  "light"  (Light DL verifies)
    conf < theta_low          →  "heavy"  (Heavy DL for hard / ambiguous flows)

Note: predict_interface.py handles the "xgb_only" shortcut for conf >= theta_high
(it skips DL entirely). This function returns "light" | "heavy" only, and is used
by both predict_interface and the test suite for routing boundary tests.
"""
from ml_dl.config import DEFAULT_THETA


def route(ml_confidence: float, theta: float = DEFAULT_THETA) -> str:
    """
    Returns "light" if ml_confidence >= theta, else "heavy".

    Parameters
    ----------
    ml_confidence : float
        The calibrated probability from XGBoost for the predicted class (0–1).
    theta : float
        The routing threshold. Defaults to DEFAULT_THETA (= THETA_HIGH = 0.90).
        Loaded from models/threshold.pkl at inference time by predict_interface.py.

    Returns
    -------
    "light" or "heavy"
    """
    return "light" if ml_confidence >= theta else "heavy"
