"""
Risk-adaptive routing: XGBoost confidence decides whether a flow goes to
the fast Light DL model or the more expensive Heavy DL model. No flow
skips DL verification.
"""
from ml_dl.config import DEFAULT_THETA


def route(ml_confidence: float, theta: float = DEFAULT_THETA) -> str:
    """
    Returns "light" if ml_confidence >= theta, else "heavy".
    """
    return "light" if ml_confidence >= theta else "heavy"
