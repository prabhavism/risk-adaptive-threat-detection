"""
Risk-adaptive routing: XGBoost confidence decides whether a flow goes
to the fast Light DL model or the more expensive Heavy DL model. No
flow skips DL verification entirely -- theta only chooses *which*
verifier looks at it, which is what keeps the "light" / "heavy" values
in docs/interfaces.md's predict() contract exactly two options.
"""
import numpy as np

from ml_dl.config import DEFAULT_THETA


def route(ml_confidence: float, theta: float = DEFAULT_THETA) -> str:
    """
    Returns "light" if ml_confidence >= theta (XGBoost is already
    fairly sure), else "heavy" (ambiguous -- needs the temporal model).
    """
    return "light" if ml_confidence >= theta else "heavy"


def routing_stats(confidences, theta: float) -> dict:
    """
    Given an array of XGBoost confidences and a candidate theta, report
    what fraction of flows would go to each path. Used by
    ml_dl/tune_threshold.py and scripts/benchmark.py to show the
    throughput/accuracy trade-off at different theta values.
    """
    confidences = np.asarray(confidences)
    light_mask = confidences >= theta
    n = len(confidences)
    return {
        "theta": float(theta),
        "n_flows": int(n),
        "pct_light": float(light_mask.mean()) if n else 0.0,
        "pct_heavy": float((~light_mask).mean()) if n else 0.0,
        "n_light": int(light_mask.sum()),
        "n_heavy": int((~light_mask).sum()),
    }
