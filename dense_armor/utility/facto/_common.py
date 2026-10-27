"""Shared helpers for the factorization machines."""

import math


def _sigmoid(z: float) -> float:
    """Numerically stable logistic function."""
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-max(min(z, 700), -700)))
    e = math.exp(max(min(z, 700), -700))
    return e / (1.0 + e)
