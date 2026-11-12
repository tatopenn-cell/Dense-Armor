"""Shared helpers for the factorization machines."""

import math


def _sigmoid(z: float) -> float:
    """Numerically stable logistic function."""
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)
