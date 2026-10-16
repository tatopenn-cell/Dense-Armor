"""Helpers shared by the ensemble classes."""

import math
from typing import Any


def as_dict(x: Any) -> dict:
    """Return ``x`` as a dict, from a dict or from a Signal-like object."""
    if hasattr(x, "to_dict"):
        return dict(x.to_dict())
    try:
        return dict(x)
    except (TypeError, ValueError):
        return {}


def finite(d: dict) -> bool:
    """True when every value of ``d`` is a finite float."""
    for v in d.values():
        if v is None:
            return False
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(fv):
            return False
    return True


def new_member(m: Any) -> Any:
    """Build a fresh member from a class, a factory or an instance."""
    if isinstance(m, type):
        return m()
    if callable(m) and not hasattr(m, "learn_one"):
        return m()
    if hasattr(m, "clone"):
        return m.clone()
    return m
