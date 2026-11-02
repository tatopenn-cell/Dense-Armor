"""Small internal helpers shared by the role base classes."""

import inspect
from typing import Any


def accepts_kwarg(fn: Any, name: str) -> bool:
    """True if ``fn`` declares ``name`` or a ``**kwargs`` catch-all."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    if name in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def call_with_t(fn, *args, t=None, **kwargs):
    """Call ``fn(*args, t=t, **kwargs)`` only if it accepts ``t``."""
    if t is None or not accepts_kwarg(fn, "t"):
        return fn(*args, **kwargs)
    return fn(*args, t=t, **kwargs)


def call_learn_one(fn, x, y=None, t=None):
    """Call ``fn(x[, y][, t=t])``, passing ``y`` and ``t`` only if declared.

    ``y`` goes in only when it is not ``None`` and ``fn`` declares a ``y``
    parameter, so an unsupervised ``learn_one(x, t=None)`` never receives
    the label in place of the timestamp.
    """
    if y is not None and accepts_kwarg(fn, "y"):
        return call_with_t(fn, x, y, t=t)
    return call_with_t(fn, x, t=t)


def sum_budgets(weighted):
    """Sum ``weight * part.budget_s`` over ``(weight, part)`` pairs.

    The budget of a wrapper or a pipeline is what its parts cost for one
    sample. Returns ``None`` as soon as one part declares no ``budget_s``:
    a total that is missing a term would be too low, and the real-time
    check would accept a loop that can miss its deadline.
    """
    total = 0.0
    for weight, part in weighted:
        budget = getattr(part, "budget_s", None)
        if budget is None:
            return None
        total += weight * budget
    return total


def worst_memory_class(parts):
    """The largest ``memory_class`` of the parts.

    ``"O(window)"`` if any part is, ``"O(1)"`` if all are, and
    ``"unknown"`` if a part declares anything else or nothing.
    """
    classes = [getattr(p, "memory_class", "unknown") for p in parts]
    if any(c not in ("O(1)", "O(window)") for c in classes):
        return "unknown"
    return "O(window)" if "O(window)" in classes else "O(1)"
