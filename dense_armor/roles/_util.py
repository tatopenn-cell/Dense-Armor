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
