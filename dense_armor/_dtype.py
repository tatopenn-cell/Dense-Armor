"""The default floating dtype of the library: float32, unless the user asks for float64."""

import numpy as np

_FLOATS = (np.dtype(np.float32), np.dtype(np.float64))
_default = np.dtype(np.float32)


def get_default_dtype() -> np.dtype:
    """The dtype the classes use for their arrays: ``float32`` by default."""
    return _default


def set_default_dtype(dtype) -> None:
    """Set the default dtype of the library to ``"float32"`` or ``"float64"``."""
    global _default
    d = np.dtype(dtype)
    if d not in _FLOATS:
        raise ValueError(f"default dtype must be float32 or float64, got {dtype!r}")
    _default = d


def x64_context(enabled: bool = True):
    """Context manager that sets x64 inside the block only, on any version of JAX."""
    import jax

    ctx = getattr(jax, "enable_x64", None)
    if ctx is None:
        from jax.experimental import enable_x64 as ctx
    return ctx(enabled)
