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


def as_default_array(x, copy: bool = False) -> np.ndarray:
    """``x`` as an array of the default dtype, float64 if a finite value does not fit in float32.

    With the default dtype at float32, a value above about 3.4e38 would become ``inf``
    and be taken for a missing value. Such an input keeps float64 instead.
    """
    a = np.asarray(x)
    d = _default
    if d == np.float32 and a.dtype != np.float32:
        with np.errstate(over="ignore", invalid="ignore"):
            try:
                a32 = a.astype(np.float32)
            except (TypeError, ValueError):
                return a.astype(d, copy=copy)
        if np.isinf(a32).any() and not np.isinf(a).any():
            return a.astype(np.float64, copy=copy)
        return a32
    return a.astype(d, copy=copy)
