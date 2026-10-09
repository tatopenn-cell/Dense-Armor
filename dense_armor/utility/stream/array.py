"""Read a numpy or JAX array as a stream of :class:`Signal` samples.

Examples:
    >>> import numpy as np
    >>> from dense_armor.utility.stream import iter_array
    >>> X = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> for sig, y in iter_array(X, names=["a", "b"], units=["m", "s"]):
    ...     print(sig["a"], sig["b"], sig.t, y)
    1.0 2.0 0.0 None
    3.0 4.0 1.0 None
"""

from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

from dense_armor.roles import Signal


def iter_array(
    X: Any,
    y: Any = None,
    t: Any = None,
    names: Iterable[str] | None = None,
    units: Iterable[str] | None = None,
) -> Iterator[tuple[Signal, Any]]:
    """Iterate rows of an array as ``(Signal, y)`` pairs.

    Args:
        X: array of shape ``(n, d)`` or ``(n,)``.
        y: optional labels of shape ``(n,)``.
        t: optional timestamps in seconds. If ``None``, ``t = float(i)``.
        names: channel names in column order.
        units: channel units.

    Yields:
        ``(signal, label)`` pairs in time order.

    Raises:
        ValueError: if a length does not match.
    """
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"X must be 1D or 2D, got shape {arr.shape}")
    n, d = arr.shape
    name_list = [f"x{i}" for i in range(d)] if names is None else list(names)
    if len(name_list) != d:
        raise ValueError(f"names has {len(name_list)} entries, X has {d} columns")
    unit_list = ["" for _ in range(d)] if units is None else list(units)
    if len(unit_list) != d:
        raise ValueError(f"units has {len(unit_list)} entries, X has {d} columns")
    y_arr = None if y is None else np.asarray(y)
    if y_arr is not None and len(y_arr) != n:
        raise ValueError(f"y has {len(y_arr)} entries, X has {n} rows")
    if t is None:
        t_arr = np.arange(n, dtype=float)
    else:
        t_arr = np.asarray(t, dtype=float)
        if len(t_arr) != n:
            raise ValueError(f"t has {len(t_arr)} entries, X has {n} rows")
    for i in range(n):
        sig = Signal(
            values=arr[i],
            names=name_list,
            units=unit_list,
            t=float(t_arr[i]),
        )
        yield sig, (None if y_arr is None else y_arr[i])
