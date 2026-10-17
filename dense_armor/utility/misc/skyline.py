"""Skyline of a sliding window, from Börzsönyi et al. (2001) and Tao and Papadias (2006).

Two definitions are used:

- dominance (Börzsönyi et al. 2001, Section 2, page 2): a point `p`
  dominates `q`, written `p << q`, if `p[d] <= q[d]` for every dimension
  `d` and `p[d] < q[d]` for at least one dimension. Ties are not
  dominance.
- skyline: the set of points of the window not dominated by any other
  point of the window.

The window maintenance is the one of Tao and Papadias (2006, Section 3,
page 3): when a new point arrives and the window is full, the oldest
point leaves and the new point enters; the skyline is recomputed on the
new window with the block nested loops (BNL) algorithm of Börzsönyi et
al. (2001, Section 3.3, page 4). The naive recomputation is O(W^2) per
update for a window of size W; the paper also gives an incremental
algorithm, which is not implemented here, my choice, because W is small
in the test and the naive algorithm is easy to verify against the direct
definition.
"""

from collections import deque
from collections.abc import Sequence
from typing import Any

import numpy as np

from dense_armor.roles import Root


def _dominates(p: np.ndarray, q: np.ndarray) -> bool:
    """True if ``p`` dominates ``q`` (Börzsönyi et al. 2001, Def. 1)."""
    return bool(np.all(p <= q) and np.any(p < q))


def _skyline(points: Sequence[np.ndarray]) -> list[int]:
    """Return the indices of the points not dominated by any other point."""
    n = len(points)
    keep = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i != j and _dominates(points[j], points[i]):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


class Skyline(Root):
    """Skyline of a sliding window over a stream of points.

    A point is a ``dict`` mapping a dimension name to a value. The
    window keeps the last ``window`` points; ``skyline()`` returns the
    points not dominated by any other point of the window, with their
    original index in the window.

    Args:
        window: window length, strictly positive.
        dims: the dimension names, in order. ``None`` takes the sorted
            keys of the first point seen.

    Raises:
        ValueError: if ``window < 1``.

    Examples:
        >>> from dense_armor.utility.misc.skyline import Skyline
        >>> s = Skyline(window=3)
        >>> _ = s.update({"x": 1.0, "y": 5.0})
        >>> _ = s.update({"x": 3.0, "y": 3.0})
        >>> _ = s.update({"x": 5.0, "y": 1.0})
        >>> len(s.skyline())
        3

    References:
        Börzsönyi, S., Kossmann, D., Stocker, K. (2001). The Skyline
        Operator. In ICDE 2001. Section 2 page 2 (dominance),
        Section 3.3 page 4 (block nested loops).
        Tao, Y., Papadias, D. (2006). Maintaining Sliding Window
        Skylines. In ICDE 2006. Section 3 page 3 (sliding window
        maintenance).
    """

    budget_s: float = 1.5e-2
    memory_class: str = "O(window * dims)"

    def __init__(self, window: int = 10, dims: Sequence[str] | None = None) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.window = int(window)
        self.dims = list(dims) if dims is not None else None
        self._buffer: deque = deque(maxlen=self.window)
        self.n_seen_: int = 0
        self.n_missing: int = 0

    def _as_point(self, p: Any) -> np.ndarray | None:
        if isinstance(p, dict):
            d = p
        else:
            to_dict = getattr(p, "to_dict", None)
            if callable(to_dict):
                d = to_dict()
            else:
                return None
        if self.dims is None:
            self.dims = sorted(d)
        try:
            arr = np.asarray([float(d[k]) for k in self.dims], dtype=float)
        except (KeyError, TypeError, ValueError):
            return None
        if not np.all(np.isfinite(arr)):
            return None
        return arr

    def update(self, point: Any) -> "Skyline":
        """Push one point, dropping the oldest when the window is full."""
        p = self._as_point(point)
        if p is None:
            self.n_missing += 1
            return self
        self._buffer.append(p)
        self.n_seen_ += 1
        return self

    def update_many(self, points: Any) -> "Skyline":
        """Push several points in order."""
        for p in points:
            _ = self.update(p)
        return self

    def skyline(self) -> list[int]:
        """Indices (in the current window) of the non-dominated points.

        The indices refer to the position of the point in the current
        window, from 0 (oldest) to ``len(window) - 1`` (newest).
        """
        pts = list(self._buffer)
        return _skyline(pts)

    def skyline_points(self) -> list[np.ndarray]:
        """The non-dominated points, as arrays in the order of ``dims``."""
        return [np.asarray(self._buffer[i]) for i in self.skyline()]
