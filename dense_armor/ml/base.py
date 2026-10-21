"""Base class of the machine learning algorithms of Dense-Armor.

A class implements the algorithm of the paper on the whole dataset:
``fit`` takes the whole array and the results are attributes with a trailing
underscore. It has no ``learn_one``: the online class of the library is the
other side of the pair, and ``benchmarks/ml`` saves the difference.
"""

from __future__ import annotations

import numpy as np

from dense_armor.roles import Root


class MLBase(Root):
    """Root of the classes: parameters, ``clone`` and ``state_dict``.

    Example:
        >>> class Mean(MLBase):
        ...     def fit(self, X, y=None):
        ...         self.mean_ = float(np.mean(X))
        ...         return self
        >>> Mean().fit([1.0, 2.0, 3.0]).mean_
        2.0
    """

    def fit(self, X, y=None):  # type: ignore[override]
        """Learn from the whole dataset; returns ``self``."""
        raise NotImplementedError


def finite(x) -> tuple[np.ndarray, int]:
    """Return the finite values of ``x`` as float64 and the number dropped.

    Example:
        >>> finite([1.0, float("nan"), 3.0])[1]
        1
    """
    a = np.asarray(x, dtype=float).ravel()
    ok = np.isfinite(a)
    return a[ok], int((~ok).sum())


def finite_pairs(x, y) -> tuple[np.ndarray, np.ndarray, int]:
    """Return the pairs where both values are finite and the number dropped.

    Example:
        >>> a, b, k = finite_pairs([1.0, float("nan"), 3.0], [1.0, 2.0, 3.0])
        >>> a.tolist(), k
        ([1.0, 3.0], 1)
    """
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok], int((~ok).sum())
