"""Univariate Gaussian with online mean and variance (Welford).

The distribution keeps the running count, mean and second central
moment. ``update`` follows Welford (1962), ``revert`` is its exact
inverse. Bishop, Pattern Recognition and Machine Learning, Section
2.3, eq. 2.42 (density). The cumulative distribution uses the
error function, eq. 1.48 in the same section.
"""

from math import erf, pi, sqrt

import numpy as np


class Gaussian:
    """Univariate Gaussian updated online by Welford's algorithm.

    Args:
        mean: initial mean. Default 0.
        var: initial variance, strictly positive. Default 1.
        n: weight of the initial state. Default 0 (uninformative).

    Raises:
        ValueError: on a non-positive ``var`` or a negative ``n``.

    Examples:
        >>> from dense_armor.utility.proba import Gaussian
        >>> g = Gaussian()
        >>> for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        ...     _ = g.update(v)
        >>> round(g.mean(), 6)
        3.0
        >>> round(g.variance(), 6)
        2.5

    References:
        Welford, B. P. (1962). Technometrics 4(3), 419-420.
        Bishop, C. M. (2006). Section 2.3, eq. 2.42.
    """

    def __init__(self, mean: float = 0.0, var: float = 1.0, n: float = 0.0) -> None:
        if var <= 0.0:
            raise ValueError(f"var must be > 0, got {var}")
        if n < 0.0:
            raise ValueError(f"n must be >= 0, got {n}")
        self._mean = float(mean)
        self._m2 = float(var) * float(n)
        self._n = float(n)

    def update(self, x: float, w: float = 1.0) -> "Gaussian":
        """Add one observation. Welford's update, Bishop eq. 2.42 context."""
        if w <= 0.0:
            raise ValueError(f"w must be > 0, got {w}")
        x = float(x)
        n_new = self._n + w
        delta = x - self._mean
        self._mean = self._mean + w * delta / n_new
        self._m2 = self._m2 + w * delta * (x - self._mean)
        self._n = n_new
        return self

    def revert(self, x: float, w: float = 1.0) -> "Gaussian":
        """Remove one observation. Exact inverse of :meth:`update`."""
        if w <= 0.0:
            raise ValueError(f"w must be > 0, got {w}")
        if self._n - w <= 0.0:
            raise ValueError("revert would make n non-positive")
        x = float(x)
        n_new = self._n - w
        mean_new = (self._n * self._mean - w * x) / n_new
        self._m2 = self._m2 - w * (x - self._mean) * (x - mean_new)
        self._mean = mean_new
        self._n = n_new
        return self

    def mean(self) -> float:
        return self._mean

    def variance(self) -> float:
        """Unbiased estimate of the variance, ``M2 / (n - 1)``."""
        if self._n < 2.0:
            raise ValueError("variance undefined for n < 2")
        return self._m2 / (self._n - 1.0)

    def n(self) -> float:
        return self._n

    def pdf(self, x: float) -> float:
        """Density at ``x``, Bishop eq. 2.42."""
        var = self._m2 / max(self._n, 1.0)
        if var <= 0.0:
            return 0.0
        c = 1.0 / sqrt(2.0 * pi * var)
        return float(c * np.exp(-0.5 * (float(x) - self._mean) ** 2 / var))

    def cdf(self, x: float) -> float:
        """Cumulative distribution at ``x``, using the error function."""
        var = self._m2 / max(self._n, 1.0)
        if var <= 0.0:
            return 0.0 if x < self._mean else 1.0
        z = (float(x) - self._mean) / sqrt(2.0 * var)
        return float(0.5 * (1.0 + erf(z)))

    def mode(self) -> float:
        """Mode of the Gaussian, equal to the mean."""
        return self._mean

    def sample(self, rng: np.random.Generator) -> float:
        """Draw one sample from the Gaussian with the current parameters."""
        var = self._m2 / max(self._n, 1.0)
        return float(rng.normal(self._mean, sqrt(max(var, 0.0))))
