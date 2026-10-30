"""Univariate Gaussian with online mean and variance (Welford).

The distribution keeps the running count, mean and second central
moment. ``update`` follows Welford (1962), ``revert`` is its exact
inverse. Bishop, Pattern Recognition and Machine Learning, Section
2.3, eq. 2.42 (density). The cumulative distribution uses the
error function, eq. 1.48 in the same section.
"""

from math import erf, pi, sqrt
from typing import Any

import numpy as np


def _two_sum(a, b):
    """Error-free transformation ``a + b = s + e`` in the dtype of ``a``."""
    s = a + b
    bb = s - a
    e = (a - (s - bb)) + (b - bb)
    return s, e


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
        Healy, J. (2022). Accurate and consistent calculation of the
            mean and the variance in Monte-Carlo simulations.
            arXiv:2206.10662v7, Algorithm 2 (bibliographic only: the PDF
            is not in the task folder).
    """

    def __init__(
        self, mean: float = 0.0, var: float = 1.0, n: float = 0.0, dtype: Any = None
    ) -> None:
        if var <= 0.0:
            raise ValueError(f"var must be > 0, got {var}")
        if n < 0.0:
            raise ValueError(f"n must be >= 0, got {n}")
        self.dtype = dtype
        self._var0 = float(var)
        self._n = float(n)
        self._K: Any = None
        self._s1: Any = None
        self._c1: Any = None
        self._s2: Any = None
        self._c2: Any = None
        if dtype is not None and np.dtype(dtype) == np.float32:
            self._f32 = True
            self._K = np.float32(mean) if n > 0.0 else None
            self._s1 = np.float32(0.0)
            self._c1 = np.float32(0.0)
            self._s2 = np.float32(np.float32(var) * np.float32(n))
            self._c2 = np.float32(0.0)
        else:
            self._f32 = False
            self._mean = float(mean)
            self._m2 = float(var) * float(n)

    def update(self, x: float, w: float = 1.0) -> "Gaussian":
        """Add one observation.

        With ``dtype=None`` (default) the update is Welford's, eq. 2.42
        of Bishop, and the result is bit for bit the one of the previous
        version. With ``dtype=numpy.float32`` the update is the
        shifted-data algorithm with ``K = x[0]`` (Healy 2022, Algorithm 2)
        applied online: the sums of ``y = x - K`` and of ``y**2`` are
        accumulated with ``TwoSum``, and the variance follows from the
        two-term formula. In this mode ``w`` must be 1.
        """
        if w <= 0.0:
            raise ValueError(f"w must be > 0, got {w}")
        x = float(x)
        if not self._f32:
            n_new = self._n + w
            delta = x - self._mean
            self._mean = self._mean + w * delta / n_new
            self._m2 = self._m2 + w * delta * (x - self._mean)
            self._n = n_new
            return self
        if w != 1.0:
            raise ValueError("dtype=numpy.float32 supports only w=1 in Gaussian.update")
        if self._K is None:
            self._K = np.float32(x)
        xf = np.float32(x)
        y = np.float32(xf - self._K)
        s1, e1 = _two_sum(self._s1, y)
        self._c1 = np.float32(self._c1 + e1)
        self._s1 = s1
        yy = np.float32(y * y)
        s2, e2 = _two_sum(self._s2, yy)
        self._c2 = np.float32(self._c2 + e2)
        self._s2 = s2
        self._n += 1.0
        return self

    def revert(self, x: float, w: float = 1.0) -> "Gaussian":
        """Remove one observation. Exact inverse of :meth:`update`.

        With ``dtype=numpy.float32`` this method is not available: the
        shifted compensated sums do not have an exact inverse, and the
        keyword's contract does not include one.
        """
        if self._f32:
            raise NotImplementedError(
                "Gaussian.revert is not available with dtype=numpy.float32"
            )
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
        if not self._f32:
            return self._mean
        if self._n <= 0.0:
            return float(self._K) if self._K is not None else 0.0
        assert self._K is not None
        n = np.float32(self._n)
        s1c = np.float32(self._s1 + self._c1)
        return float(self._K + np.float32(s1c / n))

    def variance(self) -> float:
        """Unbiased estimate of the variance, ``M2 / (n - 1)``."""
        if self._n < 2.0:
            raise ValueError("variance undefined for n < 2")
        if not self._f32:
            return self._m2 / (self._n - 1.0)
        n = np.float32(self._n)
        s1c = np.float32(self._s1 + self._c1)
        s2c = np.float32(self._s2 + self._c2)
        mean_y = np.float32(s1c / n)
        var_pop = np.float32(np.float32(s2c / n) - np.float32(mean_y * mean_y))
        factor = np.float32(n / np.float32(n - np.float32(1.0)))
        return float(np.float32(var_pop * factor))

    def n(self) -> float:
        return self._n

    def _var_biased(self) -> float:
        """Current ``m2 / n``, or the constructor's ``var`` when ``n == 0``."""
        if self._n <= 0.0:
            return self._var0
        if not self._f32:
            return self._m2 / self._n
        n = np.float32(self._n)
        s1c = np.float32(self._s1 + self._c1)
        s2c = np.float32(self._s2 + self._c2)
        mean_y = np.float32(s1c / n)
        return float(np.float32(np.float32(s2c / n) - np.float32(mean_y * mean_y)))

    def pdf(self, x: float) -> float:
        """Density at ``x``, Bishop eq. 2.42.

        Before any observation the density uses the ``var`` passed to the
        constructor; from the first observation onward it uses the
        maximum-likelihood estimate ``m2 / n``. The fallback to ``var0``
        keeps the constructor's ``var`` meaningful for an empty Gaussian.
        """
        var = self._var_biased()
        if var <= 0.0:
            return 0.0
        c = 1.0 / sqrt(2.0 * pi * var)
        m = self.mean()
        return float(c * np.exp(-0.5 * (float(x) - m) ** 2 / var))

    def cdf(self, x: float) -> float:
        """Cumulative distribution at ``x``, using the error function."""
        var = self._var_biased() if self._n > 0.0 else 0.0
        m = self.mean()
        if var <= 0.0:
            return 0.0 if x < m else 1.0
        z = (float(x) - m) / sqrt(2.0 * var)
        return float(0.5 * (1.0 + erf(z)))

    def mode(self) -> float:
        """Mode of the Gaussian, equal to the mean."""
        return self.mean()

    def sample(self, rng: np.random.Generator) -> float:
        """Draw one sample from the Gaussian with the current parameters."""
        var = self._var_biased() if self._n > 0.0 else 0.0
        return float(rng.normal(self.mean(), sqrt(max(var, 0.0))))

    def init_state(self) -> tuple:
        """Return the current state (n, mean, m2) as JAX scalars."""
        import jax.numpy as jnp

        return (
            jnp.asarray(self._n),
            jnp.asarray(self._mean),
            jnp.asarray(self._m2),
        )

    def step(self, state: tuple, v: object) -> tuple:
        """One pure Welford step: (n, mean, m2) + sample -> (state, mean).

        Pure function of its arguments, safe under ``jax.jit`` and
        ``jax.lax.scan``.
        """
        import jax.numpy as jnp

        n, mean, m2 = state
        v = jnp.asarray(v)
        n_new = n + 1.0
        delta = v - mean
        mean_new = mean + delta / n_new
        m2_new = m2 + delta * (v - mean_new)
        return (n_new, mean_new, m2_new), mean_new
