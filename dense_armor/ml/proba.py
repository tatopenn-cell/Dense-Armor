"""Distributions fitted on the whole sample: the conjugate posterior in closed form.

References:
    Murphy, K. P. (2007). Conjugate Bayesian analysis of the Gaussian
        distribution (not in the folder, bibliographic only).
    The Beta posterior of a Bernoulli sample and the Dirichlet posterior of a
        categorical sample are textbook conjugate pairs: the prior counts plus
        the observed counts.
"""

from __future__ import annotations

import math

import numpy as np

from dense_armor.ml.base import MLBase

__all__ = ["Beta", "Gaussian", "Multinomial"]


class Beta(MLBase):
    """Beta posterior of a Bernoulli sample: ``a + successes``, ``b + failures``.

    Args:
        a: prior first parameter, strictly positive.
        b: prior second parameter, strictly positive.

    Example:
        >>> d = Beta().fit([1, 1, 0, 1])
        >>> d.a_, d.b_, round(d.mean(), 4)
        (4.0, 2.0, 0.6667)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0) -> None:
        if a <= 0 or b <= 0:
            raise ValueError("a and b must be strictly positive")
        self.a = a
        self.b = b

    def fit(self, X, y=None):  # type: ignore[override]
        x = np.asarray(X, dtype=float).ravel()
        self.a_ = float(self.a + np.sum(x > 0))
        self.b_ = float(self.b + np.sum(x <= 0))
        return self

    def mean(self) -> float:
        """``a / (a + b)``."""
        return self.a_ / (self.a_ + self.b_)

    def variance(self) -> float:
        """``a b / ((a + b)^2 (a + b + 1))``."""
        s = self.a_ + self.b_
        return self.a_ * self.b_ / (s * s * (s + 1.0))

    def mode(self) -> float:
        """``(a - 1) / (a + b - 2)`` when both parameters exceed 1, else the mean."""
        if self.a_ > 1 and self.b_ > 1:
            return (self.a_ - 1.0) / (self.a_ + self.b_ - 2.0)
        return self.mean()

    def pdf(self, x: float) -> float:
        """Density at ``x`` in ``[0, 1]``."""
        if not 0.0 <= x <= 1.0:
            return 0.0
        lb = (
            math.lgamma(self.a_ + self.b_) - math.lgamma(self.a_) - math.lgamma(self.b_)
        )
        return (
            math.exp(
                lb + (self.a_ - 1.0) * math.log(x) + (self.b_ - 1.0) * math.log1p(-x)
            )
            if 0.0 < x < 1.0
            else 0.0
        )

    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        """``n`` draws."""
        return np.random.default_rng(seed).beta(self.a_, self.b_, size=n)


class Gaussian(MLBase):
    """Gaussian of a sample: the mean and the variance (``n - 1``), pooled with a prior of weight ``n``.

    The prior counts as ``n`` observations with mean ``mean`` and variance ``var``
    (``n = 0`` ignores it); the pooled mean and variance combine the two groups. The
    pooling is our derivation (the parallel combination of Chan et al. 1979), not an
    equation of the reference.

    Args:
        mean: prior mean.
        var: prior variance, strictly positive.
        n: weight of the prior in observations, zero or more.

    Example:
        >>> d = Gaussian().fit([1.0, 2.0, 3.0, 4.0])
        >>> d.mean_, round(d.var_, 4), d.n_
        (2.5, 1.6667, 4.0)
    """

    def __init__(self, mean: float = 0.0, var: float = 1.0, n: float = 0.0) -> None:
        if var <= 0 or n < 0:
            raise ValueError("var must be positive and n not negative")
        self.mean = mean
        self.var = var
        self.n = n

    def fit(self, X, y=None):  # type: ignore[override]
        x = np.asarray(X, dtype=float).ravel()
        m = x.size
        if m == 0:
            self.n_, self.mean_, self.var_ = (
                float(self.n),
                float(self.mean),
                float(self.var),
            )
            self.m2_ = float(self.var) * max(self.n - 1.0, 0.0)
            return self
        mx = float(x.mean())
        m2x = float(((x - mx) ** 2).sum())
        n0 = float(self.n)
        tot = n0 + m
        self.n_ = tot
        self.mean_ = (n0 * self.mean + m * mx) / tot
        m2 = (
            (n0 - 1.0 if n0 > 1 else 0.0) * self.var
            + m2x
            + (n0 * m / tot) * (self.mean - mx) ** 2
        )
        self.m2_ = float(m2)
        self.var_ = float(m2 / (tot - 1.0)) if tot > 1 else float(self.var)
        return self

    def _ml_var(self) -> float:
        """Maximum-likelihood variance ``m2 / n`` (the prior ``var`` for an empty sample)."""
        return self.m2_ / self.n_ if self.n_ > 0 else float(self.var)

    def pdf(self, x: float) -> float:
        """Density at ``x``, with the maximum-likelihood variance ``m2 / n``."""
        v = self._ml_var()
        if v <= 0.0:
            return 0.0
        v_safe = max(v, 1e-15)
        return math.exp(-0.5 * (x - self.mean_) ** 2 / v_safe) / math.sqrt(2.0 * math.pi * v_safe)

    def cdf(self, x: float) -> float:
        """Probability of a value at or below ``x``, with the maximum-likelihood variance."""
        v = self._ml_var()
        if v <= 0.0:
            return 0.0 if x < self.mean_ else 1.0
        return 0.5 * (1.0 + math.erf((x - self.mean_) / math.sqrt(2.0 * v)))

    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        """``n`` draws."""
        return np.random.default_rng(seed).normal(
            self.mean_, math.sqrt(self.var_), size=n
        )


class Multinomial(MLBase):
    """Dirichlet posterior of a categorical sample: the prior counts plus the observed counts.

    Args:
        alpha: prior count per category, or one number for all.
        n_categories: number of categories when ``alpha`` is one number.

    Example:
        >>> d = Multinomial(alpha=1.0, n_categories=3).fit([0, 0, 2])
        >>> d.counts_.tolist(), d.pmf(0)
        ([3.0, 1.0, 2.0], 0.5)
    """

    def __init__(self, alpha=1.0, n_categories: int = 2) -> None:
        self.alpha = alpha
        self.n_categories = n_categories

    def fit(self, X, y=None):  # type: ignore[override]
        a = np.atleast_1d(np.asarray(self.alpha, dtype=float))
        a = np.full(self.n_categories, a[0]) if a.size == 1 else a
        if np.any(a <= 0):
            raise ValueError("every prior count must be strictly positive")
        x = np.asarray(X, dtype=int).ravel()
        self.counts_ = a + np.bincount(x, minlength=a.size)[: a.size]
        return self

    def pmf(self, k: int) -> float:
        """Posterior mean probability of category ``k``."""
        return float(self.counts_[k] / self.counts_.sum())

    def mean(self) -> np.ndarray:
        """Posterior mean of the category probabilities."""
        return self.counts_ / self.counts_.sum()

    def mode(self) -> int:
        """The category with the largest count."""
        return int(np.argmax(self.counts_))

    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        """``n`` categories drawn from the posterior mean."""
        return np.random.default_rng(seed).choice(
            self.counts_.size, size=n, p=self.mean()
        )
