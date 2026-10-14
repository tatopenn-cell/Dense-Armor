"""Dirichlet-style count model over K categories, updated online.

The distribution keeps one positive counter per category and
exposes the Dirichlet posterior over the category probabilities.
An observation of category ``k`` adds 1 to ``alpha[k]``; ``update``
and ``revert`` are the two directions of the same update. Bishop,
Pattern Recognition and Machine Learning, Section 2.2, eq. 2.38
(Dirichlet density) and 2.39 (normalising constant). The mean and
the mode are standard consequences of the density and are not
numbered in Bishop.
"""

from typing import Any

import numpy as np


class Multinomial:
    """Dirichlet count model over K categories.

    Args:
        alpha: initial positive counters, one per category, or a
            scalar replicated to ``n_categories``.
        n_categories: number of categories. Used only when ``alpha``
            is a scalar. Default 2.

    Raises:
        ValueError: on a non-positive counter, or a size mismatch.

    Examples:
        >>> from dense_armor.utility.proba import Multinomial
        >>> d = Multinomial(alpha=1.0, n_categories=3)
        >>> for k in [0, 0, 1, 2]:
        ...     _ = d.update(k)
        >>> [round(float(p), 4) for p in d.mean()]
        [0.4286, 0.2857, 0.2857]

    References:
        Bishop, C. M. (2006). Pattern Recognition and Machine
        Learning. Springer. Section 2.2, eq. 2.38, 2.39.
    """

    def __init__(self, alpha: Any = 1.0, n_categories: int = 2) -> None:
        if isinstance(alpha, (int, float, np.integer, np.floating)):
            a0 = float(alpha)
            if a0 <= 0.0:
                raise ValueError(f"alpha must be > 0, got {alpha}")
            if n_categories < 1:
                raise ValueError(f"n_categories must be >= 1, got {n_categories}")
            self.alpha_ = np.full(int(n_categories), a0)
        else:
            arr = np.asarray(list(alpha), dtype=float)
            if arr.ndim != 1 or arr.size < 1:
                raise ValueError("alpha must be a 1-D non-empty array")
            if np.any(arr <= 0.0):
                raise ValueError("every alpha must be > 0")
            self.alpha_ = arr.copy()

    @property
    def n_categories(self) -> int:
        return int(self.alpha_.size)

    def _check(self, k: Any) -> int:
        try:
            ki = int(k)
        except (TypeError, ValueError):
            raise ValueError(f"category must be an integer, got {k!r}")
        if ki < 0 or ki >= self.n_categories:
            raise ValueError(f"category {ki} out of range [0, {self.n_categories})")
        return ki

    def update(self, k: Any, w: float = 1.0) -> "Multinomial":
        """Add one observation of category ``k``."""
        if w <= 0.0:
            raise ValueError(f"w must be > 0, got {w}")
        ki = self._check(k)
        self.alpha_[ki] += float(w)
        return self

    def revert(self, k: Any, w: float = 1.0) -> "Multinomial":
        """Remove one observation of category ``k``. Inverse of update."""
        if w <= 0.0:
            raise ValueError(f"w must be > 0, got {w}")
        ki = self._check(k)
        if self.alpha_[ki] <= float(w):
            raise ValueError(f"revert would make alpha[{ki}] non-positive")
        self.alpha_[ki] -= float(w)
        return self

    def pmf(self, k: Any) -> float:
        """Posterior mean probability of category ``k``."""
        ki = self._check(k)
        return float(self.alpha_[ki] / self.alpha_.sum())

    def mean(self) -> np.ndarray:
        """Posterior mean vector, ``alpha / sum(alpha)``."""
        return self.alpha_ / self.alpha_.sum()

    def mode(self) -> np.ndarray:
        """Posterior mode vector, ``(alpha - 1) / (sum(alpha) - K)``.

        Defined only when every counter is > 1.
        """
        if np.any(self.alpha_ <= 1.0):
            raise ValueError("mode is not unique when some counter <= 1")
        k = self.n_categories
        return (self.alpha_ - 1.0) / (self.alpha_.sum() - k)

    def sample(self, rng: np.random.Generator) -> int:
        """Draw one category from the posterior predictive."""
        p = self.mean()
        return int(rng.choice(self.n_categories, p=p / p.sum()))
