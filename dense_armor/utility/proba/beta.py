"""Beta distribution over a Bernoulli parameter, updated online.

The Beta is the conjugate prior of the Bernoulli likelihood: an
observation of 1 adds 1 to ``a``, an observation of 0 adds 1 to
``b``. ``update`` and ``revert`` are exactly the two directions of
this update, so the class can be used online and can forget an old
observation. Bishop, Pattern Recognition and Machine Learning,
Section 2.1, equations 2.13 (density), 2.15 (mean), 2.16
(variance). The mode is a standard consequence of the density and
is not numbered in Bishop.
"""

from typing import Any

import numpy as np


class Beta:
    """Beta distribution with online update and revert.

    Args:
        a: initial first hyperparameter, strictly positive.
        b: initial second hyperparameter, strictly positive.

    Raises:
        ValueError: on a non-positive ``a`` or ``b``.

    Examples:
        >>> from dense_armor.utility.proba import Beta
        >>> d = Beta(a=1.0, b=1.0)
        >>> for _ in range(3):
        ...     _ = d.update(1)
        >>> for _ in range(1):
        ...     _ = d.update(0)
        >>> round(d.mean(), 4)
        0.6667

    References:
        Bishop, C. M. (2006). Pattern Recognition and Machine
        Learning. Springer. Section 2.1, eq. 2.13, 2.15, 2.16.
    """

    def __init__(self, a: float = 1.0, b: float = 1.0) -> None:
        if a <= 0.0:
            raise ValueError(f"a must be > 0, got {a}")
        if b <= 0.0:
            raise ValueError(f"b must be > 0, got {b}")
        self.a = float(a)
        self.b = float(b)

    def update(self, x: Any, w: float = 1.0) -> "Beta":
        """Add one observation. ``x`` truthy adds to ``a``, else to ``b``."""
        if w <= 0.0:
            raise ValueError(f"w must be > 0, got {w}")
        if bool(x):
            self.a += float(w)
        else:
            self.b += float(w)
        return self

    def revert(self, x: Any, w: float = 1.0) -> "Beta":
        """Remove one observation. Inverse of :meth:`update`."""
        if w <= 0.0:
            raise ValueError(f"w must be > 0, got {w}")
        if bool(x):
            if self.a <= float(w):
                raise ValueError("revert would make a non-positive")
            self.a -= float(w)
        else:
            if self.b <= float(w):
                raise ValueError("revert would make b non-positive")
            self.b -= float(w)
        return self

    def pdf(self, x: float) -> float:
        """Density at ``x`` in (0, 1). 0.0 outside the open interval."""
        from math import lgamma

        if x <= 0.0 or x >= 1.0:
            return 0.0
        log_beta = lgamma(self.a) + lgamma(self.b) - lgamma(self.a + self.b)
        log_p = (self.a - 1.0) * np.log(x) + (self.b - 1.0) * np.log(1.0 - x)
        return float(np.exp(log_p - log_beta))

    def mean(self) -> float:
        """Mean of the Beta, Bishop eq. 2.15."""
        return self.a / (self.a + self.b)

    def variance(self) -> float:
        """Variance of the Beta, Bishop eq. 2.16."""
        s = self.a + self.b
        return self.a * self.b / (s * s * (s + 1.0))

    def mode(self) -> float:
        """Mode of the Beta. Defined when both hyperparameters are > 1."""
        if self.a <= 1.0 or self.b <= 1.0:
            raise ValueError("mode is not unique when a <= 1 or b <= 1")
        return (self.a - 1.0) / (self.a + self.b - 2.0)

    def sample(self, rng: np.random.Generator) -> float:
        """Draw one sample from the Beta with the current parameters."""
        return float(rng.beta(self.a, self.b))

    def init_state(self) -> tuple:
        """Return the current state (a, b) as JAX scalars."""
        import jax.numpy as jnp

        return (jnp.asarray(self.a), jnp.asarray(self.b))

    def step(self, state: tuple, v: object) -> tuple:
        """One pure update: (a, b) + observation -> ((a, b), mean).

        ``v`` is treated as 1 when ``v >= 0.5`` and as 0 otherwise,
        matching the :meth:`update` convention. Pure function of its
        arguments, safe under ``jax.jit`` and ``jax.lax.scan``.
        """
        import jax.numpy as jnp

        a, b = state
        is_one = jnp.asarray(v) >= 0.5
        a_new = jnp.where(is_one, a + 1.0, a)
        b_new = jnp.where(is_one, b, b + 1.0)
        mean = a_new / (a_new + b_new)
        return (a_new, b_new), mean
