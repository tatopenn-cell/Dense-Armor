"""Follow-the-Regularized-Leader in its proximal form."""

from typing import Any

import jax.numpy as jnp
import numpy as np


class FTRLProximal:
    """FTRL with a quadratic + L1 + L2 regularizer.

    At each step the weights are the minimizer of the accumulated linear
    losses plus the regularizer:

        w_{t+1} = argmin_w <z_t, w> + (1 / (2 * eta_t)) * ||w||^2
                  + l1 * ||w||_1 + (l2 / 2) * ||w||^2

    where z_t = sum_{s <= t} g_s and eta_t = alpha / t^beta. The closed
    form is the soft threshold

        w_{t+1, i} = -sign(z_{t, i})
                     * max(|z_{t, i}| - l1, 0) / (1 / eta_t + l2)

    The L1 term keeps at zero every coordinate whose accumulator has not
    crossed l1, so the weights are sparse.

    Args:
        alpha: numerator of the learning-rate schedule, strictly positive.
        beta: exponent of the schedule, in [0, 1].
        l1: L1 coefficient, non negative.
        l2: L2 coefficient, non negative.

    References:
        Orabona, F. (2019). A Modern Introduction to Online Learning.
        Chapter 7, Algorithm 7.2 (FTRL with linearized losses). The
        soft-threshold update is the proximal operator of the L1 norm,
        Section 7.8, Example 7.24. The source's Example 7.24 uses a
        time-varying L1 coefficient lambda*t; this implementation uses a
        constant l1, which is the original FTRL-Proximal form
        (McMahan 2011, Section 3.1).

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.optim import FTRLProximal
        >>> opt = FTRLProximal(alpha=1.0, beta=0.5, l1=0.0, l2=0.0)
        >>> w = opt.step(np.zeros(2), np.array([1.0, -1.0]))
        >>> bool(np.allclose(w, [-1.0, 1.0]))
        True
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        l1: float = 0.0,
        l2: float = 0.0,
    ) -> None:
        if alpha <= 0.0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        if l1 < 0.0:
            raise ValueError(f"l1 must be >= 0, got {l1}")
        if l2 < 0.0:
            raise ValueError(f"l2 must be >= 0, got {l2}")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.l1 = float(l1)
        self.l2 = float(l2)
        self._z: np.ndarray = np.zeros(0)
        self._t = 0

    def __eq__(self, other: object) -> Any:
        if type(other) is not type(self):
            return NotImplemented
        a = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        b = {k: v for k, v in other.__dict__.items() if not k.startswith("_")}
        return a == b

    def __repr__(self) -> str:
        return (
            f"FTRLProximal(alpha={self.alpha}, beta={self.beta}, "
            f"l1={self.l1}, l2={self.l2})"
        )

    def step(self, weights: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """One update: accumulate the gradient, then soft threshold."""
        if self._z.size == 0:
            self._z = np.zeros_like(weights)
        self._t += 1
        self._z = self._z + gradient
        eta = self.alpha / (self._t**self.beta)
        return (
            -np.sign(self._z)
            * np.maximum(np.abs(self._z) - self.l1, 0.0)
            / (1.0 / eta + self.l2)
        )

    def init_state(self, dim: int) -> dict[str, Any]:
        """Return a zero accumulator and a zero step counter."""
        return {"z": jnp.zeros(dim), "t": jnp.int32(0)}

    def update(
        self,
        state: dict[str, Any],
        weights: jnp.ndarray,
        gradient: jnp.ndarray,
    ) -> tuple[dict[str, Any], jnp.ndarray]:
        """Pure update, safe under jax.jit and jax.lax.scan.

        The weights argument is unused: the FTRL iterate depends only on
        the accumulator and the step counter.
        """
        t = state["t"] + 1
        z = state["z"] + gradient
        eta = self.alpha / (t.astype(jnp.float32) ** self.beta)
        w_new = (
            -jnp.sign(z)
            * jnp.maximum(jnp.abs(z) - self.l1, 0.0)
            / (1.0 / eta + self.l2)
        )
        return {"z": z, "t": t}, w_new
