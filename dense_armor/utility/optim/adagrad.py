"""AdaGrad, adaptive per-coordinate step sizes."""

from typing import Any

import jax.numpy as jnp
import numpy as np


class AdaGrad:
    """AdaGrad with per-coordinate adaptive step sizes.

    Update on one sample:

        G_t = G_{t-1} + g_t * g_t
        w_{t+1} = w_t - lr * g_t / (sqrt(G_t) + eps)

    with G_0 = 0. Coordinates whose gradient has been large in the past
    get smaller effective steps.

    Args:
        lr: base learning rate, strictly positive.
        eps: numerical guard added to the square root, strictly positive.

    References:
        Orabona, F. (2019). A Modern Introduction to Online Learning.
        Chapter 4, Algorithm 4.1 (AdaGrad). The source uses a
        per-coordinate step size eta_{t,i} = sqrt(2) D_i /
        (2 sqrt(sum g^2)), where D_i is the diameter along coordinate i.
        This implementation uses a single scalar lr shared by all
        coordinates; Theorem 4.32 still bounds the regret for this
        variant, with a coordinate-wise term instead of a global one.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.optim import AdaGrad
        >>> opt = AdaGrad(lr=0.5)
        >>> w = opt.step(np.zeros(2), np.array([1.0, 1.0]))
        >>> bool(np.allclose(w, [-0.5, -0.5], atol=1e-6))
        True
    """

    def __init__(self, lr: float = 0.1, eps: float = 1e-8) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        self.lr = float(lr)
        self.eps = float(eps)
        self._G: np.ndarray = np.zeros(0)

    def __eq__(self, other: object) -> Any:
        if type(other) is not type(self):
            return NotImplemented
        a = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        b = {k: v for k, v in other.__dict__.items() if not k.startswith("_")}
        return a == b

    def __repr__(self) -> str:
        return f"AdaGrad(lr={self.lr}, eps={self.eps})"

    def step(self, weights: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """One update, keeping the running squared gradient inside."""
        if self._G.size == 0:
            self._G = np.zeros_like(weights)
        self._G = self._G + gradient * gradient
        return weights - self.lr * gradient / (np.sqrt(self._G) + self.eps)

    def init_state(self, dim: int) -> dict[str, Any]:
        """Return the zero accumulator G."""
        return {"G": jnp.zeros(dim)}

    def update(
        self,
        state: dict[str, Any],
        weights: jnp.ndarray,
        gradient: jnp.ndarray,
    ) -> tuple[dict[str, Any], jnp.ndarray]:
        """Pure update, safe under jax.jit and jax.lax.scan."""
        G = state["G"] + gradient * gradient
        w_new = weights - self.lr * gradient / (jnp.sqrt(G) + self.eps)
        return {"G": G}, w_new
