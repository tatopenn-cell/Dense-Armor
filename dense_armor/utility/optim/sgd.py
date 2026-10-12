"""Stochastic gradient descent with a fixed learning rate."""

from typing import Any

import jax.numpy as jnp
import numpy as np


class SGD:
    """Fixed-step gradient descent.

    Update on one sample:

        w_{t+1} = w_t - lr * g_t

    with lr > 0 the learning rate and g_t the gradient of the loss on the
    current sample.

    Args:
        lr: learning rate, strictly positive.

    References:
        Orabona, F. (2019). A Modern Introduction to Online Learning,
        Chapter 2, Algorithm 2.1 (Online Gradient Descent) with a
        constant step size.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.optim import SGD
        >>> opt = SGD(lr=0.1)
        >>> w = opt.step(np.zeros(2), np.array([1.0, 1.0]))
        >>> bool(np.allclose(w, [-0.1, -0.1]))
        True
    """

    def __init__(self, lr: float = 0.1) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        self.lr = float(lr)

    def __eq__(self, other: object) -> Any:
        if type(other) is not type(self):
            return NotImplemented
        a = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        b = {k: v for k, v in other.__dict__.items() if not k.startswith("_")}
        return a == b

    def __repr__(self) -> str:
        return f"SGD(lr={self.lr})"

    def step(self, weights: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """One update: w - lr * g."""
        return weights - self.lr * gradient

    def init_state(self, dim: int) -> dict[str, Any]:
        """Return an empty state: SGD keeps no memory."""
        return {}

    def update(
        self,
        state: dict[str, Any],
        weights: jnp.ndarray,
        gradient: jnp.ndarray,
    ) -> tuple[dict[str, Any], jnp.ndarray]:
        """Pure update, safe under jax.jit and jax.lax.scan."""
        return state, weights - self.lr * gradient
