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
        Orabona, F. (2019). A Modern Introduction to Online Learning.
        Chapter 2. Algorithm 2.1 is "Projected Online Gradient Descent
        (OGD)", page 24: the update is
        x_{t+1} = Pi_V(x_t - eta_t g_t), with Pi_V the projection on a
        non-empty closed convex set V of diameter D.

        Theorem 2.13, page 25, gives the regret bound
            sum_t (l_t(x_t) - l_t(u)) <=
                D^2 / (2 eta_T) + sum_t (eta_t / 2) ||g_t||^2
                - (1 / (2 eta_T)) ||x_{T+1} - u||^2,
        with a non-increasing step size (0 < eta_{t+1} <= eta_t). When
        the step size is constant the last negative term drops and the
        bound becomes ||u - x_1||^2 / (2 eta) + (eta / 2) sum_t
        ||g_t||^2.

        This implementation does not project: it returns w - lr * g
        directly. The theorem is stated for the projected algorithm, so
        it does not apply here as is. The update rule of the projected
        algorithm without the projection matches the code.

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
