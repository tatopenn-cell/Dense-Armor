"""Stochastic gradient descent with an optional momentum term."""

from typing import Any

import jax.numpy as jnp
import numpy as np


class SGD:
    """Gradient descent with optional momentum and Nesterov momentum.

    Without momentum, update on one sample:

        w_{t+1} = w_t - lr * g_t

    with lr > 0 the learning rate and g_t the gradient of the loss on the
    current sample.

    With ``momentum = mu > 0``, the update keeps a velocity ``u`` and
    follows Eq.(4), page 4 of Ruder (2016):

        u_t = mu * u_{t-1} + lr * g_t
        w_{t+1} = w_t - u_t

    With ``nesterov = True`` (which requires ``mu > 0``), the update
    returns ``w_t - lr * g_t - mu * u_t`` with the new ``u_t``, the form
    of Eq.(7), page 3 of Bengio, Boulanger-Lewandowski and Pascanu
    (2012). The gradient the caller passes is taken at the current
    weights (the "stored point" of Bengio et al.); the peeked-ahead
    effect of the Nesterov update of Eq.(1) of that page is recovered by
    the extra ``- mu * u_t`` term.

    Args:
        lr: learning rate, strictly positive.
        momentum: momentum coefficient mu, in [0, 1). ``0.0`` is plain
            gradient descent and every other method keeps the behaviour
            of the previous versions.
        nesterov: if True, uses the Nesterov form of the update; requires
            ``momentum > 0``.

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

        Ruder, S. (2016). An overview of gradient descent optimization
        algorithms. arXiv:1609.04747. Section 4.1, Eq.(4), page 4: the
        momentum ``v_t = gamma v_{t-1} + eta grad J(theta)`` and
        ``theta = theta - v_t``.

        Bengio, Y., Boulanger-Lewandowski, N., Pascanu, R. (2012).
        Advances in optimizing recurrent networks. arXiv:1212.0901.
        Section 3.5, Eq.(6) and Eq.(7), page 3: the simplified Nesterov
        momentum.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.optim import SGD
        >>> opt = SGD(lr=0.1)
        >>> w = opt.step(np.zeros(2), np.array([1.0, 1.0]))
        >>> bool(np.allclose(w, [-0.1, -0.1]))
        True

        >>> opt = SGD(lr=0.1, momentum=0.9)
        >>> w1 = float(opt.step(np.array(1.0), np.array(1.0)))
        >>> w2 = float(opt.step(np.array(w1), np.array(1.0)))
        >>> round(w1, 4), round(w2, 4)
        (0.9, 0.71)
    """

    def __init__(
        self,
        lr: float = 0.1,
        momentum: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if nesterov and momentum == 0.0:
            raise ValueError("nesterov=True requires momentum > 0.0")
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)
        self._u: np.ndarray | None = None

    def __eq__(self, other: object) -> Any:
        if type(other) is not type(self):
            return NotImplemented
        a = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        b = {k: v for k, v in other.__dict__.items() if not k.startswith("_")}
        return a == b

    def __repr__(self) -> str:
        if self.momentum == 0.0:
            return f"SGD(lr={self.lr})"
        return f"SGD(lr={self.lr}, momentum={self.momentum}, nesterov={self.nesterov})"

    def step(self, weights: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """One update: w - lr * g, or the momentum form."""
        if self.momentum == 0.0:
            return weights - self.lr * gradient
        if self._u is None:
            self._u = np.zeros_like(weights)
        self._u = self.momentum * self._u + self.lr * gradient
        if self.nesterov:
            return weights - self.lr * gradient - self.momentum * self._u
        return weights - self._u

    def init_state(self, dim: int) -> dict[str, Any]:
        """Return an empty state, or the zero velocity with momentum."""
        if self.momentum == 0.0:
            return {}
        return {"u": jnp.zeros(dim)}

    def update(
        self,
        state: dict[str, Any],
        weights: jnp.ndarray,
        gradient: jnp.ndarray,
    ) -> tuple[dict[str, Any], jnp.ndarray]:
        """Pure update, safe under jax.jit and jax.lax.scan."""
        if self.momentum == 0.0:
            return state, weights - self.lr * gradient
        u = self.momentum * state["u"] + self.lr * gradient
        if self.nesterov:
            w = weights - self.lr * gradient - self.momentum * u
        else:
            w = weights - u
        return {"u": u}, w
