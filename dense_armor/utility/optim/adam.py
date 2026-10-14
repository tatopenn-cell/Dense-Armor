"""Adam, adaptive moment estimation."""

from typing import Any

import jax.numpy as jnp
import numpy as np


class Adam:
    """Adam with bias-corrected moment estimates.

    Update on one sample, following Algorithm 1 of Kingma and Ba (2014):

        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        w_{t+1} = w_t - lr * m_hat / (sqrt(v_hat) + eps)

    The bias correction 1 - beta^t matters for the first few steps, when
    m and v are still pulled toward zero.

    Args:
        lr: step size, strictly positive.
        beta1: decay of the first moment, in [0, 1).
        beta2: decay of the second moment, in [0, 1).
        eps: numerical guard, strictly positive.

    References:
        Kingma, D. P., Ba, J. (2014). Adam: a method for stochastic
        optimization. arXiv:1412.6980. Algorithm 1, page 2.

        Theorem 4.1, page 4, bounds the regret with rate O(sqrt(T))
        under these conditions: bounded gradients (||grad f||_2 <= G
        and ||grad f||_inf <= G_inf at every step), bounded iterates
        (||theta_n - theta_m||_2 <= D and ||theta_m - theta_n||_inf <=
        D_inf for any two steps), beta1^2 / sqrt(beta2) < 1, a learning
        rate decaying as alpha_t = alpha / sqrt(t), and a first moment
        coefficient decaying as beta_{1,t} = beta1 * lambda^(t-1) with
        lambda in (0, 1). The paper uses lambda = 1 - 1e-8 in the
        experiments and states that decaying beta_{1,t} towards zero is
        required by the proof.

        This implementation uses a constant ``lr`` and a constant
        ``beta1``, so the conditions on alpha_t and beta_{1,t} do not
        hold and Theorem 4.1 does not apply to it. The update rule
        (Algorithm 1) is the same.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.optim import Adam
        >>> opt = Adam(lr=0.1)
        >>> w = opt.step(np.zeros(2), np.array([1.0, 1.0]))
        >>> bool(np.allclose(w, [-0.1, -0.1], atol=1e-6))
        True
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self._m: np.ndarray = np.zeros(0)
        self._v: np.ndarray = np.zeros(0)
        self._t = 0

    def __eq__(self, other: object) -> Any:
        if type(other) is not type(self):
            return NotImplemented
        a = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        b = {k: v for k, v in other.__dict__.items() if not k.startswith("_")}
        return a == b

    def __repr__(self) -> str:
        return (
            f"Adam(lr={self.lr}, beta1={self.beta1}, "
            f"beta2={self.beta2}, eps={self.eps})"
        )

    def step(self, weights: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """One update, keeping both moments and the step counter."""
        if self._m.size == 0:
            self._m = np.zeros_like(weights)
            self._v = np.zeros_like(weights)
        self._t += 1
        self._m = self.beta1 * self._m + (1.0 - self.beta1) * gradient
        self._v = self.beta2 * self._v + (1.0 - self.beta2) * gradient * gradient
        m_hat = self._m / (1.0 - self.beta1**self._t)
        v_hat = self._v / (1.0 - self.beta2**self._t)
        return weights - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def init_state(self, dim: int) -> dict[str, Any]:
        """Return zero moments and a zero step counter."""
        return {
            "m": jnp.zeros(dim),
            "v": jnp.zeros(dim),
            "t": jnp.int32(0),
        }

    def update(
        self,
        state: dict[str, Any],
        weights: jnp.ndarray,
        gradient: jnp.ndarray,
    ) -> tuple[dict[str, Any], jnp.ndarray]:
        """Pure update, safe under jax.jit and jax.lax.scan."""
        t = state["t"] + 1
        m = self.beta1 * state["m"] + (1.0 - self.beta1) * gradient
        v = self.beta2 * state["v"] + (1.0 - self.beta2) * gradient * gradient
        m_hat = m / (1.0 - self.beta1**t)
        v_hat = v / (1.0 - self.beta2**t)
        w_new = weights - self.lr * m_hat / (jnp.sqrt(v_hat) + self.eps)
        return {"m": m, "v": v, "t": t}, w_new
