"""RMSProp, an adaptive learning-rate optimizer."""

from typing import Any

import jax.numpy as jnp
import numpy as np


class RMSProp:
    """RMSProp with an exponentially decaying average of squared gradients.

    Update on one sample, following Eq.(18), page 7 of Ruder (2016):

        e_t = rho * e_{t-1} + (1 - rho) * g_t^2
        w_{t+1} = w_t - lr * g_t / sqrt(e_t + eps)

    with ``e_t`` the running mean of the squared gradients and ``eps`` a
    numerical guard inside the square root, as in the Eq.(18) of the
    paper. The paper calls the decay ``gamma`` and uses ``0.9``; the
    learning rate it suggests is ``0.001``.

    Args:
        lr: step size, strictly positive.
        rho: decay of the squared-gradient average, in [0, 1).
        eps: numerical guard, strictly positive.

    References:
        Ruder, S. (2016). An overview of gradient descent optimization
        algorithms. arXiv:1609.04747. Section 4.5, Eq.(18), page 7:
        ``E[g^2]_t = 0.9 E[g^2]_{t-1} + 0.1 g_t^2`` and
        ``theta_{t+1} = theta_t - eta / sqrt(E[g^2]_t + eps) * g_t``.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.optim import RMSProp
        >>> opt = RMSProp(lr=0.01)
        >>> w = opt.step(np.array(1.0), np.array(1.0))
        >>> round(float(w), 5)
        0.96838
    """

    def __init__(
        self,
        lr: float = 0.001,
        rho: float = 0.9,
        eps: float = 1e-8,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if not 0.0 <= rho < 1.0:
            raise ValueError(f"rho must be in [0, 1), got {rho}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        self.lr = float(lr)
        self.rho = float(rho)
        self.eps = float(eps)
        self._e: np.ndarray | None = None

    def __eq__(self, other: object) -> Any:
        if type(other) is not type(self):
            return NotImplemented
        a = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        b = {k: v for k, v in other.__dict__.items() if not k.startswith("_")}
        return a == b

    def __repr__(self) -> str:
        return f"RMSProp(lr={self.lr}, rho={self.rho}, eps={self.eps})"

    def step(self, weights: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """One update, keeping the running average of squared gradients."""
        if self._e is None:
            self._e = np.zeros_like(weights)
        self._e = self.rho * self._e + (1.0 - self.rho) * gradient * gradient
        return weights - self.lr * gradient / np.sqrt(self._e + self.eps)

    def init_state(self, dim: int) -> dict[str, Any]:
        """Return the zero average of squared gradients."""
        return {"e": jnp.zeros(dim)}

    def update(
        self,
        state: dict[str, Any],
        weights: jnp.ndarray,
        gradient: jnp.ndarray,
    ) -> tuple[dict[str, Any], jnp.ndarray]:
        """Pure update, safe under jax.jit and jax.lax.scan."""
        e = self.rho * state["e"] + (1.0 - self.rho) * gradient * gradient
        w = weights - self.lr * gradient / jnp.sqrt(e + self.eps)
        return {"e": e}, w
