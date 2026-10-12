"""Online Newton Step: second-order online update."""

from typing import Any

import jax.numpy as jnp
import numpy as np


class OnlineNewtonStep:
    """Second-order online update with a rank-one running Hessian.

    Update on one sample:

        A_t = A_{t-1} + g_t g_t^T
        w_{t+1} = w_t - (1 / gamma) * A_t^{-1} g_t

    with A_0 = eps * I. The inverse is kept in the state and updated by
    the Sherman-Morrison formula, so no linear solve is needed. A
    direction that has been repeatedly excited gets a smaller step.

    Args:
        gamma: denominator of the step size, strictly positive.
        eps: initial scale of the Hessian, strictly positive.

    References:
        Hazan, E., Agarwal, A., Kale, S. (2007). Logarithmic regret
        algorithms for online convex optimization. Machine Learning 69,
        Algorithm 1, page 10. The update A_t = A_{t-1} + g_t g_t^T,
        w_{t+1} = w_t - (1/gamma) A_t^{-1} g_t follows this Algorithm.
        Theorem 2, page 11, gives Regret_T <= 5((1/alpha) + G D) n log T
        under alpha-exp-concave losses, bounded gradients ||grad f|| <= G,
        and bounded feasible set of diameter D. This implementation omits
        the projection step of Algorithm 1, leaves gamma and eps as
        free hyperparameters, and does not use the source's formulas
        gamma = (1/2) min(1/(4 G D), alpha), eps = 1/(gamma^2 D^2).
        Orabona, F. (2019), Chapter 7, Section 7.10 states the same
        algorithm in FTRL form (w = -S^{-1} sum g), which is a different
        but equivalent iteration.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.optim import OnlineNewtonStep
        >>> opt = OnlineNewtonStep(gamma=1.0, eps=1.0)
        >>> w = opt.step(np.zeros(2), np.array([1.0, 0.0]))
        >>> bool(np.allclose(w, [-0.5, 0.0]))
        True
    """

    def __init__(self, gamma: float = 1.0, eps: float = 1e-6) -> None:
        if gamma <= 0.0:
            raise ValueError(f"gamma must be > 0, got {gamma}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        self.gamma = float(gamma)
        self.eps = float(eps)
        self._A_inv: np.ndarray = np.zeros(0)

    def __eq__(self, other: object) -> Any:
        if type(other) is not type(self):
            return NotImplemented
        a = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        b = {k: v for k, v in other.__dict__.items() if not k.startswith("_")}
        return a == b

    def __repr__(self) -> str:
        return f"OnlineNewtonStep(gamma={self.gamma}, eps={self.eps})"

    def step(self, weights: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """One update, keeping A^{-1} via Sherman-Morrison."""
        if self._A_inv.size == 0:
            self._A_inv = np.eye(weights.size) / self.eps
        v = self._A_inv @ gradient
        denom = 1.0 + float(gradient @ v)
        self._A_inv = self._A_inv - np.outer(v, v) / denom
        return weights - (self._A_inv @ gradient) / self.gamma

    def init_state(self, dim: int) -> dict[str, Any]:
        """Return A^{-1} = eye(dim) / eps."""
        return {"A_inv": jnp.eye(dim) / self.eps}

    def update(
        self,
        state: dict[str, Any],
        weights: jnp.ndarray,
        gradient: jnp.ndarray,
    ) -> tuple[dict[str, Any], jnp.ndarray]:
        """Pure update, safe under jax.jit and jax.lax.scan."""
        A_inv = state["A_inv"]
        v = A_inv @ gradient
        denom = 1.0 + gradient @ v
        A_inv_new = A_inv - jnp.outer(v, v) / denom
        w_new = weights - (A_inv_new @ gradient) / self.gamma
        return {"A_inv": A_inv_new}, w_new
