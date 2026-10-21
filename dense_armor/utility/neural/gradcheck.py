"""Gradient check against central finite differences.

For a scalar loss ``L = sum(g * forward(x))`` with a fixed random ``g``,
compute the central finite differences of ``L`` in each element of
``x`` under ``jax.enable_x64(True)`` and return the largest relative
error against ``backward(g)``. The global precision is not changed
outside the call.

``eps = 1e-6`` is the choice of this code, evaluated under
``jax.enable_x64(True)``: with float64 the subtraction in the finite
difference keeps enough digits at ``eps = 1e-6``, and the truncation
error of the central difference (order ``eps**2``) stays below the
tolerance the tests use (``1e-7`` on a correct pair). A float32 grid
would lose the small perturbation in the subtraction at this step.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def _max_relative_error(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.maximum(np.abs(a) + np.abs(b), 1e-12)
    return float(np.max(np.abs(a - b) / denom))


def gradcheck(
    forward: Callable[[Any], Any],
    backward: Callable[[Any, Any, Any], Any],
    x: Any,
    eps: float = 1e-6,
) -> float:
    """Largest relative error between ``backward`` and finite differences.

    Args:
        forward: the forward function, ``forward(x) -> y``.
        backward: the backward function, ``backward(x, y, g) -> dx``.
        x: the input array.
        eps: the step of the central finite difference. Default ``1e-6``.

    Returns:
        The largest relative error over every element of ``x``.

    Raises:
        ValueError: if the shape of ``backward`` differs from ``x``.
    """
    with jax.enable_x64(True):
        x64 = jnp.asarray(x, dtype=jnp.float64)
        key = jax.random.PRNGKey(0)
        g = jax.random.normal(key, jnp.shape(forward(x64)), dtype=jnp.float64)

        def L(xx):
            return jnp.sum(g * forward(xx))

        dx_an = jnp.asarray(backward(x64, forward(x64), g), dtype=jnp.float64)
        if dx_an.shape != x64.shape:
            raise ValueError(f"backward shape {dx_an.shape}, expected {x64.shape}")

        flat = np.asarray(x64).reshape(-1)
        grad_fd = np.empty_like(flat)
        for i in range(flat.size):
            xp = flat.copy()
            xm = flat.copy()
            xp[i] += eps
            xm[i] -= eps
            xp_j = jnp.asarray(xp.reshape(np.shape(x64)))
            xm_j = jnp.asarray(xm.reshape(np.shape(x64)))
            grad_fd[i] = float((L(xp_j) - L(xm_j)) / (2.0 * eps))
        grad_fd = grad_fd.reshape(np.shape(x64))

        return _max_relative_error(np.asarray(dx_an), grad_fd)
