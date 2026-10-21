"""He and Xavier initialisations for rectifier networks.

References:
    He, K., Zhang, X., Ren, S., Sun, J. (2015). Delving Deep into
        Rectifiers: Surpassing Human-Level Performance on ImageNet
        Classification. arXiv:1502.01852. Eqn.(10), page 4, for
        ``a = 0``; Eqn.(15), page 5, for any ``a``.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp


def fan_in(shape: tuple[int, ...]) -> int:
    """The fan-in of a weight array: ``shape[0]``.

    Args:
        shape: the shape of the weight array.
    """
    if len(shape) == 0:
        raise ValueError("shape must not be empty")
    return int(shape[0])


def he_normal(key: Any, shape: tuple[int, ...], a: float = 0.0) -> jax.Array:
    """Zero-mean Gaussian with standard deviation ``sqrt(2/((1+a**2) fan_in))``.

    Eqn.(15), page 5 of He et al. 2015: ``(1/2)(1+a^2) n_l Var[w_l] = 1``.
    With ``a = 0`` this is the ReLU case (Eqn.(10), page 4); with
    ``a = 1`` it is the linear case, which the paper names "the same
    as [7]" (Glorot and Bengio).

    Args:
        key: a ``jax.random`` key. No default.
        shape: the shape of the returned array.
        a: the negative slope of the PReLU. Default ``0.0`` (ReLU).
    """
    n = fan_in(shape)
    std = math.sqrt(2.0 / ((1.0 + a * a) * n))
    return std * jax.random.normal(key, shape)


def xavier_normal(key: Any, shape: tuple[int, ...]) -> jax.Array:
    """Zero-mean Gaussian with standard deviation ``sqrt(1 / fan_in)``.

    The linear case ``a = 1`` of Eqn.(15), page 5 of He et al. 2015,
    which the paper names "the same as [7]" (Glorot and Bengio).

    Args:
        key: a ``jax.random`` key. No default.
        shape: the shape of the returned array.
    """
    n = fan_in(shape)
    std = math.sqrt(1.0 / n)
    return std * jax.random.normal(key, shape)


def zeros_bias(n: int) -> jax.Array:
    """Zero biases of length ``n``."""
    return jnp.zeros((int(n),))


def glorot_uniform(key: Any, shape: tuple[int, ...]) -> jax.Array:
    """Uniform initialisation of Glorot and Bengio (2010), Eqn.(16) p.5.

    ``W ~ U[-sqrt(6 / (fan_in + fan_out)), sqrt(6 / (fan_in + fan_out))]``,
    with ``fan_in = shape[0]`` and ``fan_out = shape[1]``. For a 1-D
    shape ``fan_out = 1``. The source is open: PMLR 9:249-256, page 5,
    Eqn.(16). The Gaussian case of the same paper is ``xavier_normal``
    above; the two share the scale ``1 / (fan_in + fan_out)`` and differ
    only in the distribution.
    """
    if len(shape) < 1:
        raise ValueError("shape must not be empty")
    n_in = int(shape[0])
    n_out = int(shape[1]) if len(shape) > 1 else 1
    bound = (6.0 / (n_in + n_out)) ** 0.5
    return jax.random.uniform(key, shape, minval=-bound, maxval=bound)
