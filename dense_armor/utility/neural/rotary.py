"""Rotary position embedding of queries and keys."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def rotary_angles(
    length: int, d_k: int, offset: int = 0, base: float = 10000.0
) -> tuple[jax.Array, jax.Array]:
    """The cosines and sines of the rotation at each position.

    Eq.(15), page 5 of Su et al. (2021), Section 3.2.2: the pair ``i`` of
    the ``d_k`` dimensions (``i`` from 1 to ``d_k / 2``) is rotated by the
    angle ``m * theta_i`` at position ``m``, with ``theta_i =
    10000^(-2(i-1)/d_k)``.

    Args:
        length: the number of positions.
        d_k: the dimension of one head; must be even.
        offset: the position of the first row. Default ``0``.
        base: the constant ``10000`` of the paper.

    Returns:
        ``(cos, sin)``, each of shape ``(length, d_k // 2)``.

    Raises:
        ValueError: if ``d_k`` is odd.

    Examples:
        >>> from dense_armor.utility.neural.rotary import rotary_angles
        >>> cos, sin = rotary_angles(2, 4)
        >>> [round(float(v), 4) for v in cos[1]]
        [0.5403, 0.9999]
    """
    if d_k % 2 != 0:
        raise ValueError(f"d_k must be even, got {d_k}")
    i = jnp.arange(d_k // 2)
    theta = base ** (-2.0 * i / d_k)
    pos = jnp.arange(offset, offset + length)[:, None]
    angles = pos * theta[None, :]
    return jnp.cos(angles), jnp.sin(angles)


def apply_rotary(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Rotate each adjacent pair of the last axis of ``x``.

    The realisation of Eq.(34), page 7 of Su et al. (2021): the pair
    ``(x_{2i-1}, x_{2i})`` becomes ``(x_{2i-1} cos - x_{2i} sin, x_{2i-1}
    sin + x_{2i} cos)``. The rotation is orthogonal (page 5), so the norm of
    ``x`` does not change.

    Args:
        x: array of shape ``(..., length, d_k)``.
        cos: cosines, shape ``(length, d_k // 2)``.
        sin: sines, shape ``(length, d_k // 2)``.

    Returns:
        An array of the shape of ``x``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.rotary import (
        ...     apply_rotary, rotary_angles)
        >>> cos, sin = rotary_angles(2, 2)
        >>> y = apply_rotary(jnp.array([[1.0, 0.0], [1.0, 0.0]]), cos, sin)
        >>> [round(float(v), 4) for v in y[1]]
        [0.5403, 0.8415]
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return jnp.stack([out_even, out_odd], axis=-1).reshape(x.shape)


def apply_rotary_transpose(g: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """The gradient of :func:`apply_rotary`: the rotation by the opposite angle.

    The rotation matrix is orthogonal (page 5 of Su et al. 2021), so its
    transpose is its inverse.

    Args:
        g: array of shape ``(..., length, d_k)``.
        cos: cosines, shape ``(length, d_k // 2)``.
        sin: sines, shape ``(length, d_k // 2)``.

    Returns:
        An array of the shape of ``g``.
    """
    return apply_rotary(g, cos, -sin)
