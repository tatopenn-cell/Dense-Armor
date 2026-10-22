"""Gradient clipping by global L2 norm."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from dense_armor.utility.neural.params import Param


def clip_grad_norm(params: list[Param], max_norm: float) -> float:
    """Clip the global L2 norm of the gradients of ``params``.

    The clipping follows Algorithm 1, page 6 of Pascanu, Mikolov and
    Bengio (2013): if the L2 norm of all gradients taken together is at
    least ``max_norm``, every gradient ``g`` becomes
    ``g * max_norm / norm``. The norm is global (across every ``Param``
    of the list), not per parameter.

    Args:
        params: the list of ``Param`` whose ``grad`` is clipped in place.
        max_norm: the threshold, strictly positive.

    Returns:
        The L2 norm of the gradients before the clipping, as a ``float``.
        A norm of 0 leaves the gradients unchanged and returns ``0.0``.

    Raises:
        ValueError: if ``max_norm <= 0``.

    References:
        Pascanu, R., Mikolov, T., Bengio, Y. (2013). On the difficulty
        of training recurrent neural networks. arXiv:1211.5063.
        Section 3.2, Algorithm 1, page 6: ``g <- threshold / ||g|| * g``
        if ``||g|| >= threshold``.
    """
    if max_norm <= 0.0:
        raise ValueError(f"max_norm must be > 0, got {max_norm}")
    total_sq = 0.0
    for p in params:
        g = np.asarray(p.grad)
        total_sq += float((g * g).sum())
    norm = float(np.sqrt(total_sq))
    if norm > 0.0 and norm >= max_norm:
        scale = max_norm / norm
        for p in params:
            p.grad = jnp.asarray(np.asarray(p.grad) * scale)
    return norm
