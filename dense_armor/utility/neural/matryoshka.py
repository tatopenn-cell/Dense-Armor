"""The Matryoshka Representation Learning loss of Kusupati et al. 2022.

One embedding is trained so that its first ``m`` coordinates are a
usable code for several sizes ``m``. There is one linear head per size,
``W(m)`` of shape ``(n_classes, m)`` without bias, and the loss is
``sum_m c_m L(W(m) z[..., :m], y)`` with ``L`` the softmax
cross-entropy averaged over the batch (Eq.(1), page 4 of Kusupati et
al. 2022).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from dense_armor.utility.neural.init import xavier_normal
from dense_armor.utility.neural.losses import (
    softmax_cross_entropy_backward,
    softmax_cross_entropy_forward,
)
from dense_armor.utility.neural.params import Param


class MatryoshkaLoss:
    """The multi-size loss of Matryoshka Representation Learning.

    ``dims`` is a strictly increasing list of sizes, each at most ``d``,
    for example ``[8, 16, 32]`` for ``d = 32`` (the usual set halves the
    size until a low bottleneck, page 4 of Kusupati et al. 2022). There
    is one linear head per size, ``W(m)`` of shape ``(n_classes, m)``
    without bias, drawn with ``xavier_normal``.

    ``forward(z, y)`` returns ``sum_m c_m * L(W(m) z[..., :m], y)`` with
    ``L`` the softmax cross-entropy averaged over the batch and
    ``c_m = weights[m]`` (default 1 for every size), Eq.(1), page 4.
    ``backward()`` returns ``dz`` of the shape of ``z`` (the gradients
    of the heads on the first ``m`` coordinates, summed) and adds to
    the ``grad`` of each head. ``logits(z, m)`` returns the logits of
    the head of size ``m``; ``truncate(z, m)`` returns ``z[..., :m]``;
    ``params()`` lists the heads.

    Args:
        dims: the sizes, strictly increasing, each at most ``d``.
        d: the size of the embedding.
        n_classes: the number of classes, at least 2.
        key: a ``jax.random`` key, split once per head.
        weights: optional weights, one per size, all non-negative.

    Raises:
        ValueError: on bad ``dims``, ``n_classes < 2``, a ``weights`` of
            another length, or a negative weight.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.matryoshka import MatryoshkaLoss
        >>> ml = MatryoshkaLoss([1, 2], 2, 2, jax.random.PRNGKey(0))
        >>> len(ml.params())
        2
        >>> ml.logits(jnp.array([1.0, 0.0]), 1).shape
        (2,)
    """

    def __init__(
        self,
        dims: Any,
        d: int,
        n_classes: int,
        key: Any,
        weights: Any = None,
    ) -> None:
        dims = list(dims)
        if not dims:
            raise ValueError("dims must have at least one size")
        for i in range(1, len(dims)):
            if dims[i] <= dims[i - 1]:
                raise ValueError(f"dims must be strictly increasing, got {dims}")
        if dims[-1] > d:
            raise ValueError(f"dims[-1]={dims[-1]} exceeds d={d}")
        if n_classes < 2:
            raise ValueError(f"n_classes must be at least 2, got {n_classes}")
        if weights is None:
            weights = [1.0] * len(dims)
        else:
            weights = list(weights)
            if len(weights) != len(dims):
                raise ValueError(
                    f"weights has {len(weights)} entries, expected {len(dims)}"
                )
            for w in weights:
                if w < 0:
                    raise ValueError(f"weights must be non-negative, got {w}")
        self.dims = dims
        self.d = int(d)
        self.n_classes = int(n_classes)
        self.weights = [float(w) for w in weights]
        keys = jax.random.split(key, len(dims))
        self.heads_: list[Param] = [
            Param(xavier_normal(keys[i], (dims[i], n_classes)).T)
            for i in range(len(dims))
        ]
        self.z_: jax.Array | None = None
        self.y_: jax.Array | None = None
        self._logits_per_m: list[jax.Array] = []

    def __repr__(self) -> str:
        return (
            f"MatryoshkaLoss(dims={self.dims}, d={self.d}, "
            f"n_classes={self.n_classes}, weights={self.weights})"
        )

    def params(self) -> list[Param]:
        """The heads, one per size."""
        return list(self.heads_)

    def truncate(self, z: jax.Array, m: int) -> jax.Array:
        """Return the first ``m`` coordinates of ``z``."""
        return jnp.asarray(z)[..., :m]

    def logits(self, z: jax.Array, m: int) -> jax.Array:
        """Return the logits of the head of size ``m``."""
        i = self.dims.index(m)
        z_m = self.truncate(z, m)
        return z_m @ self.heads_[i].value.T

    def forward(self, z: jax.Array, y: Any) -> jax.Array:
        """Return the weighted sum of the losses of the heads."""
        z = jnp.asarray(z)
        squeeze = z.ndim == 1
        z2 = z[None] if squeeze else z
        y_arr = jnp.asarray(y)
        y2 = y_arr[None] if squeeze else y_arr
        total = jnp.zeros(())
        self._logits_per_m = []
        for i, m in enumerate(self.dims):
            logits = z2[:, :m] @ self.heads_[i].value.T
            self._logits_per_m.append(logits)
            L = softmax_cross_entropy_forward(logits, y2)
            total = total + self.weights[i] * L
        self.z_ = z
        self.y_ = y_arr
        return total

    def backward(self) -> jax.Array:
        """Return ``dz`` and add to the grad of each head."""
        if self.z_ is None or self.y_ is None:
            raise RuntimeError("call forward() before backward()")
        z = self.z_
        squeeze = z.ndim == 1
        z2 = z[None] if squeeze else z
        y_arr = self.y_
        y2 = y_arr[None] if squeeze else y_arr
        dz2 = jnp.zeros_like(z2)
        for i, m in enumerate(self.dims):
            logits = self._logits_per_m[i]
            d_logits = softmax_cross_entropy_backward(logits, y2) * self.weights[i]
            W = self.heads_[i].value
            dW = d_logits.T @ z2[:, :m]
            self.heads_[i].grad = self.heads_[i].grad + dW
            dz2 = dz2.at[:, :m].add(d_logits @ W)
        return dz2[0] if squeeze else dz2
