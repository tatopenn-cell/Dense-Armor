"""Losses with a forward and a backward, on a batch.

Each loss is a pair of pure functions, ``<name>_forward`` and
``<name>_backward``, and a class with the same two methods.

The definitions (mean squared error, the mean over the batch of the
softmax cross-entropy, the binary cross-entropy with logits) are
textbook definitions and carry no citation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def mse_forward(pred: jax.Array, target: jax.Array) -> jax.Array:
    """Mean over all elements of ``(pred - target)**2``."""
    d = pred - target
    return jnp.mean(d * d)


def mse_backward(pred: jax.Array, target: jax.Array) -> jax.Array:
    """``2 * (pred - target) / size``."""
    d = pred - target
    return 2.0 * d / jnp.size(d)


def softmax_cross_entropy_forward(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Mean over the batch of ``-log softmax(logits)[label]``, in log-sum-exp form."""
    m = jnp.max(logits, axis=-1, keepdims=True)
    lse = m + jnp.log(jnp.sum(jnp.exp(logits - m), axis=-1, keepdims=True))
    picked = jnp.take_along_axis(logits - lse, labels[:, None], axis=-1)
    return -jnp.mean(picked)


def softmax_cross_entropy_backward(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """``(softmax(logits) - onehot(labels)) / n``."""
    y = jax.nn.softmax(logits, axis=-1)
    n = logits.shape[0]
    onehot = jax.nn.one_hot(labels, logits.shape[-1], dtype=logits.dtype)
    return (y - onehot) / n


def bce_with_logits_forward(logits: jax.Array, targets: jax.Array) -> jax.Array:
    """Mean over the batch of ``max(z,0) - z*t + log(1 + exp(-|z|))``."""
    z = logits
    t = targets
    per = jnp.maximum(z, 0.0) - z * t + jnp.log1p(jnp.exp(-jnp.abs(z)))
    return jnp.mean(per)


def bce_with_logits_backward(logits: jax.Array, targets: jax.Array) -> jax.Array:
    """``(sigmoid(z) - t) / n``, with ``n`` the number of elements."""
    return (jax.nn.sigmoid(logits) - targets) / jnp.size(logits)


class MSE:
    """Mean squared error with a cached last prediction and target."""

    def __init__(self) -> None:
        self._pred: jax.Array | None = None
        self._target: jax.Array | None = None

    def forward(self, pred: jax.Array, target: jax.Array) -> jax.Array:
        self._pred = pred
        self._target = target
        return mse_forward(pred, target)

    def backward(self) -> jax.Array:
        if self._pred is None or self._target is None:
            raise RuntimeError("call forward() before backward()")
        return mse_backward(self._pred, self._target)


class SoftmaxCrossEntropy:
    """Softmax cross-entropy with a cached last logits and labels."""

    def __init__(self) -> None:
        self._logits: jax.Array | None = None
        self._labels: jax.Array | None = None

    def forward(self, logits: jax.Array, labels: jax.Array) -> jax.Array:
        self._logits = logits
        self._labels = labels
        return softmax_cross_entropy_forward(logits, labels)

    def backward(self) -> jax.Array:
        if self._logits is None or self._labels is None:
            raise RuntimeError("call forward() before backward()")
        return softmax_cross_entropy_backward(self._logits, self._labels)


class BCEWithLogits:
    """Binary cross-entropy with logits, with a cached last logits and targets."""

    def __init__(self) -> None:
        self._logits: jax.Array | None = None
        self._targets: jax.Array | None = None

    def forward(self, logits: jax.Array, targets: jax.Array) -> jax.Array:
        self._logits = logits
        self._targets = targets
        return bce_with_logits_forward(logits, targets)

    def backward(self) -> jax.Array:
        if self._logits is None or self._targets is None:
            raise RuntimeError("call forward() before backward()")
        return bce_with_logits_backward(self._logits, self._targets)


def huber_forward(pred: jax.Array, target: jax.Array, delta: float = 1.0) -> jax.Array:
    """Mean of the Huber loss with parameter ``delta``.

    The formula is the one in Chinot (2019), page 4:
    ``l(u, y) = (1/2)(y - u)**2`` if ``|u - y| <= delta``,
    ``delta*|y - u| - delta**2/2`` otherwise. With ``r = pred - target``,
    the loss is ``(1/2) r**2`` if ``|r| <= delta`` and
    ``delta*(|r| - delta/2)`` otherwise.
    """
    if delta <= 0.0:
        raise ValueError(f"delta must be positive, got {delta}")
    r = pred - target
    small = 0.5 * r * r
    big = delta * (jnp.abs(r) - 0.5 * delta)
    return jnp.mean(jnp.where(jnp.abs(r) <= delta, small, big))


def huber_backward(pred: jax.Array, target: jax.Array, delta: float = 1.0) -> jax.Array:
    """Gradient of the Huber loss with respect to ``pred``.

    ``clip(pred - target, -delta, delta) / n``, with ``n`` the number of
    elements. The clip is the derivative of the loss in the two branches:
    ``r`` inside the quadratic region, ``delta*sign(r)`` outside.
    """
    if delta <= 0.0:
        raise ValueError(f"delta must be positive, got {delta}")
    r = pred - target
    return jnp.clip(r, -delta, delta) / jnp.size(r)


class Huber:
    """Huber loss with a cached last prediction and target."""

    def __init__(self, delta: float = 1.0) -> None:
        if delta <= 0.0:
            raise ValueError(f"delta must be positive, got {delta}")
        self.delta = float(delta)
        self._pred: jax.Array | None = None
        self._target: jax.Array | None = None

    def forward(self, pred: jax.Array, target: jax.Array) -> jax.Array:
        self._pred = pred
        self._target = target
        return huber_forward(pred, target, self.delta)

    def backward(self) -> jax.Array:
        if self._pred is None or self._target is None:
            raise RuntimeError("call forward() before backward()")
        return huber_backward(self._pred, self._target, self.delta)
