"""More losses: L1, label-smoothed cross-entropy, KL divergence, hinge,
focal.

Each loss is a pair of pure functions, ``<name>_forward`` and
``<name>_backward``, and a class with the same two methods that caches
the last call, in the style of
:mod:`dense_armor.utility.neural.losses`.

References:
    Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., Wojna, Z.
        (2016). Rethinking the Inception Architecture for Computer
        Vision. arXiv:1512.00567. Section 7, page 7: the label-smoothing
        regularization ``q'(k) = (1 - eps) delta_{k,y} + eps / K``.

    Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollar, P. (2017).
        Focal Loss for Dense Object Detection. arXiv:1708.02002. Page 1:
        ``FL(p_t) = -(1 - p_t)**gamma * log(p_t)``. Page 3, Eq.(5): the
        ``alpha``-balanced form.

    The L1 loss, the KL divergence and the hinge loss are textbook
    definitions and carry no citation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def l1_forward(pred: jax.Array, target: jax.Array) -> jax.Array:
    """Mean of ``abs(pred - target)`` over all elements."""
    return jnp.mean(jnp.abs(pred - target))


def l1_backward(pred: jax.Array, target: jax.Array) -> jax.Array:
    """``sign(pred - target) / size`` (0 where the difference is 0)."""
    return jnp.sign(pred - target) / jnp.size(pred)


def label_smoothing_ce_forward(
    logits: jax.Array, labels: jax.Array, eps: float = 0.1
) -> jax.Array:
    """Mean over the batch of the softmax cross-entropy with smoothed labels."""
    K = logits.shape[-1]
    m = jnp.max(logits, axis=-1, keepdims=True)
    lse = m + jnp.log(jnp.sum(jnp.exp(logits - m), axis=-1, keepdims=True))
    logp = logits - lse
    onehot = jax.nn.one_hot(labels, K, dtype=logits.dtype)
    q = (1.0 - eps) * onehot + eps / K
    return -jnp.mean(jnp.sum(q * logp, axis=-1))


def label_smoothing_ce_backward(
    logits: jax.Array, labels: jax.Array, eps: float = 0.1
) -> jax.Array:
    """``(softmax(logits) - q'(k)) / n`` with ``n`` the batch size."""
    K = logits.shape[-1]
    n = logits.shape[0]
    onehot = jax.nn.one_hot(labels, K, dtype=logits.dtype)
    q = (1.0 - eps) * onehot + eps / K
    return (jax.nn.softmax(logits, axis=-1) - q) / n


def kldiv_forward(logits: jax.Array, target_probs: jax.Array) -> jax.Array:
    """Mean over the batch of ``sum_k y_k (log y_k - log softmax(logits)_k)``."""
    m = jnp.max(logits, axis=-1, keepdims=True)
    lse = m + jnp.log(jnp.sum(jnp.exp(logits - m), axis=-1, keepdims=True))
    logp = logits - lse
    safe_y = jnp.where(target_probs > 0.0, target_probs, 1.0)
    log_y = jnp.where(target_probs > 0.0, jnp.log(safe_y), 0.0)
    per = jnp.sum(target_probs * (log_y - logp), axis=-1)
    return jnp.mean(per)


def kldiv_backward(logits: jax.Array, target_probs: jax.Array) -> jax.Array:
    """``(softmax(logits) - y) / n`` with ``n`` the batch size."""
    n = logits.shape[0]
    return (jax.nn.softmax(logits, axis=-1) - target_probs) / n


def hinge_forward(pred: jax.Array, y: jax.Array) -> jax.Array:
    """Mean of ``max(0, 1 - y * pred)`` over all elements."""
    return jnp.mean(jnp.maximum(0.0, 1.0 - y * pred))


def hinge_backward(pred: jax.Array, y: jax.Array) -> jax.Array:
    """``-y / size`` where ``1 - y * pred > 0``, 0 elsewhere."""
    mask = (1.0 - y * pred) > 0.0
    return jnp.where(mask, -y, jnp.zeros_like(pred)) / jnp.size(pred)


def focal_forward(
    logits: jax.Array,
    y: jax.Array,
    gamma: float = 2.0,
    alpha: float | None = None,
) -> jax.Array:
    """Mean of the ``alpha``-balanced focal loss over all elements."""
    p = jax.nn.sigmoid(logits)
    p_t = jnp.where(y == 1.0, p, 1.0 - p)
    log_p_t = jnp.where(
        y == 1.0, jax.nn.log_sigmoid(logits), jax.nn.log_sigmoid(-logits)
    )
    if alpha is not None:
        a_t = jnp.where(y == 1.0, alpha, 1.0 - alpha)
    else:
        a_t = jnp.ones_like(logits)
    per = -a_t * (1.0 - p_t) ** gamma * log_p_t
    return jnp.mean(per)


def focal_backward(
    logits: jax.Array,
    y: jax.Array,
    gamma: float = 2.0,
    alpha: float | None = None,
) -> jax.Array:
    """``d loss / d logits`` of the focal loss, mean over all elements."""
    p = jax.nn.sigmoid(logits)
    p_t = jnp.where(y == 1.0, p, 1.0 - p)
    if alpha is not None:
        a_t = jnp.where(y == 1.0, alpha, 1.0 - alpha)
    else:
        a_t = jnp.ones_like(logits)
    dpt_dz = jnp.where(y == 1.0, p * (1.0 - p), -p * (1.0 - p))
    dlogpt_dz = jnp.where(y == 1.0, 1.0 - p, -p)
    log_pt = jnp.where(
        y == 1.0, jax.nn.log_sigmoid(logits), jax.nn.log_sigmoid(-logits)
    )
    term1 = a_t * gamma * (1.0 - p_t) ** (gamma - 1.0) * dpt_dz * log_pt
    term2 = -a_t * (1.0 - p_t) ** gamma * dlogpt_dz
    return (term1 + term2) / jnp.size(logits)


class L1Loss:
    """Mean absolute error with cached last call."""

    def __init__(self) -> None:
        self._pred: jax.Array | None = None
        self._target: jax.Array | None = None

    def forward(self, pred: jax.Array, target: jax.Array) -> jax.Array:
        self._pred = pred
        self._target = target
        return l1_forward(pred, target)

    def backward(self) -> jax.Array:
        if self._pred is None or self._target is None:
            raise RuntimeError("call forward() before backward()")
        return l1_backward(self._pred, self._target)


class LabelSmoothingCE:
    """Softmax cross-entropy with smoothed labels, cached last call."""

    def __init__(self, eps: float = 0.1) -> None:
        if not 0.0 <= eps < 1.0:
            raise ValueError(f"eps must be in [0, 1), got {eps}")
        self.eps = float(eps)
        self._logits: jax.Array | None = None
        self._labels: jax.Array | None = None

    def forward(self, logits: jax.Array, labels: jax.Array) -> jax.Array:
        self._logits = logits
        self._labels = labels
        return label_smoothing_ce_forward(logits, labels, self.eps)

    def backward(self) -> jax.Array:
        if self._logits is None or self._labels is None:
            raise RuntimeError("call forward() before backward()")
        return label_smoothing_ce_backward(self._logits, self._labels, self.eps)


class KLDivLoss:
    """Kullback-Leibler divergence ``sum_k y_k (log y_k - log p_k)``."""

    def __init__(self) -> None:
        self._logits: jax.Array | None = None
        self._targets: jax.Array | None = None

    def forward(self, logits: jax.Array, target_probs: jax.Array) -> jax.Array:
        self._logits = logits
        self._targets = target_probs
        return kldiv_forward(logits, target_probs)

    def backward(self) -> jax.Array:
        if self._logits is None or self._targets is None:
            raise RuntimeError("call forward() before backward()")
        return kldiv_backward(self._logits, self._targets)


class HingeLoss:
    """Hinge loss ``max(0, 1 - y * pred)`` for ``y`` in {-1, +1}."""

    def __init__(self) -> None:
        self._pred: jax.Array | None = None
        self._y: jax.Array | None = None

    def forward(self, pred: jax.Array, y: jax.Array) -> jax.Array:
        self._pred = pred
        self._y = y
        return hinge_forward(pred, y)

    def backward(self) -> jax.Array:
        if self._pred is None or self._y is None:
            raise RuntimeError("call forward() before backward()")
        return hinge_backward(self._pred, self._y)


class FocalLoss:
    """Focal loss ``-alpha_t (1 - p_t)**gamma * log(p_t)``."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | None = None,
    ) -> None:
        if gamma < 0.0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if alpha is not None and not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.gamma = float(gamma)
        self.alpha = None if alpha is None else float(alpha)
        self._logits: jax.Array | None = None
        self._y: jax.Array | None = None

    def forward(self, logits: jax.Array, y: jax.Array) -> jax.Array:
        self._logits = logits
        self._y = y
        return focal_forward(logits, y, self.gamma, self.alpha)

    def backward(self) -> jax.Array:
        if self._logits is None or self._y is None:
            raise RuntimeError("call forward() before backward()")
        return focal_backward(self._logits, self._y, self.gamma, self.alpha)
