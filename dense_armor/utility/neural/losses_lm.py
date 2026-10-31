"""Next-token loss for a causal language model.

Mean of the softmax cross-entropy at position ``t`` against the token
at ``t+1``, over the positions whose target is not ``ignore_index``.

References:
    Vaswani et al. 2017, Section 3.1 page 3: the prediction at position
    ``i`` depends only on the outputs at positions less than ``i``, so
    the target of position ``t`` is the token at ``t+1``. The masking of
    padded targets is our choice, declared below.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def _log_softmax(z: jax.Array) -> jax.Array:
    """Log-softmax on the last axis, log-sum-exp stable."""
    m = jnp.max(z, axis=-1, keepdims=True)
    lse = m + jnp.log(jnp.sum(jnp.exp(z - m), axis=-1, keepdims=True))
    return z - lse


class NextTokenLoss:
    """Cross-entropy of the logits at ``t`` against the token at ``t+1``.

    The mean is over the positions whose target is not ``ignore_index``,
    not over ``T - 1``: a fixed denominator would dilute the valid
    targets with the padding and make a short sequence look better than
    it is. The count is therefore the number of valid targets. The
    ``-100`` value is the convention of the field; the masking itself
    is our choice, not stated in Vaswani et al. 2017.

    Args:
        ignore_index: a target with this value does not enter the loss
            and does not enter the count. Default ``-100``.

    Raises:
        ValueError: on ``forward`` if no position has a valid target.

    Examples:
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.losses_lm import NextTokenLoss
        >>> L = NextTokenLoss()
        >>> logits = jnp.array([[[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]]])
        >>> y = jnp.array([[1, 0, 1]])
        >>> round(float(L.forward(logits, y)), 4)
        0.1269
    """

    def __init__(self, ignore_index: int = -100) -> None:
        self.ignore_index = int(ignore_index)
        self._logits: jax.Array | None = None
        self._y: jax.Array | None = None
        self._mask: jax.Array | None = None

    def __repr__(self) -> str:
        return f"NextTokenLoss(ignore_index={self.ignore_index})"

    def forward(self, logits: jax.Array, y: jax.Array) -> jax.Array:
        """Mean cross-entropy of the prediction at ``t`` against ``y[t+1]``.

        Args:
            logits: shape ``(..., T, V)``.
            y: shape ``(..., T)``, integer token ids.

        Returns:
            A scalar loss.

        Raises:
            ValueError: if no position has a valid target.
        """
        logits = jnp.asarray(logits)
        y = jnp.asarray(y)
        if logits.shape[:-2] != y.shape[:-1]:
            raise ValueError(
                f"leading shapes differ: {logits.shape[:-2]} vs {y.shape[:-1]}"
            )
        if logits.shape[-2] != y.shape[-1]:
            raise ValueError(
                f"T differs: logits has {logits.shape[-2]}, y has {y.shape[-1]}"
            )
        target = y[..., 1:]
        mask = target != self.ignore_index
        n_valid = int(jnp.sum(mask))
        if n_valid == 0:
            raise ValueError(
                "no position has a valid target (all ignore_index, or T = 1)"
            )
        lp = _log_softmax(logits[..., :-1, :])
        picked = jnp.take_along_axis(lp, jnp.where(mask, target, 0)[..., None], axis=-1)[..., 0]
        masked = jnp.where(mask, picked, 0.0)
        loss = -jnp.sum(masked) / n_valid
        self._logits = logits
        self._y = y
        self._mask = mask
        return loss

    def backward(self) -> jax.Array:
        """Gradient of the loss with respect to ``logits``."""
        if self._logits is None or self._y is None or self._mask is None:
            raise RuntimeError("call forward() before backward()")
        logits = self._logits
        y = self._y
        mask = self._mask
        V = logits.shape[-1]
        n_valid = int(jnp.sum(mask))
        target = y[..., 1:]
        soft = jax.nn.softmax(logits[..., :-1, :], axis=-1)
        onehot = jax.nn.one_hot(jnp.where(mask, target, 0), V, dtype=soft.dtype)
        d = (soft - onehot) * mask[..., None] / n_valid
        pad = jnp.zeros_like(logits[..., -1:, :])
        return jnp.concatenate([d, pad], axis=-2)

    def __call__(self, logits: jax.Array, y: jax.Array) -> jax.Array:
        return self.forward(logits, y)


def perplexity(loss: float) -> float:
    """``exp(loss)``: the effective vocabulary the model hesitates over."""
    return math.exp(float(loss))
