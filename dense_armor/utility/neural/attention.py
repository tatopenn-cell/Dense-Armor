"""Attention layers: scaled dot-product, multi-head, block.

The pure function ``scaled_dot_product_attention`` and its gradient
follow Eq.(1), page 4 of Vaswani et al. 2017, with the mask set to
``-inf`` before the softmax (page 5). ``MultiHeadSelfAttention`` follows
the multi-head definition on page 5. ``sinusoidal_positions`` follows
page 6. ``TransformerBlock`` follows the encoder block of the paper:
``LayerNorm(x + Sublayer(x))`` on page 3, with the feed-forward network
of Eq.(2), page 5.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from dense_armor.utility.neural.init import xavier_normal
from dense_armor.utility.neural.layers import Dense, Dropout
from dense_armor.utility.neural.norm import LayerNorm
from dense_armor.utility.neural.params import Param


def scaled_dot_product_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: jax.Array | None = None,
) -> jax.Array:
    """``softmax(q k^T / sqrt(d_k)) v`` (Eq.(1), page 4).

    The mask, if given, is a boolean array broadcastable to the shape of
    the scores; the positions where it is false are set to ``-inf``
    before the softmax (page 5). Works on any leading axis.

    Args:
        q: queries, shape ``(..., T_q, d_k)``.
        k: keys, shape ``(..., T_k, d_k)``.
        v: values, shape ``(..., T_k, d_v)``.
        mask: optional boolean mask.

    Returns:
        The attention output, shape ``(..., T_q, d_v)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.attention import (
        ...     scaled_dot_product_attention,
        ... )
        >>> q = jnp.array([[1.0, 0.0]])
        >>> k = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        >>> v = jnp.array([[1.0], [2.0]])
        >>> scaled_dot_product_attention(q, k, v).shape
        (1, 1)
    """
    q = jnp.asarray(q)
    k = jnp.asarray(k)
    v = jnp.asarray(v)
    d_k = q.shape[-1]
    scores = q @ jnp.swapaxes(k, -1, -2) / jnp.sqrt(d_k)
    if mask is not None:
        scores = jnp.where(mask, scores, -jnp.inf)
    weights = jax.nn.softmax(scores, axis=-1)
    return weights @ v


def scaled_dot_product_attention_backward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Gradients ``(dq, dk, dv)`` of the scaled dot-product attention.

    Args:
        q, k, v: as in the forward function.
        g: the gradient of the output.
        mask: the same mask used in the forward.

    Returns:
        The three gradients, of the shapes of ``q``, ``k``, ``v``.
    """
    q = jnp.asarray(q)
    k = jnp.asarray(k)
    v = jnp.asarray(v)
    g = jnp.asarray(g)
    d_k = q.shape[-1]
    scores = q @ jnp.swapaxes(k, -1, -2) / jnp.sqrt(d_k)
    if mask is not None:
        scores = jnp.where(mask, scores, -jnp.inf)
    weights = jax.nn.softmax(scores, axis=-1)
    d_weights = g @ jnp.swapaxes(v, -1, -2)
    dv = jnp.swapaxes(weights, -1, -2) @ g
    d_scores = weights * (
        d_weights - jnp.sum(d_weights * weights, axis=-1, keepdims=True)
    )
    d_scores = d_scores / jnp.sqrt(d_k)
    dq = d_scores @ k
    dk = jnp.swapaxes(d_scores, -1, -2) @ q
    return dq, dk, dv


def sinusoidal_positions(length: int, d_model: int) -> jax.Array:
    """The sinusoidal positional encoding of page 6.

    ``PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`` and
    ``PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))``.

    Args:
        length: the number of positions.
        d_model: the encoding dimension; must be even.

    Returns:
        An array of shape ``(length, d_model)``.

    Raises:
        ValueError: if ``d_model`` is odd.

    Examples:
        >>> from dense_armor.utility.neural.attention import (
        ...     sinusoidal_positions,
        ... )
        >>> pe = sinusoidal_positions(2, 4)
        >>> [round(float(v), 4) for v in pe[0]]
        [0.0, 1.0, 0.0, 1.0]
        >>> [round(float(v), 4) for v in pe[1]]
        [0.8415, 0.5403, 0.01, 0.9999]
    """
    if d_model % 2 != 0:
        raise ValueError(f"d_model must be even, got {d_model}")
    pos = jnp.arange(length)[:, None]
    i = jnp.arange(d_model // 2)[None, :]
    angle_rates = 1.0 / (10000.0 ** (2.0 * i / d_model))
    angles = pos * angle_rates
    pe = jnp.zeros((length, d_model))
    pe = pe.at[:, 0::2].set(jnp.sin(angles))
    pe = pe.at[:, 1::2].set(jnp.cos(angles))
    return pe


class MultiHeadSelfAttention:
    """Multi-head self-attention of Vaswani et al. 2017, page 5.

    Args:
        d_model: the representation dimension.
        n_heads: the number of heads; ``d_model`` must be divisible by it.
        key: a ``jax.random`` key.
        causal: if True, each position attends only up to itself.

    Raises:
        ValueError: if ``d_model`` is not divisible by ``n_heads``.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.attention import (
        ...     MultiHeadSelfAttention,
        ... )
        >>> m = MultiHeadSelfAttention(8, 2, jax.random.PRNGKey(0))
        >>> m.forward(jnp.ones((4, 8))).shape
        (4, 8)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        key: Any,
        causal: bool = False,
    ) -> None:
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} is not divisible by n_heads {n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.causal = causal
        keys = jax.random.split(key, 4)
        self.W_Q = Param(xavier_normal(keys[0], (d_model, d_model)))
        self.W_K = Param(xavier_normal(keys[1], (d_model, d_model)))
        self.W_V = Param(xavier_normal(keys[2], (d_model, d_model)))
        self.W_O = Param(xavier_normal(keys[3], (d_model, d_model)))
        self._cache: dict[str, jax.Array] | None = None

    def __repr__(self) -> str:
        return (
            f"MultiHeadSelfAttention({self.d_model}, {self.n_heads}, "
            f"causal={self.causal})"
        )

    def params(self) -> list[Param]:
        """The four projection weights ``W_Q``, ``W_K``, ``W_V``, ``W_O``."""
        return [self.W_Q, self.W_K, self.W_V, self.W_O]

    def _split_heads(self, x: jax.Array) -> jax.Array:
        B, T, _ = x.shape
        return x.reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

    def _merge_heads(self, x: jax.Array) -> jax.Array:
        B, H, T, dk = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * dk)

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Run the attention over a sequence."""
        del training
        x = jnp.asarray(x)
        squeeze = x.ndim == 2
        if squeeze:
            x = x[None]
        B, T, D = x.shape
        Q = self._split_heads(x @ self.W_Q.value)
        K = self._split_heads(x @ self.W_K.value)
        V = self._split_heads(x @ self.W_V.value)
        mask = jnp.tril(jnp.ones((T, T), dtype=bool)) if self.causal else None
        attn_out = scaled_dot_product_attention(Q, K, V, mask)
        merged = self._merge_heads(attn_out)
        y = merged @ self.W_O.value
        self._cache = dict(
            x=x,
            Q=Q,
            K=K,
            V=V,
            merged=merged,
            mask=mask,
        )
        return y[0] if squeeze else y

    def backward(self, g: jax.Array) -> jax.Array:
        """Gradients; returns ``dx`` and adds to the four projection grads."""
        if self._cache is None:
            raise RuntimeError("call forward() before backward()")
        g = jnp.asarray(g)
        squeeze = g.ndim == 2
        if squeeze:
            g = g[None]
        B, T, D = g.shape
        merged = self._cache["merged"]
        dW_O = merged.reshape(-1, D).T @ g.reshape(-1, D)
        d_merged = g @ self.W_O.value.T
        d_attn_out = self._split_heads(d_merged)
        dQ, dK, dV = scaled_dot_product_attention_backward(
            self._cache["Q"],
            self._cache["K"],
            self._cache["V"],
            d_attn_out,
            self._cache["mask"],
        )
        dQ_m = self._merge_heads(dQ)
        dK_m = self._merge_heads(dK)
        dV_m = self._merge_heads(dV)
        x_flat = self._cache["x"].reshape(-1, D)
        dW_Q = x_flat.T @ dQ_m.reshape(-1, D)
        dW_K = x_flat.T @ dK_m.reshape(-1, D)
        dW_V = x_flat.T @ dV_m.reshape(-1, D)
        dX = dQ_m @ self.W_Q.value.T + dK_m @ self.W_K.value.T + dV_m @ self.W_V.value.T
        self.W_Q.grad = self.W_Q.grad + dW_Q
        self.W_K.grad = self.W_K.grad + dW_K
        self.W_V.grad = self.W_V.grad + dW_V
        self.W_O.grad = self.W_O.grad + dW_O
        return dX[0] if squeeze else dX

    def init_cache(self) -> tuple[jax.Array, jax.Array]:
        """Two empty arrays of shape ``(0, d_model)``."""
        return (
            jnp.zeros((0, self.d_model)),
            jnp.zeros((0, self.d_model)),
        )

    def step(
        self,
        x_t: jax.Array,
        cache: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        """One causal step against a cache of the past."""
        x_t = jnp.asarray(x_t)
        squeeze = x_t.ndim == 1
        if squeeze:
            x_t = x_t[None]
        k_cache, v_cache = cache
        q = x_t @ self.W_Q.value
        k_new = x_t @ self.W_K.value
        v_new = x_t @ self.W_V.value
        k_all = jnp.concatenate([k_cache, k_new], axis=0)
        v_all = jnp.concatenate([v_cache, v_new], axis=0)
        Q = q.reshape(1, 1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = k_all.reshape(1, -1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = v_all.reshape(1, -1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        y = scaled_dot_product_attention(Q, K, V, mask=None)
        merged = self._merge_heads(y)
        out = (merged @ self.W_O.value)[0, 0]
        return (out if squeeze else out[None]), (k_all, v_all)


class TransformerBlock:
    """The encoder block of Vaswani et al. 2017.

    ``y = LayerNorm(x + Attention(x))``, then ``z = LayerNorm(y +
    FFN(y))`` with ``FFN(x) = max(0, x W_1 + b_1) W_2 + b_2`` (page 3,
    Eq.(2) on page 5). The dropout is applied to the output of each
    sub-layer before the addition (page 8), default 0.

    Args:
        d_model: the representation dimension.
        n_heads: the number of heads.
        d_ff: the width of the feed-forward hidden layer.
        key: a ``jax.random`` key.
        causal: if True, the attention is causal.
        dropout: dropout probability on each sub-layer, default 0.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.attention import TransformerBlock
        >>> b = TransformerBlock(8, 2, 16, jax.random.PRNGKey(0))
        >>> b.forward(jnp.ones((4, 8))).shape
        (4, 8)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        key: Any,
        causal: bool = False,
        dropout: float = 0.0,
    ) -> None:
        keys = jax.random.split(key, 6)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, keys[0], causal=causal)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.ffn1 = Dense(d_model, d_ff, keys[1])
        self.ffn2 = Dense(d_ff, d_model, keys[2])
        self.dropout_p = dropout
        self.causal = causal
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout_attn = Dropout(dropout, keys[3]) if dropout > 0 else None
        self.dropout_ffn = Dropout(dropout, keys[4]) if dropout > 0 else None
        self._cache: dict[str, jax.Array] | None = None

    def __repr__(self) -> str:
        return (
            f"TransformerBlock({self.d_model}, {self.n_heads}, "
            f"{self.d_ff}, causal={self.causal}, dropout={self.dropout_p})"
        )

    def params(self) -> list[Param]:
        """Every parameter of the block, in a fixed order."""
        return (
            self.attn.params()
            + self.ln1.params()
            + self.ln2.params()
            + self.ffn1.params()
            + self.ffn2.params()
        )

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Run the block over a sequence."""
        attn_out = self.attn.forward(x, training=training)
        if self.dropout_attn is not None:
            attn_out = self.dropout_attn.forward(attn_out, training=training)
        x1 = self.ln1.forward(x + attn_out)
        ffn_h = jax.nn.relu(self.ffn1.forward(x1))
        ffn_out = self.ffn2.forward(ffn_h)
        if self.dropout_ffn is not None:
            ffn_out = self.dropout_ffn.forward(ffn_out, training=training)
        y = self.ln2.forward(x1 + ffn_out)
        self._cache = dict(
            x=x,
            attn_out=attn_out,
            x1=x1,
            ffn_h=ffn_h,
            ffn_out=ffn_out,
        )
        return y

    def backward(self, g: jax.Array) -> jax.Array:
        """Gradients; returns ``dx`` and adds to every parameter's grad."""
        if self._cache is None:
            raise RuntimeError("call forward() before backward()")
        c = self._cache
        d_x1_plus_ffn = self.ln2.backward(g)
        d_ffn_out = d_x1_plus_ffn
        if self.dropout_ffn is not None:
            d_ffn_out = self.dropout_ffn.backward(d_ffn_out)
        d_ffn_h = self.ffn2.backward(d_ffn_out)
        d_ffn1_out = d_ffn_h * (c["ffn_h"] > 0).astype(d_ffn_h.dtype)
        d_x1_from_ffn = self.ffn1.backward(d_ffn1_out)
        d_x1 = d_x1_plus_ffn + d_x1_from_ffn
        d_x_plus_attn = self.ln1.backward(d_x1)
        d_attn_out = d_x_plus_attn
        if self.dropout_attn is not None:
            d_attn_out = self.dropout_attn.backward(d_attn_out)
        d_x_attn = self.attn.backward(d_attn_out)
        return d_x_plus_attn + d_x_attn

    def init_cache(self) -> tuple[jax.Array, jax.Array]:
        """The empty cache of the underlying attention."""
        return self.attn.init_cache()

    def step(
        self,
        x_t: jax.Array,
        cache: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        """One causal step; returns ``(y_t, new_cache)``."""
        attn_out, new_cache = self.attn.step(x_t, cache)
        x1 = self.ln1.forward(x_t + attn_out)
        ffn_h = jax.nn.relu(self.ffn1.forward(x1))
        ffn_out = self.ffn2.forward(ffn_h)
        y = self.ln2.forward(x1 + ffn_out)
        return y, new_cache
