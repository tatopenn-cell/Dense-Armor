"""Cross-attention: queries from one stream, keys and values from another.

``MultiHeadCrossAttention`` is the encoder-decoder attention of
Vaswani et al. 2017: the queries come from the previous layer, the
keys and values come from the other stream (page 5, "the encoder-decoder
attention where the queries come from the previous decoder layer and
the memory keys and values from the output of the encoder"). The
multi-head split and the output projection follow page 5; the mask,
when given, is a boolean array where ``True`` allows the position,
passed to :func:`scaled_dot_product_attention` (page 5, the mask set to
``-inf`` before the softmax).

The class also carries a streaming form against a growing context, as
``MultiHeadSelfAttention`` does for the causal case: ``init_cache``
gives two empty arrays, ``append_context`` projects one context vector
with ``W_K`` and ``W_V`` and appends it, and ``step`` returns the
output for one query vector against the whole cache without changing
it.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from dense_armor.utility.neural.attention import (
    scaled_dot_product_attention,
    scaled_dot_product_attention_backward,
)
from dense_armor.utility.neural.init import xavier_normal
from dense_armor.utility.neural.params import Param


class MultiHeadCrossAttention:
    """Multi-head attention with queries and context from two streams.

    Args:
        d_model: dimension of the query stream and of the output.
        n_heads: number of heads; ``d_model`` must be divisible by it.
        key: a ``jax.random`` key.
        d_context: dimension of the context stream. ``d_model`` if
            ``None``.

    Raises:
        ValueError: if ``d_model`` is not divisible by ``n_heads``, or
            if ``d_context < 1``.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        key: Any,
        d_context: int | None = None,
    ) -> None:
        if d_context is None:
            d_context = d_model
        if d_context < 1:
            raise ValueError(f"d_context must be >= 1, got {d_context}")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} is not divisible by n_heads {n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_context = d_context
        self.d_k = d_model // n_heads
        keys = jax.random.split(key, 4)
        self.W_Q = Param(xavier_normal(keys[0], (d_model, d_model)))
        self.W_K = Param(xavier_normal(keys[1], (d_context, d_model)))
        self.W_V = Param(xavier_normal(keys[2], (d_context, d_model)))
        self.W_O = Param(xavier_normal(keys[3], (d_model, d_model)))
        self.weights_: jax.Array | None = None
        self._cache: dict[str, jax.Array] | None = None

    def __repr__(self) -> str:
        return (
            f"MultiHeadCrossAttention({self.d_model}, {self.n_heads}, "
            f"d_context={self.d_context})"
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

    def forward(
        self,
        xq: jax.Array,
        xc: jax.Array,
        training: bool = True,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        """Cross-attention with queries from ``xq`` and context ``xc``.

        Args:
            xq: queries, shape ``(Tq, d_model)`` or ``(B, Tq, d_model)``.
            xc: context, shape ``(Tc, d_context)`` or ``(B, Tc, d_context)``.
            training: unused, kept for the interface.
            mask: optional boolean mask of shape ``(Tq, Tc)``; ``True``
                allows the position.

        Returns:
            The attention output, of the shape of ``xq``.
        """
        del training
        xq = jnp.asarray(xq)
        xc = jnp.asarray(xc)
        squeeze_q = xq.ndim == 2
        squeeze_c = xc.ndim == 2
        if squeeze_q:
            xq = xq[None]
        if squeeze_c:
            xc = xc[None]
        if xq.shape[0] != xc.shape[0]:
            raise ValueError(
                f"batch mismatch: xq has batch {xq.shape[0]}, xc has {xc.shape[0]}"
            )
        B, Tq, _ = xq.shape
        Tc = xc.shape[1]
        Q = self._split_heads(xq @ self.W_Q.value)
        K = self._split_heads(xc @ self.W_K.value)
        V = self._split_heads(xc @ self.W_V.value)
        if mask is not None:
            mask = jnp.asarray(mask)
            if mask.shape != (Tq, Tc):
                raise ValueError(f"mask shape {mask.shape}, expected {(Tq, Tc)}")
        scores = Q @ jnp.swapaxes(K, -1, -2) / jnp.sqrt(self.d_k)
        if mask is not None:
            scores = jnp.where(mask, scores, -jnp.inf)
        w = jax.nn.softmax(scores, axis=-1)
        attn_out = w @ V
        merged = self._merge_heads(attn_out)
        y = merged @ self.W_O.value
        self.weights_ = w
        self._cache = dict(
            xq=xq,
            xc=xc,
            Q=Q,
            K=K,
            V=V,
            merged=merged,
            mask=mask,
            w=w,
            squeeze_q=squeeze_q,
            squeeze_c=squeeze_c,
        )
        out = y[0] if squeeze_q else y
        return out

    def backward(self, g: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Gradients; returns ``(dxq, dxc)`` and adds to the four grads."""
        if self._cache is None:
            raise RuntimeError("call forward() before backward()")
        c = self._cache
        g = jnp.asarray(g)
        if c["squeeze_q"]:
            g = g[None]
        B, Tq, D = g.shape
        merged = c["merged"]
        dW_O = merged.reshape(-1, D).T @ g.reshape(-1, D)
        d_merged = g @ self.W_O.value.T
        d_attn_out = self._split_heads(d_merged)
        dQ, dK, dV = scaled_dot_product_attention_backward(
            c["Q"],
            c["K"],
            c["V"],
            d_attn_out,
            c["mask"],
        )
        dQ_m = self._merge_heads(dQ)
        dK_m = self._merge_heads(dK)
        dV_m = self._merge_heads(dV)
        xq_flat = c["xq"].reshape(-1, D)
        xc_flat = c["xc"].reshape(-1, self.d_context)
        dW_Q = xq_flat.T @ dQ_m.reshape(-1, D)
        dW_K = xc_flat.T @ dK_m.reshape(-1, D)
        dW_V = xc_flat.T @ dV_m.reshape(-1, D)
        dXq = dQ_m @ self.W_Q.value.T
        dXc = dK_m @ self.W_K.value.T + dV_m @ self.W_V.value.T
        self.W_Q.grad = self.W_Q.grad + dW_Q
        self.W_K.grad = self.W_K.grad + dW_K
        self.W_V.grad = self.W_V.grad + dW_V
        self.W_O.grad = self.W_O.grad + dW_O
        dxq = dXq[0] if c["squeeze_q"] else dXq
        dxc = dXc[0] if c["squeeze_c"] else dXc
        return dxq, dxc

    def init_cache(self) -> tuple[jax.Array, jax.Array]:
        """Two empty arrays of shape ``(0, d_model)``."""
        return (
            jnp.zeros((0, self.d_model)),
            jnp.zeros((0, self.d_model)),
        )

    def append_context(
        self,
        cache: tuple[jax.Array, jax.Array],
        xc_t: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Project one context vector with ``W_K`` and ``W_V`` and append it."""
        xc_t = jnp.asarray(xc_t)
        squeeze = xc_t.ndim == 1
        if squeeze:
            xc_t = xc_t[None]
        k_new = xc_t @ self.W_K.value
        v_new = xc_t @ self.W_V.value
        k_cache, v_cache = cache
        k_all = jnp.concatenate([k_cache, k_new], axis=0)
        v_all = jnp.concatenate([v_cache, v_new], axis=0)
        return (k_all, v_all)

    def step(
        self,
        xq_t: jax.Array,
        cache: tuple[jax.Array, jax.Array],
    ) -> jax.Array:
        """Output for one query vector against the whole cache, unchanged."""
        xq_t = jnp.asarray(xq_t)
        squeeze = xq_t.ndim == 1
        if squeeze:
            xq_t = xq_t[None]
        k_all, v_all = cache
        q = xq_t @ self.W_Q.value
        Q = q.reshape(1, 1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = k_all.reshape(1, -1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = v_all.reshape(1, -1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        y = scaled_dot_product_attention(Q, K, V, mask=None)
        merged = self._merge_heads(y)
        out = (merged @ self.W_O.value)[0, 0]
        return out if squeeze else out[None]
