"""Decoder block of the Transformer, Vaswani et al. 2017, Section 3.1.

The decoder has three sub-layers instead of the encoder's two: masked
self-attention, cross-attention over the encoder output, and the
feed-forward network. Every sub-layer is wrapped in a residual
connection followed by layer normalization, ``LayerNorm(x + Sublayer(x))``
(page 3). The self-attention is masked so that position ``i`` attends
only up to itself (page 3, "prevent positions from attending to
subsequent positions").

References:
    Vaswani et al. 2017, Section 3.1 (page 3, decoder stacks),
    Section 3.2.3 (page 4, multi-head attention), Eq.(2) page 5
    (the feed-forward).
"""

from __future__ import annotations

from typing import Any

import jax

from dense_armor.utility.neural.attention import MultiHeadSelfAttention
from dense_armor.utility.neural.cross_attention import MultiHeadCrossAttention
from dense_armor.utility.neural.layers import Dense
from dense_armor.utility.neural.norm import LayerNorm
from dense_armor.utility.neural.params import Param


class TransformerDecoderBlock:
    """One decoder layer of Vaswani et al. 2017.

    With ``cross=False`` the block has two sub-layers (masked
    self-attention, feed-forward) and is used by a decoder-only language
    model. With ``cross=True`` it has three sub-layers (masked
    self-attention, cross-attention over ``memory``, feed-forward) and is
    used by an encoder-decoder model. Every sub-layer is post-norm:
    ``y = LayerNorm(x + Sublayer(x))``, as in Vaswani et al. (page 3).

    Args:
        d_model: the representation dimension.
        n_heads: the number of heads of both attentions.
        d_ff: the width of the feed-forward hidden layer.
        key: a ``jax.random`` key.
        cross: if True, add the cross-attention sub-layer.
        dropout: dropout probability on the output of each sub-layer,
            applied before the residual addition, default 0.

    Raises:
        ValueError: if ``d_model`` is not divisible by ``n_heads``.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.decoder import (
        ...     TransformerDecoderBlock,
        ... )
        >>> b = TransformerDecoderBlock(8, 2, 16, jax.random.PRNGKey(0))
        >>> b.forward(jnp.ones((4, 8))).shape
        (4, 8)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        key: Any,
        cross: bool = False,
        dropout: float = 0.0,
    ) -> None:
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} is not divisible by n_heads {n_heads}")
        n_keys = 8 if cross else 6
        keys = jax.random.split(key, n_keys)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, keys[0], causal=True)
        self.ln1 = LayerNorm(d_model)
        self.ffn1 = Dense(d_model, d_ff, keys[1])
        self.ffn2 = Dense(d_ff, d_model, keys[2])
        self.cross = cross
        self.dropout_p = dropout
        if cross:
            self.cross_attn = MultiHeadCrossAttention(d_model, n_heads, keys[3])
            self.ln2 = LayerNorm(d_model)
            self.ln3 = LayerNorm(d_model)
        else:
            self.cross_attn = None
            self.ln2 = LayerNorm(d_model)
            self.ln3 = None
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self._cache: dict[str, Any] | None = None

    def __repr__(self) -> str:
        return (
            f"TransformerDecoderBlock({self.d_model}, {self.n_heads}, "
            f"{self.d_ff}, cross={self.cross}, dropout={self.dropout_p})"
        )

    def params(self) -> list[Param]:
        """Every parameter of the block, in a fixed order."""
        out: list[Param] = list(self.attn.params()) + list(self.ln1.params())
        if self.cross:
            out += self.cross_attn.params()
            out += self.ln2.params()
        out += self.ffn1.params() + self.ffn2.params()
        if self.cross:
            out += self.ln3.params()
        else:
            out += self.ln2.params()
        return out

    def forward(
        self,
        x: jax.Array,
        memory: jax.Array | None = None,
        training: bool = True,
    ) -> jax.Array:
        """Run the block over a sequence.

        Args:
            x: the decoder input, shape ``(T, d_model)`` or
                ``(B, T, d_model)``.
            memory: the encoder output, shape ``(Tc, d_model)`` or
                ``(B, Tc, d_model)``. Required when ``cross=True``,
                ignored otherwise.
            training: forwarded to the layers that accept it.

        Raises:
            ValueError: if ``cross=True`` and ``memory`` is ``None``.
        """
        if self.cross and memory is None:
            raise ValueError("cross=True requires a memory array")
        del training
        attn_out = self.attn.forward(x)
        x1 = self.ln1.forward(x + attn_out)
        if self.cross:
            ca_out = self.cross_attn.forward(x1, memory)
            x2 = self.ln2.forward(x1 + ca_out)
            ffn_h = jax.nn.relu(self.ffn1.forward(x2))
            ffn_out = self.ffn2.forward(ffn_h)
            y = self.ln3.forward(x2 + ffn_out)
            self._cache = dict(
                x=x,
                attn_out=attn_out,
                x1=x1,
                ca_out=ca_out,
                x2=x2,
                ffn_h=ffn_h,
                ffn_out=ffn_out,
            )
        else:
            ffn_h = jax.nn.relu(self.ffn1.forward(x1))
            ffn_out = self.ffn2.forward(ffn_h)
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
        """Gradients; returns ``dx`` and adds to every parameter's grad.

        With ``cross=True`` the gradient on ``memory`` is discarded by
        this method (the caller can call ``cross_attn.backward`` on its
        own to obtain it, before this method consumes the cache).
        """
        if self._cache is None:
            raise RuntimeError("call forward() before backward()")
        c = self._cache
        if self.cross:
            d_x2_plus_ffn = self.ln3.backward(g)
            d_ffn_h = self.ffn2.backward(d_x2_plus_ffn)
            d_ffn1_out = d_ffn_h * (c["ffn_h"] > 0).astype(d_ffn_h.dtype)
            d_x2_from_ffn = self.ffn1.backward(d_ffn1_out)
            d_x2 = d_x2_plus_ffn + d_x2_from_ffn
            d_x1_plus_ca = self.ln2.backward(d_x2)
            dq, _d_memory = self.cross_attn.backward(d_x1_plus_ca)
            d_x1 = d_x1_plus_ca + dq
        else:
            d_x1_plus_ffn = self.ln2.backward(g)
            d_ffn_h = self.ffn2.backward(d_x1_plus_ffn)
            d_ffn1_out = d_ffn_h * (c["ffn_h"] > 0).astype(d_ffn_h.dtype)
            d_x1_from_ffn = self.ffn1.backward(d_ffn1_out)
            d_x1 = d_x1_plus_ffn + d_x1_from_ffn
        d_x_plus_attn = self.ln1.backward(d_x1)
        d_x_attn = self.attn.backward(d_x_plus_attn)
        return d_x_plus_attn + d_x_attn
