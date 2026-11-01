"""The modern decoder: RMSNorm, rotary causal attention and a gated feed-forward.

``transformer_decoder`` of ``models.py`` builds the blocks of Vaswani et al.
2017 (LayerNorm, two ``Dense``, sinusoidal positions). The decoder here is
built from ``RMSNorm``, ``MultiHeadSelfAttention(causal=True, rotary=True)``
and ``GLUFeedForward``, with a pre-norm residual in each of the two
sub-layers, and a step with a static key/value cache for ``generate``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from dense_armor.utility.deep.neural.attention import MultiHeadSelfAttention
from dense_armor.utility.deep.neural.glu import GLUFeedForward
from dense_armor.utility.deep.neural.layers import Dense, Embedding
from dense_armor.utility.deep.neural.norm_rms import RMSNorm
from dense_armor.utility.deep.neural.params import Param


class ModernDecoderBlock:
    """Pre-norm decoder block: ``x + Attn(RMSNorm(x))``, then ``+ FFN(RMSNorm(.))``.

    Args:
        d_model: the representation width.
        n_heads: the number of heads; ``d_model`` must be divisible by it
            and the head width must be even (rotary).
        d_ff: the width of the gated feed-forward.
        key: a ``jax.random`` key.
        remat: if True, ``forward`` keeps only its input and ``backward``
            recomputes the rest (``jax.checkpoint``). The gradients are the
            same as without it.

    Raises:
        ValueError: if ``d_model`` is not divisible by ``n_heads`` or the
            head width is odd.

    References:
        Zhang and Sennrich 2019, "Root mean square layer normalization",
        Eq.(4) page 3 (the norm; ``RMSNorm`` adds an ``eps`` inside the
        root, which the paper does not have); Su et al. 2021, "RoFormer",
        Eq.(15) page 5 (the rotary matrix); Shazeer 2020, "GLU variants
        improve Transformer", Eq.(6) page 2 (SwiGLU, no biases).

    Variants and choices:
        The norm comes before each sub-layer (pre-norm); Vaswani et al.
        2017 page 3 put it after the residual sum. This is our choice, not
        in the three papers above. The static key/value cache of
        ``step_static`` is also our choice: the papers do not describe one.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.deep.neural.modern_decoder import (
        ...     ModernDecoderBlock,
        ... )
        >>> b = ModernDecoderBlock(8, 2, 16, jax.random.PRNGKey(0))
        >>> b.forward(jnp.ones((4, 8))).shape
        (4, 8)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        key: Any,
        remat: bool = False,
    ) -> None:
        k1, k2 = jax.random.split(key)
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(
            d_model, n_heads, k1, causal=True, rotary=True
        )
        self.norm2 = RMSNorm(d_model)
        self.ffn = GLUFeedForward(d_model, d_ff, "silu", k2)
        self.d_model = d_model
        self.remat = remat
        self._vjp: Any = None

    def __repr__(self) -> str:
        return (
            f"ModernDecoderBlock({self.d_model}, {self.attn.n_heads}, "
            f"d_ff={self.ffn.d_ff}, remat={self.remat})"
        )

    def params(self) -> list[Param]:
        """The parameters of the two norms, the attention and the feed-forward."""
        return (
            self.norm1.params()
            + self.attn.params()
            + self.norm2.params()
            + self.ffn.params()
        )

    def _sublayers(self) -> list[Any]:
        return [self.norm1, self.attn, self.norm2, self.ffn]

    def _forward_plain(self, x: jax.Array, training: bool) -> jax.Array:
        h = self.attn.forward(self.norm1.forward(x), training=training)
        x = x + h
        h = self.ffn.forward(self.norm2.forward(x), training=training)
        return x + h

    def _forward_remat(self, x: jax.Array, training: bool) -> jax.Array:
        params = self.params()
        subs = self._sublayers()

        def pure(values: list[jax.Array], xx: jax.Array) -> jax.Array:
            saved_values = [p.value for p in params]
            saved_state = [dict(sub.__dict__) for sub in subs]
            for p, v in zip(params, values):
                p.value = v
            try:
                return self._forward_plain(xx, training)
            finally:
                for p, v in zip(params, saved_values):
                    p.value = v
                for sub, state in zip(subs, saved_state):
                    sub.__dict__.clear()
                    sub.__dict__.update(state)

        y, self._vjp = jax.vjp(jax.checkpoint(pure), [p.value for p in params], x)
        return y

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Run the block over a sequence of shape ``(T, d_model)`` or ``(B, T, d_model)``."""
        x = jnp.asarray(x)
        if self.remat:
            return self._forward_remat(x, training)
        return self._forward_plain(x, training)

    def backward(self, g: jax.Array) -> jax.Array:
        """Gradients; returns ``dx`` and adds to the ``grad`` of every parameter."""
        if self.remat:
            if self._vjp is None:
                raise RuntimeError("call forward() before backward()")
            dvalues, dx = self._vjp(g)
            for p, d in zip(self.params(), dvalues):
                p.grad = p.grad + d
            return dx
        g_ffn = self.ffn.backward(g)
        g = g + self.norm2.backward(g_ffn)
        g_attn = self.attn.backward(g)
        return g + self.norm1.backward(g_attn)

    def init_static_cache(self, length: int) -> tuple[jax.Array, jax.Array]:
        """The key/value cache of fixed shape ``(length, d_model)``, zeros."""
        return self.attn.init_static_cache(length)

    def step_static(
        self,
        x_t: jax.Array,
        cache: tuple[jax.Array, jax.Array],
        pos: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        """One token at position ``pos``; returns ``(y_t, new_cache)``."""
        h, cache = self.attn.step_static(self.norm1.forward(x_t), cache, pos)
        x_t = x_t + h
        return x_t + self.ffn.forward(self.norm2.forward(x_t)), cache


class ModernDecoder:
    """``Embedding``, ``n_layers`` modern blocks, ``RMSNorm`` and a ``Dense`` head.

    ``forward(ids)`` takes integer token ids of shape ``(T,)`` or ``(B, T)``
    and returns logits of shape ``(T, vocab)`` or ``(B, T, vocab)``. The
    positions come from the rotary embedding of every attention.

    Use ``modern_decoder`` to build it.
    """

    def __init__(
        self,
        vocab: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        max_len: int,
        d_ff: int,
        key: Any,
        remat: bool = False,
    ) -> None:
        keys = jax.random.split(key, n_layers + 2)
        self.emb = Embedding(vocab, d_model, keys[0])
        self.blocks = [
            ModernDecoderBlock(d_model, n_heads, d_ff, keys[i + 1], remat=remat)
            for i in range(n_layers)
        ]
        self.norm = RMSNorm(d_model)
        self.head = Dense(d_model, vocab, keys[n_layers + 1])
        self.max_len = int(max_len)
        self._jit_step: Any = None

    def __repr__(self) -> str:
        return (
            f"ModernDecoder(vocab={self.emb.E.value.shape[0]}, "
            f"d_model={self.emb.E.value.shape[1]}, n_layers={len(self.blocks)}, "
            f"max_len={self.max_len})"
        )

    def params(self) -> list[Param]:
        """Every parameter: embedding, blocks, final norm, head."""
        p = self.emb.params() + self.norm.params() + self.head.params()
        for b in self.blocks:
            p = p + b.params()
        return p

    def forward(self, ids: jax.Array, training: bool = True) -> jax.Array:
        """Logits for token ids of shape ``(T,)`` or ``(B, T)``."""
        h = self.emb.forward(ids, training=training)
        for b in self.blocks:
            h = b.forward(h, training=training)
        return self.head.forward(self.norm.forward(h))

    def backward(self, g: jax.Array) -> None:
        """Adds to the ``grad`` of every parameter; the ids have no gradient."""
        g = self.norm.backward(self.head.backward(g))
        for b in reversed(self.blocks):
            g = b.backward(g)
        self.emb.backward(g)

    def init_static_cache(self, length: int) -> list[tuple[jax.Array, jax.Array]]:
        """One key/value cache of shape ``(length, d_model)`` per block."""
        return [b.init_static_cache(length) for b in self.blocks]

    def step_static(
        self,
        token: jax.Array,
        caches: list[tuple[jax.Array, jax.Array]],
        pos: jax.Array,
    ) -> tuple[jax.Array, list[tuple[jax.Array, jax.Array]]]:
        """Logits of shape ``(vocab,)`` for one token at position ``pos``.

        ``caches`` comes from ``init_static_cache``. The tokens of the
        prompt are fed one by one at positions ``0, 1, ...``; the logits
        at position ``t`` equal ``forward(ids)[t]``.
        """
        h = self.emb.E.value[token]
        new = []
        for b, c in zip(self.blocks, caches):
            h, c = b.step_static(h, c, pos)
            new.append(c)
        return self.head.forward(self.norm.forward(h)), new

    def _pure_step(
        self,
        values: list[jax.Array],
        token: jax.Array,
        caches: list[tuple[jax.Array, jax.Array]],
        pos: jax.Array,
    ) -> tuple[jax.Array, list[tuple[jax.Array, jax.Array]]]:
        """``step_static`` as a function of the parameter values, for ``jax.jit``."""
        params = self.params()
        subs = [self.norm, self.head]
        for b in self.blocks:
            subs += b._sublayers()
        saved_values = [p.value for p in params]
        saved_state = [dict(s.__dict__) for s in subs]
        for p, v in zip(params, values):
            p.value = v
        try:
            return self.step_static(token, caches, pos)
        finally:
            for p, v in zip(params, saved_values):
                p.value = v
            for s, state in zip(subs, saved_state):
                s.__dict__.clear()
                s.__dict__.update(state)

    def step_compiled(
        self,
        token: jax.Array,
        caches: list[tuple[jax.Array, jax.Array]],
        pos: int,
    ) -> tuple[jax.Array, list[tuple[jax.Array, jax.Array]]]:
        """``step_static`` compiled once; the parameter values are read at each call."""
        if self._jit_step is None:
            self._jit_step = jax.jit(self._pure_step)
        return self._jit_step(
            [p.value for p in self.params()],
            jnp.asarray(token, dtype=jnp.int32),
            caches,
            jnp.asarray(pos, dtype=jnp.int32),
        )


def modern_decoder(
    vocab: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    max_len: int,
    *,
    key: Any,
    d_ff: int | None = None,
    remat: bool = False,
) -> ModernDecoder:
    """A causal language model with RMSNorm, SwiGLU and rotary attention.

    Args:
        vocab: the vocabulary size.
        d_model: the representation width.
        n_heads: the number of heads of every block.
        n_layers: the number of decoder blocks, at least one.
        max_len: the maximum sequence length (``generate`` reads it).
        key: a ``jax.random`` key, keyword only.
        d_ff: the width of the feed-forward. ``None`` uses ``4 * d_model``.
        remat: if True, every block recomputes its activations in
            ``backward`` instead of keeping them.

    Raises:
        ValueError: if ``n_layers`` is below 1, if ``d_model`` is not
            divisible by ``n_heads``, or if the head width is odd.

    Examples:
        >>> import jax
        >>> from dense_armor.utility.deep.neural.modern_decoder import modern_decoder
        >>> m = modern_decoder(8, 8, 2, 1, 4, key=jax.random.PRNGKey(0))
        >>> m.forward(jax.numpy.array([0, 1, 2, 3])).shape
        (4, 8)
    """
    if n_layers < 1:
        raise ValueError(f"n_layers must be at least 1, got {n_layers}")
    if d_model % n_heads != 0:
        raise ValueError(f"d_model {d_model} is not divisible by n_heads {n_heads}")
    return ModernDecoder(
        vocab,
        d_model,
        n_heads,
        n_layers,
        max_len,
        4 * d_model if d_ff is None else d_ff,
        key,
        remat=remat,
    )
