"""Bottleneck fusion: two streams exchange information through a few tokens.

One layer of Nagrani et al. 2021 (the attention bottleneck): each
stream is concatenated with the shared bottleneck tokens ``fsn`` on the
time axis, a :class:`TransformerBlock` of its own is applied, and the
output is split back into the stream and the stream's bottleneck
(Eq.(8), page 5). The new bottleneck is the average of the two
streams' bottlenecks (Eq.(9), page 5). The cross-stream information
passes only through the bottleneck, because the blocks do not see each
other's stream.

The initial scale 0.02 of the bottleneck tokens is from page 6 of the
paper. ``d_ff = 4 * d_model`` is our choice. Several layers are built
by calling several instances in sequence and passing ``fsn'`` on; this
module adds no stack class.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from dense_armor.utility.neural.attention import TransformerBlock
from dense_armor.utility.neural.params import Param


class BottleneckFusion:
    """One fusion layer of Nagrani et al. 2021, Eq.(7) to Eq.(9).

    Args:
        d_model: the representation dimension of both streams.
        n_heads: the number of heads of each block.
        n_bottleneck: the number of shared tokens; ``B`` in the paper.
        key: a ``jax.random`` key.
        d_ff: width of the feed-forward hidden layer of each block;
            ``4 * d_model`` if ``None``.

    Raises:
        ValueError: if ``n_bottleneck < 1``.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_bottleneck: int,
        key: Any,
        d_ff: int | None = None,
    ) -> None:
        if n_bottleneck < 1:
            raise ValueError(f"n_bottleneck must be >= 1, got {n_bottleneck}")
        if d_ff is None:
            d_ff = 4 * d_model
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_bottleneck = int(n_bottleneck)
        self.d_ff = int(d_ff)
        k_fsn, k_a, k_b = jax.random.split(key, 3)
        fsn0 = 0.02 * jax.random.normal(k_fsn, (n_bottleneck, d_model))
        self.fsn = Param(fsn0)
        self.block_a = TransformerBlock(d_model, n_heads, d_ff, k_a)
        self.block_b = TransformerBlock(d_model, n_heads, d_ff, k_b)
        self._cache: dict[str, Any] | None = None

    def __repr__(self) -> str:
        return (
            f"BottleneckFusion(d_model={self.d_model}, "
            f"n_heads={self.n_heads}, n_bottleneck={self.n_bottleneck}, "
            f"d_ff={self.d_ff})"
        )

    def params(self) -> list[Param]:
        """``fsn``, then the parameters of ``block_a``, then of ``block_b``."""
        return [self.fsn] + self.block_a.params() + self.block_b.params()

    def forward(
        self,
        a: Any,
        b: Any,
        fsn: Any = None,
        training: bool = True,
    ):
        """One fusion layer.

        Args:
            a: first stream, shape ``(Ta, d_model)`` or
                ``(B, Ta, d_model)``.
            b: second stream, shape ``(Tb, d_model)`` or
                ``(B, Tb, d_model)``.
            fsn: bottleneck of the previous layer, shape
                ``(n_bottleneck, d_model)`` or
                ``(B, n_bottleneck, d_model)``; ``self.fsn`` (repeated
                over the batch) when ``None``.
            training: forwarded to the two blocks.

            Declared limitation: a 2D ``fsn`` with a 3D stream (batch
            ``B > 1``) is supported and returns a 2D ``fsn'``, but that
            ``fsn'`` is the *mean* of the two branches' bottlenecks over
            the batch (``fsn_new = mean(fsn_raw, axis=0)``).  The output
            of one sample therefore depends on the other samples of the
            batch.  The 3D form is per-sample and has no such coupling.
            The 2D form is kept for the test ``test_2d_fsn_with_3d_streams``
            and for the callers that use it as a summary; a caller that
            needs per-sample independence passes a 3D ``fsn``.

        Returns:
            The triple ``(a', b', fsn')``.
        """
        a = jnp.asarray(a)
        b = jnp.asarray(b)
        sq_a = a.ndim == 2
        sq_b = b.ndim == 2
        a3 = a[None] if sq_a else a
        b3 = b[None] if sq_b else b
        B, Ta, D = a3.shape
        Tb = b3.shape[1]
        if fsn is None:
            fsn_used = jnp.broadcast_to(self.fsn.value, (B, self.n_bottleneck, D))
            from_self = True
            fsn_out_shape = "self"
        else:
            fsn_arr = jnp.asarray(fsn)
            fsn_used = jnp.broadcast_to(fsn_arr, (B, self.n_bottleneck, D))
            from_self = False
            fsn_out_shape = "2d" if fsn_arr.ndim == 2 else "3d"
        ab = jnp.concatenate([a3, fsn_used], axis=1)
        bb = jnp.concatenate([b3, fsn_used], axis=1)
        a_out = self.block_a.forward(ab, training=training)
        b_out = self.block_b.forward(bb, training=training)
        a_new = a_out[:, :Ta]
        b_new = b_out[:, :Tb]
        fsn_a = a_out[:, Ta:]
        fsn_b = b_out[:, Tb:]
        fsn_raw = 0.5 * (fsn_a + fsn_b)
        if fsn_out_shape == "2d":
            fsn_new = jnp.mean(fsn_raw, axis=0)
        else:
            fsn_new = fsn_raw
        self._cache = dict(
            sq_a=sq_a,
            sq_b=sq_b,
            B=B,
            Ta=Ta,
            Tb=Tb,
            a3=a3,
            b3=b3,
            ab=ab,
            bb=bb,
            fsn_used=fsn_used,
            from_self=from_self,
            fsn_out_shape=fsn_out_shape,
        )
        a_ret = a_new[0] if sq_a else a_new
        b_ret = b_new[0] if sq_b else b_new
        if fsn_out_shape == "2d":
            fsn_ret = fsn_new
        elif sq_a and sq_b:
            fsn_ret = fsn_new[0]
        else:
            fsn_ret = fsn_new
        return a_ret, b_ret, fsn_ret

    def backward(self, ga: Any, gb: Any, gfsn: Any):
        """Gradients; returns ``(da, db, dfsn)`` and adds to every grad.

        If ``fsn`` was ``None`` in the last ``forward``, the gradient of
        the bottleneck goes into ``self.fsn.grad`` and the returned
        ``dfsn`` is ``None``.
        """
        if self._cache is None:
            raise RuntimeError("call forward() before backward()")
        c = self._cache
        B = c["B"]
        Ta = c["Ta"]
        Tb = c["Tb"]
        nb = self.n_bottleneck
        ga = jnp.asarray(ga)
        gb = jnp.asarray(gb)
        if c["sq_a"]:
            ga = ga[None]
        if c["sq_b"]:
            gb = gb[None]
        if gfsn is None:
            gfsn_used = jnp.zeros((B, nb, self.d_model))
        else:
            gfsn_arr = jnp.asarray(gfsn)
            if gfsn_arr.ndim == 2:
                gfsn_used = jnp.broadcast_to(gfsn_arr[None] / B, (B, nb, self.d_model))
            else:
                gfsn_used = jnp.broadcast_to(gfsn_arr, (B, nb, self.d_model))
        d_fsn_half = 0.5 * gfsn_used
        d_ab = jnp.concatenate([ga, d_fsn_half], axis=1)
        d_bb = jnp.concatenate([gb, d_fsn_half], axis=1)
        d_ab_in = self.block_a.backward(d_ab)
        d_bb_in = self.block_b.backward(d_bb)
        da = d_ab_in[:, :Ta]
        db = d_bb_in[:, :Tb]
        d_fsn_a = d_ab_in[:, Ta:]
        d_fsn_b = d_bb_in[:, Tb:]
        dfsn = d_fsn_a + d_fsn_b
        if c["from_self"]:
            self.fsn.grad = self.fsn.grad + jnp.sum(dfsn, axis=0)
            dfsn_ret = None
        elif c["fsn_out_shape"] == "2d":
            dfsn_ret = jnp.sum(dfsn, axis=0)
        else:
            dfsn_ret = dfsn
        da_ret = da[0] if c["sq_a"] else da
        db_ret = db[0] if c["sq_b"] else db
        return da_ret, db_ret, dfsn_ret
