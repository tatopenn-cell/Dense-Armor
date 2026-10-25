"""Latent cross-attention: read an input of any length into a fixed set of latents.

``LatentCrossAttention`` is the cross-attention module of Jaegle et al.
2021 (the Perceiver): a learned latent array of size ``N`` much smaller
than the input length ``M`` is the query side, the input is the key and
the value side, and the output has the shape of the latent array
whatever ``M`` is. Page 3: the cross-attention module maps a byte
array and a latent array to a latent array, all attention modules
non-causal. Page 4: K and V are projections of the input byte array, Q
is a projection of the learned latent array of index dimension ``N``
much smaller than ``M``, the cost is ``O(M N)``.

The scale of the latent array (standard deviation 0.02) is our choice:
the paper learns the latent array and gives no scale in the pages used.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from dense_armor.utility.neural.cross_attention import MultiHeadCrossAttention
from dense_armor.utility.neural.params import Param


class LatentCrossAttention:
    """Cross-attention whose queries are a learned latent array.

    Args:
        n_latents: number of latent vectors; much smaller than the
            input length in the intended use.
        d_model: dimension of the latents and of the output.
        n_heads: number of heads; ``d_model`` must be divisible.
        key: a ``jax.random`` key.
        d_context: dimension of the input stream, ``d_model`` if ``None``.

    Raises:
        ValueError: if ``n_latents < 1``; the inner attention raises on
            ``d_model`` not divisible by ``n_heads``.
    """

    def __init__(
        self,
        n_latents: int,
        d_model: int,
        n_heads: int,
        key: Any,
        d_context: int | None = None,
    ) -> None:
        if n_latents < 1:
            raise ValueError(f"n_latents must be >= 1, got {n_latents}")
        if d_context is None:
            d_context = d_model
        self.n_latents = n_latents
        self.d_model = d_model
        self.d_context = d_context
        k_lat, k_att = jax.random.split(key)
        lat0 = 0.02 * jax.random.normal(k_lat, (n_latents, d_model))
        self.latents = Param(lat0)
        self.attention = MultiHeadCrossAttention(
            d_model, n_heads, k_att, d_context=d_context
        )
        self._cache: dict[str, Any] | None = None

    def __repr__(self) -> str:
        return (
            f"LatentCrossAttention(n_latents={self.n_latents}, "
            f"d_model={self.d_model}, d_context={self.d_context})"
        )

    def params(self) -> list[Param]:
        """The latent array followed by the four of the inner attention."""
        return [self.latents] + self.attention.params()

    def forward(self, xc: Any, training: bool = True):
        """Cross-attention of the latents against ``xc``.

        Args:
            xc: input of shape ``(M, d_context)`` or
                ``(B, M, d_context)``, with any ``M``.
            training: unused, forwarded to the inner attention.

        Returns:
            The output, of shape ``(n_latents, d_model)`` or
            ``(B, n_latents, d_model)``, whatever ``M`` is.
        """
        xc = jnp.asarray(xc)
        squeeze = xc.ndim == 2
        if squeeze:
            xc3 = xc[None]
        else:
            xc3 = xc
        B = xc3.shape[0]
        lat = jnp.broadcast_to(self.latents.value, (B, self.n_latents, self.d_model))
        y = self.attention.forward(lat, xc3, training=training)
        self._cache = dict(squeeze=squeeze, xc=xc, B=B)
        return y[0] if squeeze else y

    def backward(self, g: Any):
        """Gradients; returns ``dxc`` and adds to the latents and the inner grads.

        Args:
            g: gradient of the output, of the shape of the output.

        Returns:
            The gradient of the input ``xc``.
        """
        if self._cache is None:
            raise RuntimeError("call forward() before backward()")
        c = self._cache
        g = jnp.asarray(g)
        if c["squeeze"]:
            g = g[None]
        dlat, dxc = self.attention.backward(g)
        dlat_summed = jnp.sum(dlat, axis=0)
        self.latents.grad = self.latents.grad + dlat_summed
        return dxc[0] if c["squeeze"] else dxc
