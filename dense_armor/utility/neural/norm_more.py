"""GroupNorm and InstanceNorm.

The channels are split into ``num_groups`` consecutive groups. For each
sample and each group the mean and the variance are computed over the
spatial positions and the channels of the group, ``x_hat = (x - mu) /
sqrt(var + eps)``, then ``y = gamma * x_hat + beta``, with ``gamma`` of
ones and ``beta`` of zeros as two ``Param`` of shape ``(C,)``. There are
no running statistics: training and evaluation give the same output.

References:
    Wu, Y., He, K. (2018). Group Normalization. arXiv:1803.08494.
        Page 3, Eq.(2): ``mu_i = (1/m) sum_{k in S_i} x_k`` and
        ``sigma_i = sqrt((1/m) sum_{k in S_i} (x_k - mu_i)**2 + eps)``.
        Page 4, Eq.(7): ``S_i = {k | k_N = i_N, floor(k_C / (C/G)) =
        floor(i_C / (C/G))}``, with ``G = 32`` by default in the paper.

    Ulyanov, D., Vedaldi, A., Lempitsky, V. (2016). Instance
        Normalization: The Missing Ingredient for Fast Stylization.
        arXiv:1607.08022. Page 4: batch normalisation replaced by a
        normalisation of each image and each channel.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from dense_armor.utility.neural.params import Param


def _as_batched(x: Any, spatial_ndim: int) -> tuple[jax.Array, bool]:
    x = jnp.asarray(x)
    if x.ndim == spatial_ndim + 1:
        return x[None], True
    if x.ndim == spatial_ndim + 2:
        return x, False
    raise ValueError(
        f"expected {spatial_ndim + 1} or {spatial_ndim + 2} dimensions, got {x.ndim}"
    )


class GroupNorm:
    """Group normalization of the channels of a sample.

    Input ``(*spatial, C)`` or ``(B, *spatial, C)``, where ``spatial`` has
    ``spatial_ndim`` axes; output the same shape. There are no running
    statistics: training and evaluation give the same output.

    Args:
        num_groups: number of groups ``G``. ``num_channels`` must be a
            multiple of it.
        num_channels: number of channels ``C`` of the last axis.
        eps: constant added to the variance, default ``1e-5``.
        spatial_ndim: number of spatial axes (1 or 2), default 2.

    Raises:
        ValueError: on a non-positive argument, on ``num_channels`` not
            divisible by ``num_groups``, or on an input of the wrong
            number of dimensions.

    Examples:
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.norm_more import GroupNorm
        >>> m = GroupNorm(2, 4)
        >>> x = jnp.array([[[1.0, 2.0, 3.0, 4.0]], [[5.0, 6.0, 7.0, 8.0]]])
        >>> y = m.forward(x)
        >>> y.shape
        (2, 1, 4)
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        spatial_ndim: int = 2,
    ) -> None:
        if num_groups <= 0:
            raise ValueError(f"num_groups must be > 0, got {num_groups}")
        if num_channels <= 0:
            raise ValueError(f"num_channels must be > 0, got {num_channels}")
        if num_channels % num_groups != 0:
            raise ValueError(
                f"num_channels {num_channels} not divisible by num_groups {num_groups}"
            )
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if spatial_ndim not in (1, 2):
            raise ValueError(f"spatial_ndim must be 1 or 2, got {spatial_ndim}")
        self.num_groups = int(num_groups)
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.spatial_ndim = int(spatial_ndim)
        self.gamma = Param(jnp.ones(num_channels))
        self.beta = Param(jnp.zeros(num_channels))
        self._cache: dict[str, Any] | None = None

    def params(self) -> list[Param]:
        """The gain and the bias, in this order."""
        return [self.gamma, self.beta]

    def __repr__(self) -> str:
        return (
            f"GroupNorm({self.num_groups}, {self.num_channels}, "
            f"eps={self.eps}, spatial_ndim={self.spatial_ndim})"
        )

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Normalise each sample and each group, then apply gain and bias."""
        del training
        x_b, squeeze = _as_batched(x, self.spatial_ndim)
        C = x_b.shape[-1]
        if C != self.num_channels:
            raise ValueError(
                f"input has {C} channels, the layer expects {self.num_channels}"
            )
        B = x_b.shape[0]
        spatial_shape = x_b.shape[1:-1]
        S = 1
        for d in spatial_shape:
            S *= int(d)
        G = self.num_groups
        Cg = C // G
        xr = x_b.reshape(B, S, G, Cg)
        mu = xr.mean(axis=(1, 3), keepdims=True)
        var = ((xr - mu) ** 2).mean(axis=(1, 3), keepdims=True)
        std = jnp.sqrt(var + self.eps)
        xhat = (xr - mu) / std
        gamma_r = self.gamma.value.reshape(G, Cg)
        beta_r = self.beta.value.reshape(G, Cg)
        y = gamma_r * xhat + beta_r
        y = y.reshape(B, S, C).reshape(x_b.shape)
        self._cache = dict(
            xr=xr,
            xhat=xhat,
            std=std,
            mu=mu,
            S=S,
            G=G,
            Cg=Cg,
            squeeze=squeeze,
            in_shape=x_b.shape,
        )
        return y[0] if squeeze else y

    def backward(self, g: jax.Array) -> jax.Array:
        """Return ``dx`` and add to ``gamma.grad`` and ``beta.grad``."""
        if self._cache is None:
            raise RuntimeError("call forward() before backward()")
        c = self._cache
        g = jnp.asarray(g)
        if c["squeeze"]:
            g = g[None]
        G = c["G"]
        Cg = c["Cg"]
        S = c["S"]
        m = S * Cg
        B = c["in_shape"][0]
        g_r = g.reshape(B, S, G, Cg)
        gamma_r = self.gamma.value.reshape(G, Cg)
        dxhat = g_r * gamma_r
        xhat = c["xhat"]
        std = c["std"]
        sum_dxhat = dxhat.sum(axis=(1, 3), keepdims=True)
        sum_dxhat_xhat = (dxhat * xhat).sum(axis=(1, 3), keepdims=True)
        dx = (1.0 / (m * std)) * (m * dxhat - sum_dxhat - xhat * sum_dxhat_xhat)
        dgamma = (g_r * xhat).sum(axis=(0, 1)).reshape(self.num_channels)
        dbeta = g_r.sum(axis=(0, 1)).reshape(self.num_channels)
        self.gamma.grad = self.gamma.grad + dgamma
        self.beta.grad = self.beta.grad + dbeta
        dx = dx.reshape(c["in_shape"])
        return dx[0] if c["squeeze"] else dx


class InstanceNorm:
    """Instance normalization: one group per channel.

    Equivalent to :class:`GroupNorm` with ``num_groups = num_channels``
    (Ulyanov, Vedaldi and Lempitsky 2016, page 4).

    Args:
        num_channels: number of channels of the last axis.
        eps: constant added to the variance, default ``1e-5``.
        spatial_ndim: number of spatial axes (1 or 2), default 2.

    Raises:
        ValueError: on a non-positive argument, or on an input of the
            wrong number of dimensions.
    """

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        spatial_ndim: int = 2,
    ) -> None:
        self._gn = GroupNorm(num_channels, num_channels, eps, spatial_ndim)
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.spatial_ndim = int(spatial_ndim)

    @property
    def gamma(self) -> Param:
        return self._gn.gamma

    @property
    def beta(self) -> Param:
        return self._gn.beta

    def params(self) -> list[Param]:
        """The gain and the bias, in this order."""
        return self._gn.params()

    def __repr__(self) -> str:
        return (
            f"InstanceNorm({self.num_channels}, eps={self.eps}, "
            f"spatial_ndim={self.spatial_ndim})"
        )

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Normalise each sample and each channel, then apply gain and bias."""
        return self._gn.forward(x, training=training)

    def backward(self, g: jax.Array) -> jax.Array:
        """Return ``dx`` and add to ``gamma.grad`` and ``beta.grad``."""
        return self._gn.backward(g)
