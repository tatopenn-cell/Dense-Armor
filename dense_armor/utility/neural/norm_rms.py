"""Root mean square layer normalisation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dense_armor.utility.neural.params import Param


class RMSNorm:
    """RMS layer normalisation over the features of each sample.

    For each sample ``RMS(a) = sqrt(mean(a**2))`` and ``y = a / RMS(a) *
    gain`` (Eq.(4), page 3 of Zhang and Sennrich 2019, Section 4): the mean
    is not subtracted, so there is no re-centring and no bias. ``eps`` inside
    the root is our choice: the equation writes no constant; the value is the
    one of ``LayerNorm`` in this library.

    Args:
        n_features: number of features in the last axis.
        eps: constant added inside the root. Default ``1e-5``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.norm_rms import RMSNorm
        >>> n = RMSNorm(2, eps=1e-12)
        >>> [round(float(v), 4) for v in n.forward(jnp.array([3.0, 4.0]))]
        [0.8485, 1.1314]
    """

    def __init__(self, n_features: int, eps: float = 1e-5) -> None:
        self.n_features = int(n_features)
        self.eps = float(eps)
        self.gain = Param(jnp.ones(n_features))
        self.xhat_: jax.Array | None = None
        self.rms_: jax.Array | None = None

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Return the normalised input; ``training`` is ignored."""
        del training
        x = jnp.asarray(x)
        rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + self.eps)
        xhat = x / rms
        self.xhat_ = xhat
        self.rms_ = rms
        return self.gain.value * xhat

    def backward(self, g: jax.Array) -> jax.Array:
        """Return ``dx`` and add to ``gain.grad``."""
        if self.xhat_ is None or self.rms_ is None:
            raise RuntimeError("call forward() before backward()")
        g = jnp.asarray(g)
        xhat = self.xhat_
        sum_axes = tuple(range(g.ndim - 1))
        self.gain.grad = self.gain.grad + jnp.sum(g * xhat, axis=sum_axes)
        dxhat = g * self.gain.value
        return (
            dxhat - xhat * jnp.mean(dxhat * xhat, axis=-1, keepdims=True)
        ) / self.rms_

    def params(self) -> list[Param]:
        """The ``Param`` of the gain."""
        return [self.gain]
