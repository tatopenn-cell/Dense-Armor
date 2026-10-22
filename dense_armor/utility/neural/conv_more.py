"""Three-dimensional, depthwise and separable convolutions.

References:
    Tran, D., Bourdev, L., Fergus, R., Torresani, L., Paluri, M.
        (2015). Learning spatiotemporal features with 3D convolutional
        networks. arXiv:1412.0767. Page 1: a homogeneous architecture
        with small 3 x 3 x 3 convolution kernels in all layers is among
        the best performing architectures for 3D ConvNets.

    Chollet, F. (2017). Xception: Deep learning with depthwise
        separable convolutions. arXiv:1610.02357. Page 1 (abstract):
        the depthwise separable convolution is a depthwise convolution
        followed by a pointwise convolution. Page 2, Section 3: the
        Inception module as an intermediate step between a regular
        convolution and the depthwise separable convolution.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from dense_armor.utility.neural.conv import Conv2D
from dense_armor.utility.neural.init import he_normal, zeros_bias
from dense_armor.utility.neural.params import Param


class Conv3D:
    """3-D cross-correlation with zero padding, channels last.

    Input ``(D, H, W, c_in)`` or ``(B, D, H, W, c_in)``, output
    ``(D_out, H_out, W_out, c_out)`` or ``(B, D_out, H_out, W_out, c_out)``.
    The kernel is ``(kernel, kernel, kernel, c_in, c_out)`` and the bias
    ``(c_out,)``, drawn as in :class:`~dense_armor.utility.neural.conv.Conv2D`.

    Args:
        c_in: input channels.
        c_out: output channels.
        kernel: kernel side (cube).
        key: a ``jax.random`` key.
        stride: stride along the three spatial axes, default 1.
        padding: zero padding on each side of each axis, default 0.

    Raises:
        ValueError: if ``kernel < 1``.

    References:
        Tran et al. (2015). arXiv:1412.0767. Page 1: 3 x 3 x 3 kernels.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.conv_more import Conv3D
        >>> c = Conv3D(1, 1, 3, jax.random.PRNGKey(0))
        >>> c.forward(jnp.ones((4, 4, 4, 1))).shape
        (2, 2, 2, 1)
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel: int,
        key: Any,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        if kernel < 1:
            raise ValueError(f"kernel must be at least 1, got {kernel}")
        self.c_in = c_in
        self.c_out = c_out
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        w = he_normal(key, (kernel * kernel * kernel * c_in, c_out)).reshape(
            kernel, kernel, kernel, c_in, c_out
        )
        self.W = Param(w)
        self.b = Param(zeros_bias(c_out))
        self._cache: dict[str, Any] | None = None

    def params(self) -> list[Param]:
        """The weight and the bias, in this order."""
        return [self.W, self.b]

    def __repr__(self) -> str:
        return (
            f"Conv3D({self.c_in}, {self.c_out}, {self.kernel}, "
            f"stride={self.stride}, padding={self.padding})"
        )

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Return the 3-D cross-correlation of ``x`` with the weight, plus the bias."""
        del training
        x = jnp.asarray(x)
        dt = jnp.result_type(x.dtype, self.W.value.dtype)
        x = x.astype(dt)
        squeeze = x.ndim == 4
        if squeeze:
            x = x[None]
        p = self.padding
        y = jax.lax.conv_general_dilated(
            x,
            self.W.value.astype(dt),
            window_strides=(self.stride, self.stride, self.stride),
            padding=((p, p), (p, p), (p, p)),
            dimension_numbers=("NDHWC", "DHWIO", "NDHWC"),
        ) + self.b.value.astype(dt)
        self._cache = dict(x=x, squeeze=squeeze)
        return y[0] if squeeze else y

    def backward(self, g: jax.Array) -> jax.Array:
        """Return ``dx`` and add to ``W.grad`` and ``b.grad``."""
        if self._cache is None:
            raise RuntimeError("call forward() before backward()")
        c = self._cache
        g = jnp.asarray(g)
        if c["squeeze"]:
            g = g[None]
        x = c["x"]
        p = self.padding
        s = self.stride
        dW, dx = jax.vjp(
            lambda k, xv: jax.lax.conv_general_dilated(
                xv,
                k,
                window_strides=(s, s, s),
                padding=((p, p), (p, p), (p, p)),
                dimension_numbers=("NDHWC", "DHWIO", "NDHWC"),
            ),
            self.W.value.astype(x.dtype),
            x,
        )[1](g)
        db = g.sum(axis=(0, 1, 2, 3))
        self.W.grad = self.W.grad + dW
        self.b.grad = self.b.grad + db
        return dx[0] if c["squeeze"] else dx


class DepthwiseConv2D:
    """Depthwise 2-D convolution: one square kernel per channel.

    Input ``(H, W, channels)`` or ``(B, H, W, channels)``; the weight is
    ``(kernel, kernel, channels)`` and the bias ``(channels,)``. No
    mixing between channels.

    Args:
        channels: number of channels.
        kernel: kernel side.
        key: a ``jax.random`` key.
        stride: stride along both spatial axes, default 1.
        padding: zero padding on each side, default 0.

    Raises:
        ValueError: if ``kernel < 1``.

    References:
        Chollet (2017). arXiv:1610.02357. Page 1 and page 2, the
        depthwise convolution is the first half of a depthwise
        separable convolution.
    """

    def __init__(
        self,
        channels: int,
        kernel: int,
        key: Any,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        if kernel < 1:
            raise ValueError(f"kernel must be at least 1, got {kernel}")
        self.channels = channels
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        w = he_normal(key, (kernel * kernel * channels,)).reshape(
            kernel, kernel, channels
        )
        self.W = Param(w)
        self.b = Param(zeros_bias(channels))
        self._cache: dict[str, Any] | None = None

    def params(self) -> list[Param]:
        """The weight and the bias, in this order."""
        return [self.W, self.b]

    def __repr__(self) -> str:
        return (
            f"DepthwiseConv2D({self.channels}, {self.kernel}, "
            f"stride={self.stride}, padding={self.padding})"
        )

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Return the depthwise convolution of ``x``, plus the bias."""
        del training
        x = jnp.asarray(x)
        dt = jnp.result_type(x.dtype, self.W.value.dtype)
        x = x.astype(dt)
        squeeze = x.ndim == 3
        if squeeze:
            x = x[None]
        p = self.padding
        s = self.stride
        w = self.W.value.reshape(self.kernel, self.kernel, 1, self.channels).astype(dt)
        y = jax.lax.conv_general_dilated(
            x,
            w,
            window_strides=(s, s),
            padding=((p, p), (p, p)),
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=self.channels,
        ) + self.b.value.astype(dt)
        self._cache = dict(x=x, squeeze=squeeze)
        return y[0] if squeeze else y

    def backward(self, g: jax.Array) -> jax.Array:
        """Return ``dx`` and add to ``W.grad`` and ``b.grad``."""
        if self._cache is None:
            raise RuntimeError("call forward() before backward()")
        c = self._cache
        g = jnp.asarray(g)
        if c["squeeze"]:
            g = g[None]
        x = c["x"]
        p = self.padding
        s = self.stride
        w = self.W.value.reshape(self.kernel, self.kernel, 1, self.channels)
        dW_4d, dx = jax.vjp(
            lambda k, xv: jax.lax.conv_general_dilated(
                xv,
                k,
                window_strides=(s, s),
                padding=((p, p), (p, p)),
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
                feature_group_count=self.channels,
            ),
            w.astype(x.dtype),
            x,
        )[1](g)
        dW = dW_4d.reshape(self.kernel, self.kernel, self.channels)
        db = g.sum(axis=(0, 1, 2))
        self.W.grad = self.W.grad + dW
        self.b.grad = self.b.grad + db
        return dx[0] if c["squeeze"] else dx


class SeparableConv2D:
    """Separable 2-D convolution: depthwise then pointwise.

    A :class:`DepthwiseConv2D` followed by a pointwise
    :class:`~dense_armor.utility.neural.conv.Conv2D` with kernel 1.

    Args:
        c_in: input channels.
        c_out: output channels.
        kernel: kernel side of the depthwise part.
        key: a ``jax.random`` key.
        stride: stride along both spatial axes, default 1.
        padding: zero padding on each side, default 0.

    References:
        Chollet (2017). arXiv:1610.02357. Page 1 (abstract) and page 2:
        the depthwise separable convolution is a depthwise convolution
        followed by a pointwise convolution.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel: int,
        key: Any,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        k1, k2 = jax.random.split(key)
        self.depthwise = DepthwiseConv2D(
            c_in, kernel, k1, stride=stride, padding=padding
        )
        self.pointwise = Conv2D(c_in, c_out, 1, k2)
        self.c_in = c_in
        self.c_out = c_out
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self._cache: dict[str, Any] | None = None

    def params(self) -> list[Param]:
        """The depthwise parameters then the pointwise ones."""
        return self.depthwise.params() + self.pointwise.params()

    def __repr__(self) -> str:
        return (
            f"SeparableConv2D({self.c_in}, {self.c_out}, {self.kernel}, "
            f"stride={self.stride}, padding={self.padding})"
        )

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Return the depthwise then the pointwise convolution of ``x``."""
        y = self.depthwise.forward(x, training=training)
        out = self.pointwise.forward(y, training=training)
        self._cache = dict()
        return out

    def backward(self, g: jax.Array) -> jax.Array:
        """Return ``dx`` and add to the grads of the two parts."""
        if self._cache is None:
            raise RuntimeError("call forward() before backward()")
        d = self.pointwise.backward(g)
        dx = self.depthwise.backward(d)
        return dx
