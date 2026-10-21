"""Convolution and pooling layers.

The forward pass uses ``jax.lax.conv_general_dilated``; the backward
pass is written by hand. The output size follows Relationship 6, page
15 of Dumoulin and Visin 2018 for a convolution, and Relationship 7,
page 18, for pooling. The He initialisation uses the fan-in of a
convolution, ``k_l^2 c_l`` (Eqn.(10), page 4 of He et al. 2015).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from dense_armor.utility.neural.init import he_normal, zeros_bias
from dense_armor.utility.neural.params import Param


def _conv_out_size(i: int, k: int, s: int, p: int) -> int:
    """Output size ``floor((i + 2p - k)/s) + 1`` (Relationship 6, p.15)."""
    if i + 2 * p - k < 0:
        raise ValueError(f"kernel {k} with padding {p} is larger than input {i}")
    return (i + 2 * p - k) // s + 1


def _pool_out_size(i: int, k: int, s: int) -> int:
    """Output size ``floor((i - k)/s) + 1`` (Relationship 7, p.18)."""
    if i - k < 0:
        raise ValueError(f"kernel {k} is larger than input {i}")
    return (i - k) // s + 1


class Conv1D:
    """1-D cross-correlation with zero padding, optional causal shift.

    Input ``(L, c_in)`` or ``(B, L, c_in)``, output ``(L_out, c_out)``
    or ``(B, L_out, c_out)``. The kernel is ``(kernel, c_in, c_out)``
    and the bias ``(c_out,)``.

    Args:
        c_in: input channels.
        c_out: output channels.
        kernel: kernel length.
        key: a ``jax.random`` key.
        stride: stride along the length, default 1.
        padding: zero padding on each side, default 0.
        causal: if True, pad on the left only with ``kernel - 1`` zeros,
            require ``padding = 0`` and ``stride = 1``, and return an
            output with the length of the input.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.conv import Conv1D
        >>> c = Conv1D(2, 3, 3, jax.random.PRNGKey(0), padding=1)
        >>> c.forward(jnp.ones((5, 2))).shape
        (5, 3)
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel: int,
        key: Any,
        stride: int = 1,
        padding: int = 0,
        causal: bool = False,
    ) -> None:
        if kernel < 1:
            raise ValueError(f"kernel must be at least 1, got {kernel}")
        if causal and stride != 1:
            raise ValueError("causal=True requires stride=1")
        if causal and padding != 0:
            raise ValueError(
                "causal=True requires padding=0; the layer pads on the left only"
            )
        self.c_in = c_in
        self.c_out = c_out
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.causal = causal
        w = he_normal(key, (kernel * c_in, c_out)).reshape(kernel, c_in, c_out)
        self.W = Param(w)
        self.b = Param(zeros_bias(c_out))
        self.x_: jax.Array | None = None
        self._p_left = kernel - 1 if causal else padding
        self._p_right = 0 if causal else padding

    def params(self) -> list[Param]:
        """The weight and the bias, in this order."""
        return [self.W, self.b]

    def __repr__(self) -> str:
        return (
            f"Conv1D({self.c_in}, {self.c_out}, {self.kernel}, "
            f"stride={self.stride}, padding={self.padding}, causal={self.causal})"
        )

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Return the cross-correlation of ``x`` with the weight, plus the bias."""
        del training
        x = jnp.asarray(x)
        dt = jnp.result_type(x.dtype, self.W.value.dtype)
        x = x.astype(dt)
        squeeze = x.ndim == 2
        if squeeze:
            x = x[None]
        _conv_out_size(
            x.shape[1], self.kernel, self.stride, self._p_left + self._p_right
        )
        y = jax.lax.conv_general_dilated(
            x,
            self.W.value.astype(dt),
            window_strides=(self.stride,),
            padding=((self._p_left, self._p_right),),
            dimension_numbers=("NWC", "WIO", "NWC"),
        ) + self.b.value.astype(dt)
        self.x_ = x
        self._dt = dt
        return y[0] if squeeze else y

    def backward(self, g: jax.Array) -> jax.Array:
        """Return ``dx`` and add to ``W.grad`` and ``b.grad``."""
        if self.x_ is None:
            raise RuntimeError("call forward() before backward()")
        g = jnp.asarray(g)
        dt = self._dt
        g = g.astype(dt)
        squeeze = g.ndim == 2
        if squeeze:
            g = g[None]
        W_val = self.W.value.astype(dt)
        B, L, _ = self.x_.shape
        L_out = g.shape[1]
        s = self.stride
        K = self.kernel
        p_left = self._p_left
        p_right = self._p_right
        x_padded = jnp.pad(self.x_, [(0, 0), (p_left, p_right), (0, 0)])
        db = g.sum(axis=(0, 1))
        dW = jnp.zeros(self.W.value.shape, dtype=dt)
        for k in range(K):
            idx = jnp.arange(L_out) * s + k
            x_slice = x_padded[:, idx, :]
            dW = dW.at[k].set(jnp.einsum("nlo,nli->io", g, x_slice))
        dx_padded = jnp.zeros((B, L + p_left + p_right, self.c_in), dtype=dt)
        for k in range(K):
            idx = jnp.arange(L_out) * s + k
            contrib = jnp.einsum("nlo,io->nli", g, W_val[k])
            dx_padded = dx_padded.at[:, idx, :].add(contrib)
        dx = dx_padded[:, p_left : p_left + L, :]
        self.W.grad = self.W.grad + dW.astype(self.W.grad.dtype)
        self.b.grad = self.b.grad + db.astype(self.b.grad.dtype)
        return dx[0] if squeeze else dx

    def init_buffer(self) -> jax.Array:
        """The empty buffer of the causal step: ``kernel - 1`` zeros."""
        if not self.causal:
            raise ValueError("init_buffer() requires causal=True")
        return jnp.zeros((self.kernel - 1, self.c_in), dtype=jnp.float32)

    def step(self, x_t: jax.Array, buffer: jax.Array) -> tuple[jax.Array, jax.Array]:
        """One causal step: ``(y_t, new_buffer)``."""
        if not self.causal:
            raise ValueError("step() requires causal=True")
        x_t = jnp.asarray(x_t, dtype=jnp.float32)
        seq = jnp.concatenate([buffer, x_t[None]], axis=0)
        y = jnp.einsum("ki,kio->o", seq, self.W.value) + self.b.value
        return y, seq[1:]


class Conv2D:
    """2-D cross-correlation with zero padding, channels last.

    Input ``(H, W, c_in)`` or ``(B, H, W, c_in)``, output
    ``(H_out, W_out, c_out)`` or ``(B, H_out, W_out, c_out)``.

    Args:
        c_in: input channels.
        c_out: output channels.
        kernel: square kernel side.
        key: a ``jax.random`` key.
        stride: stride on both sides, default 1.
        padding: zero padding on each side, default 0.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.conv import Conv2D
        >>> c = Conv2D(1, 2, 3, jax.random.PRNGKey(0))
        >>> c.forward(jnp.ones((5, 5, 1))).shape
        (3, 3, 2)
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
        w = he_normal(key, (kernel * kernel * c_in, c_out)).reshape(
            kernel, kernel, c_in, c_out
        )
        self.W = Param(w)
        self.b = Param(zeros_bias(c_out))
        self.x_: jax.Array | None = None

    def params(self) -> list[Param]:
        """The weight and the bias, in this order."""
        return [self.W, self.b]

    def __repr__(self) -> str:
        return (
            f"Conv2D({self.c_in}, {self.c_out}, {self.kernel}, "
            f"stride={self.stride}, padding={self.padding})"
        )

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Return the cross-correlation of ``x`` with the weight, plus the bias."""
        del training
        x = jnp.asarray(x)
        dt = jnp.result_type(x.dtype, self.W.value.dtype)
        x = x.astype(dt)
        squeeze = x.ndim == 3
        if squeeze:
            x = x[None]
        _conv_out_size(x.shape[1], self.kernel, self.stride, self.padding)
        _conv_out_size(x.shape[2], self.kernel, self.stride, self.padding)
        y = jax.lax.conv_general_dilated(
            x,
            self.W.value.astype(dt),
            window_strides=(self.stride, self.stride),
            padding=((self.padding, self.padding), (self.padding, self.padding)),
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        ) + self.b.value.astype(dt)
        self.x_ = x
        self._dt = dt
        return y[0] if squeeze else y

    def backward(self, g: jax.Array) -> jax.Array:
        """Return ``dx`` and add to ``W.grad`` and ``b.grad``."""
        if self.x_ is None:
            raise RuntimeError("call forward() before backward()")
        g = jnp.asarray(g)
        dt = self._dt
        g = g.astype(dt)
        squeeze = g.ndim == 3
        if squeeze:
            g = g[None]
        W_val = self.W.value.astype(dt)
        B, H, W, _ = self.x_.shape
        Ho = g.shape[1]
        Wo = g.shape[2]
        s = self.stride
        K = self.kernel
        p = self.padding
        x_padded = jnp.pad(self.x_, [(0, 0), (p, p), (p, p), (0, 0)])
        db = g.sum(axis=(0, 1, 2))
        dW = jnp.zeros(self.W.value.shape, dtype=dt)
        for kh in range(K):
            for kw in range(K):
                h_idx = jnp.arange(Ho) * s + kh
                w_idx = jnp.arange(Wo) * s + kw
                x_slice = x_padded[:, h_idx[:, None], w_idx[None, :], :]
                dW = dW.at[kh, kw].set(jnp.einsum("nhwo,nhwi->io", g, x_slice))
        dx_padded = jnp.zeros((B, H + 2 * p, W + 2 * p, self.c_in), dtype=dt)
        for kh in range(K):
            for kw in range(K):
                h_idx = jnp.arange(Ho) * s + kh
                w_idx = jnp.arange(Wo) * s + kw
                contrib = jnp.einsum("nhwo,io->nhwi", g, W_val[kh, kw])
                dx_padded = dx_padded.at[:, h_idx[:, None], w_idx[None, :], :].add(
                    contrib
                )
        dx = dx_padded[:, p : p + H, p : p + W, :]
        self.W.grad = self.W.grad + dW
        self.b.grad = self.b.grad + db
        return dx[0] if squeeze else dx


class MaxPool1D:
    """1-D max pooling with the first maximum going to the gradient.

    Input ``(L, c)`` or ``(B, L, c)``. ``params()`` is empty.

    Args:
        kernel: window length.
        stride: stride, default ``kernel``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.conv import MaxPool1D
        >>> p = MaxPool1D(2)
        >>> p.forward(jnp.array([[1.0], [3.0], [2.0], [5.0]])).tolist()
        [[3.0], [5.0]]
    """

    def __init__(self, kernel: int, stride: int | None = None) -> None:
        if kernel < 1:
            raise ValueError(f"kernel must be at least 1, got {kernel}")
        self.kernel = kernel
        self.stride = kernel if stride is None else stride
        self.x_: jax.Array | None = None
        self._argmax: list[jax.Array] = []

    def __repr__(self) -> str:
        return f"MaxPool1D({self.kernel}, stride={self.stride})"

    def params(self) -> list[Param]:
        """Empty: pooling has no parameters."""
        return []

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Return the maximum of each window."""
        del training
        x = jnp.asarray(x)
        squeeze = x.ndim == 2
        if squeeze:
            x = x[None]
        L = x.shape[1]
        L_out = _pool_out_size(L, self.kernel, self.stride)
        ys = []
        argmaxes = []
        for i in range(L_out):
            start = i * self.stride
            window = x[:, start : start + self.kernel, :]
            idx = jnp.argmax(window, axis=1)
            val = jnp.take_along_axis(window, idx[:, None, :], axis=1)[:, 0, :]
            ys.append(val)
            argmaxes.append(idx)
        self.x_ = x
        self._argmax = argmaxes
        y = jnp.stack(ys, axis=1)
        return y[0] if squeeze else y

    def backward(self, g: jax.Array) -> jax.Array:
        """Return ``dx``, with ``g`` at the first maximum of each window."""
        if self.x_ is None:
            raise RuntimeError("call forward() before backward()")
        g = jnp.asarray(g)
        squeeze = g.ndim == 2
        if squeeze:
            g = g[None]
        B, _, c = g.shape[0], self.x_.shape[1], g.shape[2]
        dx = jnp.zeros(self.x_.shape, dtype=g.dtype)
        b_idx = jnp.arange(B)[:, None]
        c_idx = jnp.arange(c)[None, :]
        for i, idx in enumerate(self._argmax):
            positions = i * self.stride + idx
            dx = dx.at[b_idx, positions, c_idx].add(g[:, i, :])
        return dx[0] if squeeze else dx


class MaxPool2D:
    """2-D max pooling with the first maximum going to the gradient.

    Input ``(H, W, c)`` or ``(B, H, W, c)``. ``params()`` is empty.

    Args:
        kernel: square window side.
        stride: stride, default ``kernel``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.conv import MaxPool2D
        >>> p = MaxPool2D(2)
        >>> p.forward(jnp.ones((4, 4, 1))).shape
        (2, 2, 1)
    """

    def __init__(self, kernel: int, stride: int | None = None) -> None:
        if kernel < 1:
            raise ValueError(f"kernel must be at least 1, got {kernel}")
        self.kernel = kernel
        self.stride = kernel if stride is None else stride
        self.x_: jax.Array | None = None
        self._argmax: list[list[tuple[jax.Array, jax.Array]]] = []

    def __repr__(self) -> str:
        return f"MaxPool2D({self.kernel}, stride={self.stride})"

    def params(self) -> list[Param]:
        """Empty: pooling has no parameters."""
        return []

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Return the maximum of each 2-D window."""
        del training
        x = jnp.asarray(x)
        squeeze = x.ndim == 3
        if squeeze:
            x = x[None]
        H, W = x.shape[1], x.shape[2]
        Ho = _pool_out_size(H, self.kernel, self.stride)
        Wo = _pool_out_size(W, self.kernel, self.stride)
        ys = []
        argmaxes: list[list[tuple[jax.Array, jax.Array]]] = []
        for i in range(Ho):
            row = []
            row_argmax = []
            for j in range(Wo):
                hs = i * self.stride
                ws = j * self.stride
                window = x[:, hs : hs + self.kernel, ws : ws + self.kernel, :]
                flat = window.reshape(window.shape[0], -1, window.shape[-1])
                idx = jnp.argmax(flat, axis=1)
                val = jnp.take_along_axis(flat, idx[:, None, :], axis=1)[:, 0, :]
                row.append(val)
                row_argmax.append((idx // self.kernel, idx % self.kernel))
            ys.append(jnp.stack(row, axis=1))
            argmaxes.append(row_argmax)
        self.x_ = x
        self._argmax = argmaxes
        y = jnp.stack(ys, axis=1)
        return y[0] if squeeze else y

    def backward(self, g: jax.Array) -> jax.Array:
        """Return ``dx``, with ``g`` at the first maximum of each window."""
        if self.x_ is None:
            raise RuntimeError("call forward() before backward()")
        g = jnp.asarray(g)
        squeeze = g.ndim == 3
        if squeeze:
            g = g[None]
        B, c = g.shape[0], g.shape[3]
        dx = jnp.zeros(self.x_.shape, dtype=g.dtype)
        b_idx = jnp.arange(B)[:, None]
        c_idx = jnp.arange(c)[None, :]
        for i in range(len(self._argmax)):
            for j in range(len(self._argmax[i])):
                hi, wi = self._argmax[i][j]
                hpos = i * self.stride + hi
                wpos = j * self.stride + wi
                dx = dx.at[b_idx, hpos, wpos, c_idx].add(g[:, i, j, :])
        return dx[0] if squeeze else dx


class AvgPool1D:
    """1-D average pooling. Input ``(L, c)`` or ``(B, L, c)``.

    Args:
        kernel: window length.
        stride: stride, default ``kernel``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.conv import AvgPool1D
        >>> p = AvgPool1D(2)
        >>> p.forward(jnp.array([[1.0], [3.0], [2.0], [4.0]])).tolist()
        [[2.0], [3.0]]
    """

    def __init__(self, kernel: int, stride: int | None = None) -> None:
        if kernel < 1:
            raise ValueError(f"kernel must be at least 1, got {kernel}")
        self.kernel = kernel
        self.stride = kernel if stride is None else stride
        self.x_: jax.Array | None = None

    def __repr__(self) -> str:
        return f"AvgPool1D({self.kernel}, stride={self.stride})"

    def params(self) -> list[Param]:
        """Empty: pooling has no parameters."""
        return []

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Return the mean of each window."""
        del training
        x = jnp.asarray(x)
        squeeze = x.ndim == 2
        if squeeze:
            x = x[None]
        L = x.shape[1]
        L_out = _pool_out_size(L, self.kernel, self.stride)
        ys = []
        for i in range(L_out):
            start = i * self.stride
            window = x[:, start : start + self.kernel, :]
            ys.append(window.mean(axis=1))
        self.x_ = x
        y = jnp.stack(ys, axis=1)
        return y[0] if squeeze else y

    def backward(self, g: jax.Array) -> jax.Array:
        """Spread ``g / kernel`` over each window."""
        if self.x_ is None:
            raise RuntimeError("call forward() before backward()")
        g = jnp.asarray(g)
        squeeze = g.ndim == 2
        if squeeze:
            g = g[None]
        B, _, c = g.shape[0], self.x_.shape[1], g.shape[2]
        dx = jnp.zeros(self.x_.shape, dtype=g.dtype)
        b_idx = jnp.arange(B)[:, None]
        c_idx = jnp.arange(c)[None, :]
        share = g / float(self.kernel)
        for i in range(g.shape[1]):
            for k in range(self.kernel):
                pos = i * self.stride + k
                dx = dx.at[b_idx, pos, c_idx].add(share[:, i, :])
        return dx[0] if squeeze else dx


class AvgPool2D:
    """2-D average pooling. Input ``(H, W, c)`` or ``(B, H, W, c)``.

    Args:
        kernel: square window side.
        stride: stride, default ``kernel``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.conv import AvgPool2D
        >>> p = AvgPool2D(2)
        >>> p.forward(jnp.ones((4, 4, 1))).tolist()[0][0]
        [1.0]
    """

    def __init__(self, kernel: int, stride: int | None = None) -> None:
        if kernel < 1:
            raise ValueError(f"kernel must be at least 1, got {kernel}")
        self.kernel = kernel
        self.stride = kernel if stride is None else stride
        self.x_: jax.Array | None = None

    def __repr__(self) -> str:
        return f"AvgPool2D({self.kernel}, stride={self.stride})"

    def params(self) -> list[Param]:
        """Empty: pooling has no parameters."""
        return []

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Return the mean of each 2-D window."""
        del training
        x = jnp.asarray(x)
        squeeze = x.ndim == 3
        if squeeze:
            x = x[None]
        H, W = x.shape[1], x.shape[2]
        Ho = _pool_out_size(H, self.kernel, self.stride)
        Wo = _pool_out_size(W, self.kernel, self.stride)
        ys = []
        for i in range(Ho):
            row = []
            for j in range(Wo):
                hs = i * self.stride
                ws = j * self.stride
                window = x[:, hs : hs + self.kernel, ws : ws + self.kernel, :]
                row.append(window.mean(axis=(1, 2)))
            ys.append(jnp.stack(row, axis=1))
        self.x_ = x
        y = jnp.stack(ys, axis=1)
        return y[0] if squeeze else y

    def backward(self, g: jax.Array) -> jax.Array:
        """Spread ``g / (kernel*kernel)`` over each window."""
        if self.x_ is None:
            raise RuntimeError("call forward() before backward()")
        g = jnp.asarray(g)
        squeeze = g.ndim == 3
        if squeeze:
            g = g[None]
        B, c = g.shape[0], g.shape[3]
        dx = jnp.zeros(self.x_.shape, dtype=g.dtype)
        b_idx = jnp.arange(B)[:, None]
        c_idx = jnp.arange(c)[None, :]
        share = g / float(self.kernel * self.kernel)
        for i in range(g.shape[1]):
            for j in range(g.shape[2]):
                for kh in range(self.kernel):
                    for kw in range(self.kernel):
                        hpos = i * self.stride + kh
                        wpos = j * self.stride + kw
                        dx = dx.at[b_idx, hpos, wpos, c_idx].add(share[:, i, j, :])
        return dx[0] if squeeze else dx


class Flatten:
    """Flatten every axis after the batch axis.

    ``(B, ...)`` goes to ``(B, n)`` and ``(...)`` goes to ``(n,)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.conv import Flatten
        >>> f = Flatten()
        >>> f.forward(jnp.ones((2, 3, 4))).shape
        (2, 12)
    """

    def __init__(self) -> None:
        self.x_: jax.Array | None = None

    def __repr__(self) -> str:
        return "Flatten()"

    def params(self) -> list[Param]:
        """Empty: flatten has no parameters."""
        return []

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Return ``x`` with every axis after the batch one merged."""
        del training
        x = jnp.asarray(x)
        self.x_ = x
        if x.ndim <= 1:
            return x
        return x.reshape(x.shape[0], -1) if x.ndim >= 2 else x

    def backward(self, g: jax.Array) -> jax.Array:
        """Restore the shape of the input."""
        if self.x_ is None:
            raise RuntimeError("call forward() before backward()")
        return jnp.asarray(g).reshape(self.x_.shape)
