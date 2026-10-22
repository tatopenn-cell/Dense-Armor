"""Transposed convolutions and nearest-neighbour upsampling.

The transposed convolution builds the output by the definition: every
input position ``i`` adds ``x[i] @ W[j]`` at the output position
``i * stride + j`` for each kernel offset ``j``, into a zero array of
length ``stride * (L - 1) + kernel + output_padding``; then the
``padding`` entries at each end are removed, then the bias is added.

The output size ``o' = s(i' - 1) + a + k - 2p`` is Relationship 14,
page 26 of Dumoulin and Visin (2016), with ``i' = L`` the input length,
``s`` the stride, ``a`` the ``output_padding``, ``k`` the kernel and
``p`` the padding. With ``a = 0`` it is Relationship 13 of the same
page.

``Upsample1D`` and ``Upsample2D`` repeat each position ``factor`` times
along the length (and along both spatial axes); they carry no
parameter. They are engineering definitions (the nearest-neighbour
repetition) and carry no citation.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from dense_armor.utility.neural.init import he_normal, zeros_bias
from dense_armor.utility.neural.params import Param


def _out_size(L: int, k: int, s: int, p: int, a: int) -> int:
    """``s*(L-1) + a + k - 2p`` (Relationship 14, page 26)."""
    o = s * (L - 1) + a + k - 2 * p
    if o < 1:
        raise ValueError(
            f"output size {o} is below 1 (L={L}, k={k}, s={s}, p={p}, a={a})"
        )
    return o


class ConvTranspose1D:
    """1-D transposed convolution, channels last.

    Input ``(L, c_in)`` or ``(B, L, c_in)``, output ``(o', c_out)`` or
    ``(B, o', c_out)``. The kernel is ``(kernel, c_in, c_out)`` and the
    bias ``(c_out,)``, as in :class:`~dense_armor.utility.neural.conv.Conv1D`.

    Args:
        c_in: input channels.
        c_out: output channels.
        kernel: kernel length.
        key: a ``jax.random`` key.
        stride: stride along the length, default 1.
        padding: number of output positions removed at each end, default 0.
        output_padding: ``a`` of Relationship 14, in ``[0, stride-1]``,
            default 0.

    Raises:
        ValueError: on ``kernel < 1``, ``padding < 0``,
            ``output_padding < 0`` or ``output_padding >= stride``, or
            an output size below 1.

    References:
        Dumoulin, V., Visin, F. (2016). A guide to convolution
        arithmetic for deep learning. arXiv:1603.07285. Page 26,
        Relationship 13: ``o' = s(i' - 1) + k - 2p``; Relationship 14:
        ``o' = s(i' - 1) + a + k - 2p`` with ``a = (i + 2p - k) mod s``
        the number of zeros added to the bottom and right edges of the
        input.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.convtranspose import ConvTranspose1D
        >>> c = ConvTranspose1D(1, 1, 3, jax.random.PRNGKey(0), stride=2)
        >>> c.W.value = jnp.array([[[1.0]], [[10.0]], [[100.0]]])
        >>> c.b.value = jnp.array([0.0])
        >>> y = c.forward(jnp.array([[1.0], [2.0]]))
        >>> y.reshape(-1).tolist()
        [1.0, 10.0, 102.0, 20.0, 200.0]
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel: int,
        key: Any,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
    ) -> None:
        if kernel < 1:
            raise ValueError(f"kernel must be at least 1, got {kernel}")
        if padding < 0:
            raise ValueError(f"padding must be non-negative, got {padding}")
        if output_padding < 0:
            raise ValueError(
                f"output_padding must be non-negative, got {output_padding}"
            )
        if output_padding >= stride:
            raise ValueError(
                f"output_padding {output_padding} must be < stride {stride}"
            )
        self.c_in = c_in
        self.c_out = c_out
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        w = he_normal(key, (kernel * c_in, c_out)).reshape(kernel, c_in, c_out)
        self.W = Param(w)
        self.b = Param(zeros_bias(c_out))
        self._cache: dict[str, Any] | None = None

    def params(self) -> list[Param]:
        """The weight and the bias, in this order."""
        return [self.W, self.b]

    def __repr__(self) -> str:
        return (
            f"ConvTranspose1D({self.c_in}, {self.c_out}, {self.kernel}, "
            f"stride={self.stride}, padding={self.padding}, "
            f"output_padding={self.output_padding})"
        )

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Return the transposed convolution of ``x`` with the weight, plus the bias."""
        del training
        x = jnp.asarray(x)
        dt = jnp.result_type(x.dtype, self.W.value.dtype)
        x = x.astype(dt)
        squeeze = x.ndim == 2
        if squeeze:
            x = x[None]
        B, L, _ = x.shape
        _out_size(L, self.kernel, self.stride, self.padding, self.output_padding)
        L_full = self.stride * (L - 1) + self.kernel + self.output_padding
        out = jnp.zeros((B, L_full, self.c_out), dtype=dt)
        for i in range(L):
            for j in range(self.kernel):
                out = out.at[:, i * self.stride + j, :].add(
                    x[:, i, :] @ self.W.value[j].astype(dt)
                )
        if self.padding > 0:
            out = out[:, self.padding : L_full - self.padding, :]
        out = out + self.b.value.astype(dt)
        self._cache = dict(x=x, squeeze=squeeze, L_full=L_full)
        return out[0] if squeeze else out

    def backward(self, g: jax.Array) -> jax.Array:
        """Return ``dx`` and add to ``W.grad`` and ``b.grad``."""
        if self._cache is None:
            raise RuntimeError("call forward() before backward()")
        c = self._cache
        g = jnp.asarray(g)
        if c["squeeze"]:
            g = g[None]
        x = c["x"]
        L_full = c["L_full"]
        B, L, _ = x.shape
        L_out = g.shape[1]
        g_full = jnp.zeros((B, L_full, self.c_out), dtype=g.dtype)
        g_full = g_full.at[:, self.padding : self.padding + L_out, :].set(g)
        dx = jnp.zeros_like(x)
        dW = jnp.zeros_like(self.W.value)
        for i in range(L):
            for j in range(self.kernel):
                g_ij = g_full[:, i * self.stride + j, :]
                dx = dx.at[:, i, :].add(g_ij @ self.W.value[j].T)
                dW = dW.at[j].add(x[:, i, :].T @ g_ij)
        db = g_full.sum(axis=(0, 1))
        self.W.grad = self.W.grad + dW
        self.b.grad = self.b.grad + db
        return dx[0] if c["squeeze"] else dx


class ConvTranspose2D:
    """2-D transposed convolution, channels last.

    Input ``(H, W, c_in)`` or ``(B, H, W, c_in)``, output
    ``(o_h', o_w', c_out)`` or ``(B, o_h', o_w', c_out)``. The kernel is
    ``(kernel, kernel, c_in, c_out)`` and the bias ``(c_out,)``.

    Args:
        c_in: input channels.
        c_out: output channels.
        kernel: kernel side.
        key: a ``jax.random`` key.
        stride: stride along both spatial axes, default 1.
        padding: number of output positions removed at each border, default 0.
        output_padding: ``a`` of Relationship 14, in ``[0, stride-1]``,
            default 0.

    Raises:
        ValueError: as in :class:`ConvTranspose1D`.

    References:
        Dumoulin, V., Visin, F. (2016). arXiv:1603.07285. Page 26,
        Relationship 13 and Relationship 14.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel: int,
        key: Any,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
    ) -> None:
        if kernel < 1:
            raise ValueError(f"kernel must be at least 1, got {kernel}")
        if padding < 0:
            raise ValueError(f"padding must be non-negative, got {padding}")
        if output_padding < 0:
            raise ValueError(
                f"output_padding must be non-negative, got {output_padding}"
            )
        if output_padding >= stride:
            raise ValueError(
                f"output_padding {output_padding} must be < stride {stride}"
            )
        self.c_in = c_in
        self.c_out = c_out
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        w = he_normal(key, (kernel * kernel * c_in, c_out)).reshape(
            kernel, kernel, c_in, c_out
        )
        self.W = Param(w)
        self.b = Param(zeros_bias(c_out))
        self._cache: dict[str, Any] | None = None

    def params(self) -> list[Param]:
        """The weight and the bias, in this order."""
        return [self.W, self.b]

    def __repr__(self) -> str:
        return (
            f"ConvTranspose2D({self.c_in}, {self.c_out}, {self.kernel}, "
            f"stride={self.stride}, padding={self.padding}, "
            f"output_padding={self.output_padding})"
        )

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Return the transposed convolution of ``x`` with the weight, plus the bias."""
        del training
        x = jnp.asarray(x)
        dt = jnp.result_type(x.dtype, self.W.value.dtype)
        x = x.astype(dt)
        squeeze = x.ndim == 3
        if squeeze:
            x = x[None]
        B, H, W, _ = x.shape
        k = self.kernel
        s = self.stride
        _out_size(H, k, s, self.padding, self.output_padding)
        _out_size(W, k, s, self.padding, self.output_padding)
        H_full = s * (H - 1) + k + self.output_padding
        W_full = s * (W - 1) + k + self.output_padding
        out = jnp.zeros((B, H_full, W_full, self.c_out), dtype=dt)
        for i in range(H):
            for j in range(W):
                for ki in range(k):
                    for kj in range(k):
                        out = out.at[:, i * s + ki, j * s + kj, :].add(
                            x[:, i, j, :] @ self.W.value[ki, kj].astype(dt)
                        )
        if self.padding > 0:
            out = out[
                :,
                self.padding : H_full - self.padding,
                self.padding : W_full - self.padding,
                :,
            ]
        out = out + self.b.value.astype(dt)
        self._cache = dict(x=x, squeeze=squeeze, H_full=H_full, W_full=W_full)
        return out[0] if squeeze else out

    def backward(self, g: jax.Array) -> jax.Array:
        """Return ``dx`` and add to ``W.grad`` and ``b.grad``."""
        if self._cache is None:
            raise RuntimeError("call forward() before backward()")
        c = self._cache
        g = jnp.asarray(g)
        if c["squeeze"]:
            g = g[None]
        x = c["x"]
        H_full = c["H_full"]
        W_full = c["W_full"]
        B, H, W, _ = x.shape
        k = self.kernel
        s = self.stride
        g_full = jnp.zeros((B, H_full, W_full, self.c_out), dtype=g.dtype)
        gh, gw = g.shape[1], g.shape[2]
        g_full = g_full.at[
            :,
            self.padding : self.padding + gh,
            self.padding : self.padding + gw,
            :,
        ].set(g)
        dx = jnp.zeros_like(x)
        dW = jnp.zeros_like(self.W.value)
        for i in range(H):
            for j in range(W):
                for ki in range(k):
                    for kj in range(k):
                        g_ij = g_full[:, i * s + ki, j * s + kj, :]
                        dx = dx.at[:, i, j, :].add(g_ij @ self.W.value[ki, kj].T)
                        dW = dW.at[ki, kj].add(x[:, i, j, :].T @ g_ij)
        db = g_full.sum(axis=(0, 1, 2))
        self.W.grad = self.W.grad + dW
        self.b.grad = self.b.grad + db
        return dx[0] if c["squeeze"] else dx


class Upsample1D:
    """Nearest-neighbour upsampling along the length, channels last."""

    def __init__(self, factor: int) -> None:
        if factor < 1:
            raise ValueError(f"factor must be at least 1, got {factor}")
        self.factor = int(factor)
        self._cache: dict[str, Any] | None = None

    def params(self) -> list[Param]:
        """No parameter."""
        return []

    def __repr__(self) -> str:
        return f"Upsample1D(factor={self.factor})"

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Repeat each position ``factor`` times along the length."""
        del training
        x = jnp.asarray(x)
        squeeze = x.ndim == 2
        if squeeze:
            x = x[None]
        y = jnp.repeat(x, self.factor, axis=1)
        self._cache = dict(squeeze=squeeze, L=x.shape[1])
        return y[0] if squeeze else y

    def backward(self, g: jax.Array) -> jax.Array:
        """Sum each block of ``factor`` positions."""
        if self._cache is None:
            raise RuntimeError("call forward() before backward()")
        c = self._cache
        g = jnp.asarray(g)
        if c["squeeze"]:
            g = g[None]
        B, L_full, C = g.shape
        L = c["L"]
        g3 = g.reshape(B, L, self.factor, C)
        dx = g3.sum(axis=2)
        return dx[0] if c["squeeze"] else dx


class Upsample2D:
    """Nearest-neighbour upsampling along both spatial axes, channels last."""

    def __init__(self, factor: int) -> None:
        if factor < 1:
            raise ValueError(f"factor must be at least 1, got {factor}")
        self.factor = int(factor)
        self._cache: dict[str, Any] | None = None

    def params(self) -> list[Param]:
        """No parameter."""
        return []

    def __repr__(self) -> str:
        return f"Upsample2D(factor={self.factor})"

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Repeat each position ``factor`` times along both spatial axes."""
        del training
        x = jnp.asarray(x)
        squeeze = x.ndim == 3
        if squeeze:
            x = x[None]
        y = jnp.repeat(jnp.repeat(x, self.factor, axis=1), self.factor, axis=2)
        self._cache = dict(squeeze=squeeze, H=x.shape[1], W=x.shape[2])
        return y[0] if squeeze else y

    def backward(self, g: jax.Array) -> jax.Array:
        """Sum each block of ``factor x factor`` positions."""
        if self._cache is None:
            raise RuntimeError("call forward() before backward()")
        c = self._cache
        g = jnp.asarray(g)
        if c["squeeze"]:
            g = g[None]
        B, H_full, W_full, C = g.shape
        H, W = c["H"], c["W"]
        f = self.factor
        g4 = g.reshape(B, H, f, W, f, C)
        dx = g4.sum(axis=(2, 4))
        return dx[0] if c["squeeze"] else dx
