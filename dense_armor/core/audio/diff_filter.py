"""Differentiable filters in JAX: the cookbook low-pass, a biquad layer, a time-varying all-pole filter.

Sources:

- M. Comunita, C. J. Steinmetz, J. D. Reiss, "NablAFx: A framework for differentiable black-box
  and gray-box modeling of audio effects" (2025), Sec. 2.2.1: the filters are biquads with the
  coefficients of the Audio EQ Cookbook (the low-pass formulas are given there), and a cascade of
  second-order sections has the product of their transfer functions (its Eq. 2).
- C.-Y. Yu, C. Mitcheltree, A. Carson, S. Bilbao, J. D. Reiss, G. Fazekas, "Differentiable
  all-pole filters for time-varying audio systems", DAFx-24 (2024): the filter is its Eq. (1),
  the gradient with respect to the input is Eq. (7) and the gradient with respect to the
  coefficients is Eq. (8), both in Sec. 3.

``JaxBiquad`` follows the layer pattern of ``dense_armor.core.deep.neural``: its coefficients are
``Param`` objects, ``forward(x, training=True)`` filters and ``backward(g)`` returns the gradient
of the input and adds to the gradient of the coefficients, so it goes into a ``Sequential``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from dense_armor.core.deep.neural.params import Param


def _concrete(*values: Any) -> bool:
    return not any(isinstance(v, jax.core.Tracer) for v in values)


def jax_lowpass(
    f0: Any, sr: Any, q: Any = 0.7071067811865476
) -> tuple[jax.Array, jax.Array]:
    """The low-pass biquad of the cookbook as ``jax`` arrays, with gradients for ``f0``, ``sr`` and ``q``.

    The same numbers as ``dense_armor.core.audio.biquad.lowpass``, normalised by ``a0``, so
    ``b`` and ``a`` have three entries and ``a[0]`` is 1.

    Args:
        f0: cutoff frequency in Hz, in ``(0, sr / 2)``.
        sr: sampling rate in Hz.
        q: quality factor, positive.

    Returns:
        ``(b, a)``, two arrays of shape ``(3,)``.

    Raises:
        ValueError: if ``f0`` is not in ``(0, sr / 2)`` or ``q`` is not positive. The test is
            made only when the arguments are numbers; inside ``jax.grad`` or ``jax.jit`` they are
            traced and it is skipped.

    Example:
        >>> b, a = jax_lowpass(1000.0, 8000.0)
        >>> b.shape, a.shape
        ((3,), (3,))
    """
    if _concrete(f0, sr, q):
        if not 0.0 < float(f0) < float(sr) / 2.0:
            raise ValueError(f"f0 must be in (0, sr/2), got {f0} with sr {sr}")
        if float(q) <= 0.0:
            raise ValueError(f"q must be positive, got {q}")
    f0, sr, q = jnp.asarray(f0), jnp.asarray(sr), jnp.asarray(q)
    dt = jnp.result_type(f0.dtype, sr.dtype, q.dtype, jnp.float32)
    w0 = 2.0 * jnp.pi * f0.astype(dt) / sr.astype(dt)
    c = jnp.cos(w0)
    al = jnp.sin(w0) / (2.0 * q.astype(dt))
    a0 = 1.0 + al
    b = jnp.stack([(1.0 - c) / 2.0, 1.0 - c, (1.0 - c) / 2.0]) / a0
    a = jnp.stack([a0, -2.0 * c, 1.0 - al]) / a0
    return b, a


@jax.jit
def jax_biquad(b: Any, a: Any, x: Any) -> jax.Array:
    """Filter ``x`` with the biquad ``(b, a)``, ``y[n] = b0 x[n] + b1 x[n-1] + b2 x[n-2] - a1 y[n-1] - a2 y[n-2]``.

    The recurrence is a ``lax.scan``, so ``jax.grad`` goes through it. The coefficients are divided
    by ``a[0]``. The time axis is axis 0 for a 1D input and axis ``-2`` otherwise, the layout of
    ``Conv1D``: ``(L,)``, ``(L, channels)`` or ``(batch, L, channels)``.

    Args:
        b: three numerator coefficients.
        a: three denominator coefficients.
        x: the signal.

    Returns:
        The filtered signal, with the shape of ``x``.

    Example:
        >>> b, a = jax_lowpass(1000.0, 8000.0)
        >>> float(jax_biquad(b, a, jnp.array([1.0, 0.0, 0.0]))[0]) == float(b[0])
        True
    """
    b, a, x = jnp.asarray(b), jnp.asarray(a), jnp.asarray(x)
    dt = jnp.result_type(b.dtype, a.dtype, x.dtype, jnp.float32)
    b, a, x = b.astype(dt), a.astype(dt), x.astype(dt)
    b, a = b / a[0], a / a[0]
    xt = x if x.ndim == 1 else jnp.moveaxis(x, -2, 0)

    def step(carry, xn):
        x1, x2, y1, y2 = carry
        y = b[0] * xn + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
        return (xn, x1, y, y1), y

    z = jnp.zeros_like(xt[0])
    _, yt = lax.scan(step, (z, z, z, z), xt)
    return yt if x.ndim == 1 else jnp.moveaxis(yt, 0, -2)


class JaxBiquad:
    """A biquad layer with trainable coefficients, in the layer pattern of ``core.deep.neural``.

    ``b`` and ``a`` are ``Param`` objects (``b`` and ``a`` of the cookbook divided by ``a[0]``).
    ``forward`` filters along the time axis (see :func:`jax_biquad`) and remembers what ``backward``
    needs; ``backward(g)`` returns the gradient of the input and adds to ``b.grad`` and ``a.grad``.
    ``a[0]`` starts at 1 and has a gradient, the one through the division by ``a[0]``; an optimiser
    can move it away from 1 and the filter then normalises by it, so what matters is the ratios.

    Stability is not checked, because the coefficients change while training: unlike ``Biquad``,
    a pole outside the unit circle is not refused.

    Args:
        b: three numerator coefficients.
        a: three denominator coefficients, ``a[0]`` not zero.

    Raises:
        ValueError: if a coefficient list has not three entries or ``a[0]`` is zero.

    Example:
        >>> b, a = jax_lowpass(1000.0, 8000.0)
        >>> f = JaxBiquad(b, a)
        >>> f.forward(jnp.ones(4)).shape
        (4,)
    """

    def __init__(self, b: Any, a: Any) -> None:
        b, a = jnp.asarray(b), jnp.asarray(a)
        if b.shape != (3,) or a.shape != (3,):
            raise ValueError("b and a must have three coefficients each")
        if float(a[0]) == 0.0:
            raise ValueError("a[0] must not be zero")
        self.b = Param(b / a[0])
        self.a = Param(a / a[0])
        self._vjp: Any = None

    def params(self) -> list[Param]:
        """The numerator and the denominator, in this order."""
        return [self.b, self.a]

    def __repr__(self) -> str:
        return "JaxBiquad(b, a)"

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Filter ``x``."""
        del training
        y, self._vjp = jax.vjp(jax_biquad, self.b.value, self.a.value, jnp.asarray(x))
        return y

    def backward(self, g: jax.Array) -> jax.Array:
        """Return the gradient of the input and add to ``b.grad`` and ``a.grad``."""
        if self._vjp is None:
            raise RuntimeError("call forward() before backward()")
        db, da, dx = self._vjp(jnp.asarray(g, dtype=self.b.value.dtype))
        self.b.grad = self.b.grad + db.astype(self.b.grad.dtype)
        self.a.grad = self.a.grad + da.astype(self.a.grad.dtype)
        return dx


@jax.jit
def _allpole_scan(a: jax.Array, x: jax.Array) -> jax.Array:
    """Eq. (1) of Yu et al. as a plain ``lax.scan``: ``a`` is ``(n, M)``, ``x`` is ``(n,)``."""
    m = a.shape[1]

    def step(hist, ax):
        an, xn = ax
        y = xn - jnp.dot(an, hist)
        return jnp.concatenate([y[None], hist[:-1]]), y

    _, y = lax.scan(step, jnp.zeros(m, a.dtype), (a, x))
    return y


@jax.custom_vjp
def _allpole(a: jax.Array, x: jax.Array) -> jax.Array:
    return _allpole_scan(a, x)


def _allpole_fwd(a, x):
    y = _allpole_scan(a, x)
    return y, (a, y)


def _allpole_bwd(res, g):
    a, y = res
    n, m = a.shape
    k = jnp.arange(m)
    # Eq. (7): the gradient of x is the same filter run backwards in time with a_hat_i(n) = a_i(n + i).
    a_pad = jnp.concatenate([a, jnp.zeros((m, m), a.dtype)])
    a_hat = a_pad[jnp.arange(n)[:, None] + k[None, :] + 1, k[None, :]]
    gx = _allpole_scan(a_hat[::-1], g[::-1])[::-1]
    # Eq. (8): dL/da_i(n) = - dL/dx(n) y(n - i).
    y_pad = jnp.concatenate([jnp.zeros(m, y.dtype), y])
    y_past = y_pad[jnp.arange(n)[:, None] + m - k[None, :] - 1]
    return -gx[:, None] * y_past, gx


_allpole.defvjp(_allpole_fwd, _allpole_bwd)


def jax_allpole_time_varying(a: Any, x: Any) -> jax.Array:
    """The all-pole filter ``y(n) = x(n) - sum_i a_i(n) y(n - i)`` with one set of coefficients per sample.

    Eq. (1) of Yu et al. (2024). The gradients are the ones of their Sec. 3: the gradient of ``x``
    is the same filter run backwards in time with ``a_hat_i(n) = a_i(n + i)`` (Eq. 7), and the
    gradient of ``a_i(n)`` is ``-dL/dx(n) y(n - i)`` (Eq. 8), so the backward pass needs one more
    filtering and a product, and does not unroll the recursion.

    Args:
        a: the coefficients ``a_1 .. a_M``: shape ``(n, M)`` for one set per sample, or ``(M,)``
            for the same set at every sample.
        x: the input, shape ``(n,)``.

    Returns:
        The output, shape ``(n,)``.

    Raises:
        ValueError: if ``x`` is not 1D or ``a`` is not ``(M,)`` or ``(len(x), M)`` with ``M >= 1``.

    Example:
        >>> jax_allpole_time_varying([-0.5], jnp.array([1.0, 0.0, 0.0, 0.0])).tolist()
        [1.0, 0.5, 0.25, 0.125]
    """
    a, x = jnp.asarray(a), jnp.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got shape {x.shape}")
    if a.ndim == 1:
        a = jnp.broadcast_to(a, (x.shape[0], a.shape[0]))
    if a.ndim != 2 or a.shape[1] < 1 or a.shape[0] != x.shape[0]:
        raise ValueError(f"a must be (M,) or ({x.shape[0]}, M), got shape {a.shape}")
    dt = jnp.result_type(a.dtype, x.dtype, jnp.float32)
    return _allpole(a.astype(dt), x.astype(dt))
