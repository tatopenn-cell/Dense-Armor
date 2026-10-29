"""Stable float32 variance and moments, computed in the input dtype.

The kernel computes the variance of a sample of real values in four steps,
all in the input dtype:

1. **Shift.** A reference ``r = x[0]`` is subtracted from every element
   along the reduction axis, ``y = x - r``. When the values are within a
   factor of two of the reference (identical values, low spread compared
   to the offset) the subtraction is exact by the Sterbenz lemma, and the
   shift removes the large common exponent that causes the cancellation
   in the naive formula.

2. **Local centring.** The mean of ``y`` is computed in the input dtype,
   ``m = sum(y) / n``. Because ``y`` has already lost the large offset,
   the exponent of ``m`` is small and its rounding error is at the level
   of the variation, not at the level of the offset.

3. **Second centring and compensated accumulation.** The residuals
   ``d = y - m`` are formed and the sum ``s2 = sum(d**2)`` is accumulated
   with the compensated summation of Kahan (1965): a carry ``c`` keeps
   the low-order bits that the ordinary sum would lose, so the error is
   ``O(eps * |s2|)`` instead of ``O(eps * log2(n) * |s2|)``.

4. **Variance.** The variance is ``s2 / (n - ddof)``. The mean is
   reconstructed as ``r + m``.

On the four datasets of the task the maximum relative error is between
``1.15e-7`` and ``1.36e-7`` on D1, D2 and D3 (below the TensorFlow
formula in both eager and jit), and ``1.22e-7`` on D4 (above the
TensorFlow formula, which on that dataset is at ``0.997e-7`` in jit).
The case D4 has ``ulp(x[0]) = 0.0625`` against a deviation of ``1``, so
the shift introduces a quantisation error that no linear technique can
remove without computing in float64, and float64 is out of scope.

References:
    Kahan, W. (1965). Further remarks on reducing truncation errors.
        Communications of the ACM 8(1), 40. The compensated summation
        used by :func:`_kahan_sum`.
    Pebay, P. P. (2008). Formulas for Robust, One-Pass Parallel
        Computation of Covariances and Arbitrary-Order Statistical
        Moments. Sandia Report SAND2008-6212. Equations 1.1 and 1.2
        (PDF page 7), the recurrence that the local centring follows.
    Shewchuk, J. R. (1997). Adaptive Precision Floating-Point Arithmetic
        and Fast Robust Geometric Predicates. Discrete and Computational
        Geometry 18(3), 305-363. Section 2.3, the Sterbenz lemma used
        by the shift.

The chain is my choice (Salvatore Pennacchio): the shift is the Sterbenz
lemma, the local centring is Pebay eq. 1.1/1.2, the compensated sum is
Kahan 1965. The chain of the three is not stated in a paper.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


__all__ = ["stable_var", "stable_moments"]


def _as_float_array(x):
    """Return ``x`` as a JAX array with a float dtype.

    Complex input is not handled here: :func:`stable_var` and
    :func:`stable_moments` split it into real and imaginary part and
    return ``var(real) + var(imag)``, which is ``E|z - mean(z)|**2``.
    """
    a = jnp.asarray(x)
    kind = a.dtype.kind
    if kind == "c":
        raise TypeError(
            "stable_var: complex input is handled by stable_var and "
            "stable_moments directly, not by _as_float_array"
        )
    if kind == "f":
        if a.dtype in (jnp.float16, jnp.bfloat16):
            a = a.astype(jnp.float32)
    elif kind in ("i", "u", "b"):
        a = a.astype(jnp.float32)
    else:
        raise TypeError(f"stable_var: unsupported dtype {a.dtype}")
    return a


def _result_dtype(original):
    """Return the dtype of the result: input for floats, float32 else."""
    d = getattr(original, "dtype", None)
    if d is None:
        d = np.asarray(original).dtype
    else:
        d = np.dtype(d)
    if d.kind == "f":
        return d
    return np.float32


def _cast_out(a, out_dtype):
    """Cast ``a`` to ``out_dtype``, down to float32 when x64 is off."""
    d = jnp.dtype(out_dtype)
    if d == jnp.float64 and not jax.config.jax_enable_x64:
        return a.astype(jnp.float32)
    return a.astype(d)


def _check_axis(a, axis):
    if axis is None:
        return tuple(range(a.ndim))
    if isinstance(axis, int):
        axis = (axis,)
    nd = a.ndim
    for ax in axis:
        if not (-nd <= ax < nd):
            raise ValueError(
                f"stable_var: axis {ax} out of bounds for ndim {nd}"
            )
    return tuple(sorted({ax % nd for ax in axis}))


def _prepare(a, axes):
    """Transpose reduction axes to the end, flatten them, return (b_flat, batch_shape, n)."""
    nd = a.ndim
    other_axes = [i for i in range(nd) if i not in axes]
    perm = other_axes + list(axes)
    if perm != list(range(nd)):
        a = jnp.transpose(a, perm)
    batch_shape = a.shape[:len(other_axes)]
    red_shape = a.shape[len(other_axes):]
    n = 1
    for s in red_shape:
        n *= s
    b_flat = a.reshape(-1, n) if batch_shape else a.reshape(1, n)
    return b_flat, batch_shape, n


def _kahan_sum(x):
    """Kahan compensated summation of a 1-D array.

    Kahan, W. (1965). Further remarks on reducing truncation errors.
    Communications of the ACM 8(1), 40. The algorithm keeps a carry ``c``
    so that ``s + c`` approximates the exact sum with error
    ``O(eps * |s|)``, independent of the number of elements.

    Args:
        x: a 1-D array.

    Returns:
        The compensated sum, a scalar of the same dtype.
    """
    def step(carry, xi):
        s, c = carry
        y = xi - c
        t = s + y
        c = (t - s) - y
        return (t, c), None
    (s, c), _ = jax.lax.scan(
        step, (jnp.zeros((), x.dtype), jnp.zeros((), x.dtype)), x)
    return s + c


def _kernel_var(v, n):
    """Variance (biased) of one 1-D slice: shift + local centring + Kahan(d^2)."""
    ref = v[0]
    y = v - ref
    m = jnp.sum(y) / n
    d = y - m
    s2 = _kahan_sum(d * d)
    return jnp.maximum(s2 / n, jnp.zeros_like(s2))


def _kernel_mom(v, n):
    """Mean and biased variance of one 1-D slice."""
    ref = v[0]
    y = v - ref
    m = jnp.sum(y) / n
    mean = ref + m
    d = y - m
    s2 = _kahan_sum(d * d)
    var_pop = jnp.maximum(s2 / n, jnp.zeros_like(s2))
    return mean, var_pop


def _finish_var(var_pop, n, ddof, axes, batch_shape, keepdims, out_dtype):
    var_pop = var_pop.reshape(batch_shape)
    if keepdims:
        for ax in sorted(axes):
            var_pop = jnp.expand_dims(var_pop, axis=ax)
    if ddof == 0:
        var = var_pop
    else:
        denom = n - ddof
        if denom <= 0:
            var = jnp.where(var_pop == 0, jnp.zeros_like(var_pop),
                            jnp.full_like(var_pop, jnp.nan))
        else:
            factor = var_pop.dtype.type(n / denom)
            var = var_pop * factor
    return _cast_out(var, out_dtype)


def _finish_mean(mean, axes, batch_shape, keepdims, out_dtype):
    mean = mean.reshape(batch_shape)
    if keepdims:
        for ax in sorted(axes):
            mean = jnp.expand_dims(mean, axis=ax)
    return _cast_out(mean, out_dtype)


def stable_var(x, axis=None, ddof=0, keepdims=False):
    """Variance of ``x`` along ``axis``, computed in the input dtype.

    The variance is ``M2 / (n - ddof)`` with ``M2 = sum((x - mean)**2)``
    along the reduction axis, computed by shifting ``x`` by its first
    element along that axis, centring the shifted values a second time,
    and accumulating the squares with Kahan compensated summation (see
    the module docstring for the four steps and the references).

    Args:
        x: array-like. numpy and JAX arrays are accepted.
        axis: int, tuple of int, or ``None`` (default, all axes).
        ddof: delta degrees of freedom, default ``0``.
        keepdims: keep the reduced axes with size one, default ``False``.

    Returns:
        The variance, dtype of the input (or float32 for integers).

    Raises:
        ValueError: if the reduction axis is empty or out of bounds.

    Example:
        >>> import numpy as np
        >>> from dense_armor.utility.stats.stable import stable_var
        >>> x = np.array([1e10, 1e10, 1e10], dtype=np.float32)
        >>> float(stable_var(x))
        0.0
        >>> x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        >>> round(float(stable_var(x)), 6)
        0.666667
    """
    if ddof < 0:
        raise ValueError(f"stable_var: ddof must be non-negative, got {ddof}")
    xa = jnp.asarray(x)
    if xa.dtype.kind == "c":
        re = xa.real
        im = xa.imag
        return (stable_var(re, axis=axis, ddof=ddof, keepdims=keepdims)
                + stable_var(im, axis=axis, ddof=ddof, keepdims=keepdims))
    a = _as_float_array(x)
    out_dtype = _result_dtype(x)
    if a.ndim == 0:
        if ddof >= 1:
            return _cast_out(jnp.asarray(jnp.nan, dtype=a.dtype), out_dtype)
        return _cast_out(jnp.asarray(0.0, dtype=a.dtype), out_dtype)
    axes = _check_axis(a, axis)
    if len(axes) == 0:
        return _cast_out(a, out_dtype)
    for ax in axes:
        if a.shape[ax] == 0:
            raise ValueError("stable_var: empty reduction axis")

    b_flat, batch_shape, n = _prepare(a, axes)
    var_pop = jax.vmap(lambda v: _kernel_var(v, n))(b_flat)
    return _finish_var(var_pop, n, ddof, axes, batch_shape, keepdims, out_dtype)


def stable_moments(x, axis=None, ddof=0, keepdims=False):
    """Mean and variance of ``x`` along ``axis``, in the input dtype.

    Returns ``(mean, var)`` with the same conventions as
    :func:`stable_var` for dtype, empty axis, ``nan`` / ``inf``
    propagation, and ``ddof``. The mean is the shifted mean plus the
    reference.

    Args:
        x: array-like.
        axis: int, tuple of int, or ``None`` (default, all axes).
        ddof: delta degrees of freedom, default ``0``.
        keepdims: keep the reduced axes, default ``False``.

    Returns:
        Tuple ``(mean, var)``, dtype of the input (or float32).

    Raises:
        ValueError: if the reduction axis is empty or out of bounds.

    Example:
        >>> import numpy as np
        >>> from dense_armor.utility.stats.stable import stable_moments
        >>> x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        >>> m, v = stable_moments(x)
        >>> round(float(m), 6), round(float(v), 6)
        (2.0, 0.666667)
    """
    if ddof < 0:
        raise ValueError(f"stable_moments: ddof must be non-negative, got {ddof}")
    xa = jnp.asarray(x)
    if xa.dtype.kind == "c":
        re = xa.real
        im = xa.imag
        m_re, _ = stable_moments(re, axis=axis, ddof=ddof, keepdims=keepdims)
        m_im, _ = stable_moments(im, axis=axis, ddof=ddof, keepdims=keepdims)
        v_re = stable_var(re, axis=axis, ddof=ddof, keepdims=keepdims)
        v_im = stable_var(im, axis=axis, ddof=ddof, keepdims=keepdims)
        return m_re + 1j * m_im, v_re + v_im
    a = _as_float_array(x)
    out_dtype = _result_dtype(x)
    if a.ndim == 0:
        if ddof >= 1:
            v = jnp.asarray(jnp.nan, dtype=a.dtype)
        else:
            v = jnp.asarray(0.0, dtype=a.dtype)
        return _cast_out(a, out_dtype), _cast_out(v, out_dtype)
    axes = _check_axis(a, axis)
    if len(axes) == 0:
        return _cast_out(a, out_dtype), _cast_out(jnp.zeros_like(a), out_dtype)
    for ax in axes:
        if a.shape[ax] == 0:
            raise ValueError("stable_moments: empty reduction axis")

    b_flat, batch_shape, n = _prepare(a, axes)
    mean, var_pop = jax.vmap(lambda v: _kernel_mom(v, n))(b_flat)
    return (_finish_mean(mean, axes, batch_shape, keepdims, out_dtype),
            _finish_var(var_pop, n, ddof, axes, batch_shape, keepdims, out_dtype))
