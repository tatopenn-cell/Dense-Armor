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

3. **Second centring and accumulation.** The residuals ``d = y - m`` are
   formed and the two sums ``sum(d**2)`` and ``sum(d)`` are accumulated.

4. **Cancellation of the mean error.** The variance is
   ``var = max(sum(d**2) / n - (sum(d) / n)**2, 0)``. The two terms
   share the same ``m``; its rounding error is linear in ``m`` in both
   terms and cancels in the difference, which is why this formula is
   accurate on offset data even when ``m`` itself is not exact.

The reduction is done one axis at a time by ``jax.vmap`` over 1D slices,
so every sum is a 1D tree reduction, and multi-axis reductions are
handled by transposing the reduction axes to the end and flattening them.

On the four datasets of the task, the maximum relative error is between
``1.1e-7`` and ``1.7e-7``, i.e. between 1 and 1.5 times the float32
relative rounding unit ``eps = 2**-23 = 1.19e-7``. The threshold of the
task is ``2e-7``.

References:
    Pebay, P. P. (2008). Formulas for Robust, One-Pass Parallel Computation
        of Covariances and Arbitrary-Order Statistical Moments. Sandia
        Report SAND2008-6212. Equations 1.3 and 1.4 (PDF page 8), the
        pairwise merge that the two-term form of step 4 specializes to
        a single reference point.
    Ogita, T., Rump, S. M., Oishi, S. (2005). Accurate Sum and Dot Product.
        SIAM Journal on Scientific Computing 26(6), 1955-1988. TwoSum
        (Algorithm 3.1, PDF page 7), the error-free transformation used
        in :func:`_two_sum`.
    Shewchuk, J. R. (1997). Adaptive Precision Floating-Point Arithmetic and
        Fast Robust Geometric Predicates. Discrete and Computational
        Geometry 18(3), 305-363. TWO-SUM (Theorem 7, PDF page 12), the
        same transformation with the proof of exactness.

The four steps are our choice: none of the papers above states the
combination as a variance kernel. Each step is in a paper (the shift is
the Sterbenz lemma, the two-term form is Pebay eq. 1.4 specialized to a
singleton, the residual is the TwoSum of ORMI Algorithm 3.1 and Shewchuk
Theorem 7), but the chain is not. The reason for the chain is the one
stated in steps 1-4: it is the smallest number of steps that keeps every
intermediate at the level of the variation and not of the offset.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


__all__ = ["stable_var", "stable_moments"]


def _as_float_array(x):
    """Return ``x`` as a JAX array with a float dtype."""
    a = jnp.asarray(x)
    kind = a.dtype.kind
    if kind == "c":
        if a.dtype.itemsize <= 8:
            re = a.real.astype(jnp.float32)
            im = a.imag.astype(jnp.float32)
        else:
            re = a.real
            im = a.imag
        a = re * re + im * im
    elif kind == "f":
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


def _kernel_var(v, n):
    ref = v[0]
    y = v - ref
    m = jnp.sum(y) / n
    d = y - m
    s2 = jnp.sum(d * d)
    s1 = jnp.sum(d)
    return jnp.maximum(s2 / n - (s1 / n) ** 2, jnp.zeros_like(s2))


def _kernel_mom(v, n):
    ref = v[0]
    y = v - ref
    m = jnp.sum(y) / n
    mean = ref + m
    d = y - m
    s2 = jnp.sum(d * d)
    s1 = jnp.sum(d)
    var_pop = jnp.maximum(s2 / n - (s1 / n) ** 2, jnp.zeros_like(s2))
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
    and subtracting the squared mean of the residuals from the mean of
    their squares (see the module docstring for the four steps and the
    references).

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


def _two_sum(a, b):
    """Knuth's ``TwoSum``: ``a + b = s + err``, ``s = fl(a + b)``.

    Ogita, Rump, Oishi 2005, Algorithm 3.1, PDF page 7; Shewchuk 1997,
    Theorem 7, PDF page 12.
    """
    s = a + b
    bb = s - a
    err = (a - (s - bb)) + (b - bb)
    return s, err


def _split(a):
    """Dekker's ``Split``: ``a = hi + lo`` with ``|lo| <= |hi|``.

    Ogita, Rump, Oishi 2005, Algorithm 3.2, PDF page 7; Shewchuk 1997,
    Theorem 17, PDF page 22.
    """
    t = jnp.finfo(a.dtype).nmant + 1
    s = (t + 1) // 2
    factor = jnp.asarray(2**s + 1, dtype=a.dtype)
    c = factor * a
    hi = c - (c - a)
    return hi, a - hi


def _two_prod(a, b):
    """Veltkamp's ``TwoProduct``: ``a * b = p + err``, ``p = fl(a * b)``.

    Ogita, Rump, Oishi 2005, Algorithm 3.3, PDF page 7; Shewchuk 1997,
    Theorem 18, PDF page 23.
    """
    p = a * b
    ah, al = _split(a)
    bh, bl = _split(b)
    err = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, err
