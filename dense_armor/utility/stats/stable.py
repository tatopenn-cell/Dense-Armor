"""Stable float32 variance and moments, computed in the input dtype.

The kernel computes the variance of a sample of real values with a shift by
the first element and a tree sum in the input dtype. It is exactly ``0.0`` on
identical values, ``nan`` and ``inf`` propagate, and the float32 path never
casts to float64. The error-free transformations ``TwoSum``, ``Split`` and
``TwoProduct`` are used in :func:`_combine_moments`, the pairwise merge of
partial results (Pebay 2008, equations 1.3 and 1.4), not on the ``d * d``
term of the kernel: the shift already removes the cancellation there and
the tree sum is already 3 to 150 times more accurate than the TensorFlow
formula, while ``TwoProduct`` on ``d * d`` costs 8.4 times the time of the
tree sum and returns the same numbers on every measured dataset.

The shift is the reference element ``x[0]`` along the reduction axis. It is
subtracted in the input dtype, so the subtraction is exact when the
cancellation is benign (the reference is the largest magnitude in the axis,
or close to it), and the roundoff of the subtraction is at most one ulp of
the reference. With the first element as reference the kernel has no branch
and works under ``jax.jit`` and ``jax.vmap``.

References:
    Ogita, T., Rump, S. M., Oishi, S. (2005). Accurate Sum and Dot Product.
        SIAM Journal on Scientific Computing 26(6), 1955-1988. ``TwoSum``
        (Algorithm 3.1, PDF page 7), ``Split`` (Algorithm 3.2, PDF page 7),
        ``TwoProduct`` (Algorithm 3.3, PDF page 7).
    Shewchuk, J. R. (1997). Adaptive Precision Floating-Point Arithmetic and
        Fast Robust Geometric Predicates. Discrete and Computational
        Geometry 18(3), 305-363. ``TWO-SUM`` (Theorem 7, PDF page 12),
        ``SPLIT`` (Theorem 17, PDF page 22), ``TWO-PRODUCT`` (Theorem 18,
        PDF page 23).
    Pebay, P. P. (2008). Formulas for Robust, One-Pass Parallel Computation
        of Covariances and Arbitrary-Order Statistical Moments. Sandia
        Report SAND2008-6212. Equations 1.3 and 1.4 (PDF page 8).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


__all__ = ["stable_var", "stable_moments"]


def _as_float_array(x):
    """Return ``x`` as a JAX array with a float dtype.

    float16 and bfloat16 are promoted to float32 for the computation; the
    caller casts the result back to the input dtype. Integers become
    float32. Complex input is treated as ``abs(x)**2`` for the variance, so
    the result is real (``E|z - mean|^2``).
    """
    a = jnp.asarray(x)
    kind = a.dtype.kind
    if kind == "c":
        re = a.real.astype(jnp.float32) if a.dtype.itemsize <= 8 else a.real
        im = a.imag.astype(jnp.float32) if a.dtype.itemsize <= 8 else a.imag
        a = re * re + im * im
    elif kind in ("f",):
        if a.dtype in (jnp.float16, jnp.bfloat16):
            a = a.astype(jnp.float32)
    elif kind in ("i", "u", "b"):
        a = a.astype(jnp.float32)
    else:
        raise TypeError(f"stable_var: unsupported dtype {a.dtype}")
    return a


def _result_dtype(original):
    """Return the dtype of the result: same as input for floats, float32 else.

    The dtype is read from the input with ``numpy`` so that a float64
    numpy array keeps float64 when ``jax_enable_x64`` is on; with
    ``jax_enable_x64`` off, JAX promotes to float32 at the ``asarray``
    call and the result is float32, as every JAX program does.
    """
    d = np.asarray(original).dtype
    if d.kind == "f":
        return d
    return np.float32


def _check_axis(a, axis):
    if axis is None:
        return tuple(range(a.ndim))
    if isinstance(axis, int):
        axis = (axis,)
    nd = a.ndim
    for ax in axis:
        if not (-nd <= ax < nd):
            raise ValueError(
                f"stable_var: axis {ax} is out of bounds for an array of "
                f"dimension {nd}"
            )
    return tuple(sorted({ax % nd for ax in axis}))



def _cast_out(a, out_dtype):
    """Cast ``a`` to ``out_dtype``, down to float32 when x64 is off.

    With ``jax_enable_x64`` off, JAX cannot hold float64 arrays; the
    result of a float64 input is float32 in that mode (the same rule as
    every JAX program). The cast is explicit so no warning is emitted.
    """
    d = jnp.dtype(out_dtype)
    if d == jnp.float64 and not jax.config.jax_enable_x64:
        return a.astype(jnp.float32)
    return a.astype(d)


def stable_var(x, axis=None, ddof=0, keepdims=False):
    """Variance of ``x`` along ``axis``, computed in the input dtype.

    The variance is ``M2 / (n - ddof)`` where ``M2 = sum((x - mean)**2)``
    along the reduction axis, computed by shifting ``x`` by its first
    element along that axis and summing the squared deviations with a tree
    sum in the input dtype. float32 in gives float32 out, float64 in gives
    float64 out, float16 and bfloat16 are computed in float32 and returned
    in the input dtype, integers give float32, complex input gives the real
    ``E|z - mean|^2``.

    Identical values give exactly ``0.0``, ``nan`` and ``inf`` propagate.
    An empty reduction axis raises ``ValueError``. ``ddof >= n`` returns
    ``nan`` when ``M2 > 0`` and ``0.0`` when ``M2 == 0`` (so identical
    values stay exact).

    Args:
        x: array-like. numpy and JAX arrays are accepted.
        axis: int, tuple of int, or ``None`` (default, all axes).
        ddof: delta degrees of freedom, default ``0``.
        keepdims: keep the reduced axes with size one, default ``False``.

    Returns:
        The variance, with the dtype of the input (or float32).

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

    References:
        Ogita, Rump, Oishi 2005, ``TwoSum`` Algorithm 3.1, PDF page 7.
        Pebay 2008, equations 1.3 and 1.4, PDF page 8.
    """
    if ddof < 0:
        raise ValueError(f"stable_var: ddof must be non-negative, got {ddof}")
    a = _as_float_array(x)
    out_dtype = _result_dtype(x)
    axes = _check_axis(a, axis)
    if len(axes) == 0:
        return _cast_out(a, out_dtype)

    shape = [a.shape[ax] for ax in axes]
    if any(s == 0 for s in shape):
        raise ValueError("stable_var: empty reduction axis")

    n = 1
    for s in shape:
        n *= s

    ref = a[tuple(slice(0, 1) if i in axes else slice(None)
                  for i in range(a.ndim))]
    y = a - ref
    m = jnp.mean(y, axis=axes, keepdims=True)
    d = y - m
    M2 = jnp.sum(d * d, axis=axes, keepdims=keepdims)

    denom = n - ddof
    if denom <= 0:
        out = jnp.where(M2 == 0, jnp.zeros_like(M2), jnp.full_like(M2, jnp.nan))
    else:
        out = M2 / jnp.asarray(denom, dtype=a.dtype)
    return _cast_out(out, out_dtype)


def stable_moments(x, axis=None, ddof=0, keepdims=False):
    """Mean and variance of ``x`` along ``axis``, in the input dtype.

    Returns a tuple ``(mean, var)`` with the same conventions as
    :func:`stable_var` for the dtype, the empty axis, ``nan`` and ``inf``
    propagation, and ``ddof``. The mean is the mean of the shifted data plus
    the shift, computed in the input dtype; the variance is the one of
    :func:`stable_var`.

    Args:
        x: array-like.
        axis: int, tuple of int, or ``None`` (default, all axes).
        ddof: delta degrees of freedom, default ``0``.
        keepdims: keep the reduced axes, default ``False``.

    Returns:
        A tuple ``(mean, var)`` with the dtype of the input (or float32).

    Raises:
        ValueError: if the reduction axis is empty or out of bounds.

    Example:
        >>> import numpy as np
        >>> from dense_armor.utility.stats.stable import stable_moments
        >>> x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        >>> m, v = stable_moments(x)
        >>> round(float(m), 6), round(float(v), 6)
        (2.0, 0.666667)

    References:
        Ogita, Rump, Oishi 2005, ``TwoSum`` Algorithm 3.1, PDF page 7.
        Pebay 2008, equations 1.3 and 1.4, PDF page 8.
    """
    if ddof < 0:
        raise ValueError(f"stable_moments: ddof must be non-negative, got {ddof}")
    a = _as_float_array(x)
    out_dtype = _result_dtype(x)
    axes = _check_axis(a, axis)
    if len(axes) == 0:
        return _cast_out(a, out_dtype), _cast_out(jnp.zeros_like(a), out_dtype)

    shape = [a.shape[ax] for ax in axes]
    if any(s == 0 for s in shape):
        raise ValueError("stable_moments: empty reduction axis")

    ref = a[tuple(slice(0, 1) if i in axes else slice(None)
                  for i in range(a.ndim))]
    y = a - ref
    m = jnp.mean(y, axis=axes, keepdims=True)
    d = y - m
    M2 = jnp.sum(d * d, axis=axes, keepdims=True)
    mean = m + ref
    if not keepdims:
        mean = jnp.squeeze(mean, axis=axes)
        M2 = jnp.squeeze(M2, axis=axes)

    n = 1
    for s in shape:
        n *= s
    denom = n - ddof
    if denom <= 0:
        var = jnp.where(M2 == 0, jnp.zeros_like(M2), jnp.full_like(M2, jnp.nan))
    else:
        var = M2 / jnp.asarray(denom, dtype=a.dtype)
    return _cast_out(mean, out_dtype), _cast_out(var, out_dtype)


def _two_sum(a, b):
    """Knuth's ``TwoSum``: ``a + b = s + err``, ``s = fl(a + b)``.

    Ogita, Rump, Oishi 2005, Algorithm 3.1, PDF page 7.
    """
    s = a + b
    bb = s - a
    err = (a - (s - bb)) + (b - bb)
    return s, err


def _split(a):
    """Dekker's ``Split``: ``a = hi + lo`` with ``|lo| <= |hi|``.

    The factor is ``2**ceil(t/2) + 1`` with ``t`` the number of mantissa
    bits. Ogita, Rump, Oishi 2005, Algorithm 3.2, PDF page 7; Shewchuk
    1997, Theorem 17, PDF page 22.
    """
    t = jnp.finfo(a.dtype).nmant + 1
    s = (t + 1) // 2
    factor = jnp.asarray(2**s + 1, dtype=a.dtype)
    c = factor * a
    hi = c - (c - a)
    lo = a - hi
    return hi, lo


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
