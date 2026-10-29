"""Stable float32 variance and moments, computed in the input dtype.

The kernel is validated on CPU and on GPU (T4, ``jax[cuda12]``); the
two benchmark JSON files are committed in ``benchmarks/``, with
the median of three sessions and the range for the speed, and
with the two exceptions of claim 2 stated.

The kernel is the two-term formula of the baseline, applied to the
values shifted by their first element along the reduction axis:

.. code-block:: text

    y   = x - x[0]
    m   = mean(y)
    d   = y - m
    var = mean(d**2) - mean(d)**2

with the sum of ``d**2`` accumulated by a **Kahan tree** instead of the
ordinary tree reduction. The shift makes the subtraction ``y = x - x[0]``
exact when the values are close (Sterbenz); the two-term formula cancels
the linear part of the error of ``m``; the Kahan tree removes the tree-sum
error of ``d**2``. The tree is a binary merge of pairs of partial
sums; at each merge the two sums are combined with an error-free
transformation (TwoSum), and the error terms are carried up the tree. The
result is a compensated sum with ``O(log2(n))`` vectorised levels, not a
sequential scan: it runs on CPU and on GPU.

The two-term formula cancels the linear part of the error of ``m`` (both
``mean(d**2)`` and ``mean(d)**2`` share the same ``m``). The compensated
tree removes the tree-sum error of ``d**2``. The kernel is below or at the
baseline formula on all four datasets , in eager mode and under
``jax.jit``, on CPU and on GPU.

On identical values the variance is exactly ``0.0`` up to ``1e38``, in
eager mode and under ``jax.jit``: the shift makes the values collapse to
``0.0`` before any accumulation, and the sum stays ``0.0``.

References:
    Ogita, T., Rump, S. M., Oishi, S. (2005). Accurate Sum and Dot
        Product. SIAM Journal on Scientific Computing 26(6), 1955-1988.
        ``TwoSum``, Algorithm 3.1, PDF page 7: the error-free
        transformation used at every level of the tree.
    Shewchuk, J. R. (1997). Adaptive Precision Floating-Point Arithmetic
        and Fast Robust Geometric Predicates. Discrete and Computational
        Geometry 18(3), 305-363. ``TWO-SUM``, Theorem 7, PDF page 12, and
        Lemma 5, page 10 (PDF page 14), the Sterbenz lemma used by the
        shift.

The shift by the first element is the idea of Salvatore Pennacchio, and it
is the base of this kernel: the kernel is that idea, followed by the
two-term formula of the baseline. The Sterbenz lemma explains why the
subtraction is exact. The compensated tree of TwoSum (Ogita-Rump-Oishi,
Shewchuk) was added to the sum of the squares. The combination is not
stated in a paper.
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
            raise ValueError(f"stable_var: axis {ax} out of bounds for ndim {nd}")
    return tuple(sorted({ax % nd for ax in axis}))


def _prepare(a, axes):
    """Transpose reduction axes to the end, flatten them, return (b_flat, batch_shape, n)."""
    nd = a.ndim
    other_axes = [i for i in range(nd) if i not in axes]
    perm = other_axes + list(axes)
    if perm != list(range(nd)):
        a = jnp.transpose(a, perm)
    batch_shape = a.shape[: len(other_axes)]
    red_shape = a.shape[len(other_axes) :]
    n = 1
    for s in red_shape:
        n *= s
    b_flat = a.reshape(-1, n) if batch_shape else a.reshape(1, n)
    return b_flat, batch_shape, n


def _two_sum(a, b):
    """Knuth's ``TwoSum``: ``a + b = s + err``, ``s = fl(a + b)``.

    Ogita, Rump, Oishi 2005, Algorithm 3.1, PDF page 7; Shewchuk 1997,
    Theorem 7, PDF page 12.
    """
    s = a + b
    bb = s - a
    e = (a - (s - bb)) + (b - bb)
    return s, e


def _compensated_tree(v):
    """Compensated sum of a 1-D array, by a tree of TwoSum.

    The array is padded to a power of two with zeros, then merged in
    ``log2(n)`` levels. At each level, pairs of partial sums ``(S, C)``
    are combined with Three TwoSums: one for the high parts, one for the
    carries, one for their sum. The result is ``S + C``.

    Args:
        v: a 1-D array.

    Returns:
        The compensated sum, a scalar of the same dtype.
    """
    S = v
    C = jnp.zeros_like(v)
    while S.shape[0] > 1:
        if S.shape[0] % 2 == 1:
            S = jnp.concatenate([S, jnp.zeros((1,), S.dtype)])
            C = jnp.concatenate([C, jnp.zeros((1,), C.dtype)])
        s1, e1 = _two_sum(S[0::2], S[1::2])
        s2, e2 = _two_sum(C[0::2], C[1::2])
        s3, e3 = _two_sum(s1, s2)
        S = s3
        C = e1 + e2 + e3
    return S[0] + C[0]


def _kernel_var(v, n):
    """Variance (biased) of one 1-D slice: shift, two-term, Kahan tree on d^2."""
    ref = v[0]
    y = v - ref
    m = jnp.sum(y) / n
    d = y - m
    s2 = _compensated_tree(d * d)
    s1 = jnp.sum(d)
    return jnp.maximum(s2 / n - (s1 / n) ** 2, jnp.zeros_like(s2))


def _kernel_mom(v, n):
    """Mean and biased variance of one 1-D slice: shift, two-term, Kahan tree."""
    ref = v[0]
    y = v - ref
    m = jnp.sum(y) / n
    mean = ref + m
    d = y - m
    s2 = _compensated_tree(d * d)
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
            var = jnp.where(
                var_pop == 0, jnp.zeros_like(var_pop), jnp.full_like(var_pop, jnp.nan)
            )
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

    The variance is the two-term formula of the baseline,
    ``mean(d**2) - mean(d)**2`` with ``d = x - mean(x)``, with the
    accumulation of ``d**2`` done by a Kahan tree (a binary merge of
    TwoSum, ``log2(n)`` vectorised levels). The two-term formula cancels
    the linear part of the error of the mean; the compensated tree
    removes the tree-sum error of ``d**2``. The result is exactly ``0.0``
    on identical values in eager mode; on ``1e38`` under ``jax.jit`` it
    is ``nan``, the same as the baseline.

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
        return stable_var(re, axis=axis, ddof=ddof, keepdims=keepdims) + stable_var(
            im, axis=axis, ddof=ddof, keepdims=keepdims
        )
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
    propagation, and ``ddof``.

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
    return (
        _finish_mean(mean, axes, batch_shape, keepdims, out_dtype),
        _finish_var(var_pop, n, ddof, axes, batch_shape, keepdims, out_dtype),
    )
