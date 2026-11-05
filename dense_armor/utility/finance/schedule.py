"""Schedules that split a parent order into slices over time."""

import numpy as np


def twap_schedule(total: float, n: int) -> np.ndarray:
    """Split ``total`` into ``n`` equal slices that sum to ``total``.

    This is the time weighted average price schedule: the parent order is
    sold at constant speed, ``total / n`` per bucket, which is the optimal
    static strategy when the traded volume is constant (Remark 3, PDF
    page 5 of Kato 2014). The last slice absorbs the floating point
    residue, so the array sums to ``total``.

    Args:
        total: the size of the parent order, in shares; must be ``>= 0``.
        n: the number of slices; must be ``>= 1``.

    Returns:
        A float64 array of shape ``(n,)`` whose sum is ``total``.

    Raises:
        ValueError: if ``n < 1`` or ``total < 0``.

    Examples:
        >>> twap_schedule(3.0, 3)
        array([1., 1., 1.])
        >>> float(twap_schedule(1.0, 4).sum())
        1.0

    References:
        Kato, T. (2014). An optimal execution problem with market impact
        and stochastic volume. Remark 3, PDF page 5: when the volume is a
        constant, the optimal strategy sells at constant speed.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got n={n}")
    if total < 0:
        raise ValueError(f"total must be >= 0, got total={total}")
    total = float(total)
    slice_size = total / n
    slices: np.ndarray = np.full(n, slice_size, dtype=np.float64)
    slices[-1] = total - slice_size * (n - 1)
    return slices


def vwap_schedule(total: float, history: np.ndarray) -> np.ndarray:
    """Split ``total`` across buckets proportionally to a harmonic mean.

    The weight of bucket ``j`` is ``u[j] = 1 / mean_i(1 / history[i, j])``,
    the harmonic mean of the volume traded in that bucket over the past
    sessions. The schedule is ``total * u / u.sum()``, so the slices sum to
    ``total`` (the sell-off condition, equation (2.3), PDF page 2 of Kato
    2014). The harmonic mean is the expectation of the volume in Theorem 2,
    PDF page 5, where ``u_t = 1 / E[1 / v_t]`` and ``E[1 / v_t]`` is read as
    a harmonic mean. When the volume is constant across buckets, this
    schedule equals :func:`twap_schedule` (Remark 3, PDF page 5).

    The model of the theorem assumes a temporary market impact of the form
    ``kappa_tilde * zeta / v`` (equation (3.2), PDF page 3), a static
    strategy (``A_stat``, PDF page 4), a risk-neutral trader (PDF page 1
    and page 5), and an unaffected price independent of the volume process
    (PDF page 3).

    Args:
        total: the size of the parent order, in shares; must be ``>= 0``.
        history: an array of shape ``(n_sessions, n_buckets)`` with the
            volume traded in each bucket in each past session; every value
            must be finite and strictly positive.

    Returns:
        A float64 array of shape ``(n_buckets,)`` whose sum is ``total``.

    Raises:
        ValueError: if ``total < 0``, if ``history`` is not 2-D, if it has
            no bucket, or if a volume is not finite and strictly positive.

    Examples:
        >>> h = np.array([[1.0, 2.0], [4.0, 2.0]])
        >>> [round(float(x), 4) for x in vwap_schedule(3.6, h)]
        [1.6, 2.0]
        >>> [round(float(x), 4) for x in vwap_schedule(3.6, np.ones((2, 2)))]
        [1.8, 1.8]

    References:
        Kato, T. (2014). An optimal execution problem with market impact
        and stochastic volume. Sell-off condition equation (2.3), PDF
        page 2; temporary impact equation (3.2), PDF page 3; static
        strategy ``A_stat``, PDF page 4; Theorem 2 and Remark 3, PDF
        page 5.
    """
    if total < 0:
        raise ValueError(f"total must be >= 0, got total={total}")
    history = np.asarray(history, dtype=np.float64)
    if history.ndim != 2:
        raise ValueError(
            f"history must be 2-D (n_sessions, n_buckets), got ndim={history.ndim}"
        )
    if history.shape[1] < 1:
        raise ValueError(f"history must have at least one bucket, got {history.shape}")
    if not np.all(np.isfinite(history)):
        raise ValueError("history has a volume that is not finite")
    if not np.all(history > 0):
        raise ValueError("history has a volume that is not strictly positive")
    inverse = 1.0 / history
    u = 1.0 / inverse.mean(axis=0)
    total = float(total)
    return total * u / u.sum()
