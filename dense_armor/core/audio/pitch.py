"""Fundamental frequency by the YIN method: the difference function and its normalisation.

Source: Strahl and Mueller (2025), "dYIN and dSWIPE: Differentiable Variants of Classical
Fundamental Frequency Estimators", IEEE/ACM TASLP 33, Sec. II-A, PDF pages 2 and 3, which
restates YIN (de Cheveigne and Kawahara 2002, not in the project folder):

* the difference function ``d(tau) = sum_{n=0}^{N-1} [x(n) - x(n + tau)]^2`` of a frame ``x``
  taken as zero outside ``[0, N - 1]`` (Eq. 1);
* the cumulative mean normalised difference function (CMNDF), ``d'(0) = 1`` and
  ``d'(tau) = d(tau) / ((1 / tau) sum_{j=1}^{tau} d(j))`` (Eq. 2), for the lags
  ``0 .. ceil(sr / f0_min)``;
* step 1: the smallest lag at which ``d'`` has a local minimum below an absolute threshold;
* step 2: parabolic interpolation around that lag, the minimum of the parabola is the period.

Step 3 of YIN (the second pass with the search range narrowed by the neighbouring frames) is
described in the paper in one sentence and is not implemented.
"""

import math

import numpy as np

from dense_armor.roles import Root


def cmndf(x, tau_max: int) -> np.ndarray:
    """The CMNDF ``d'(tau)`` of a frame for ``tau = 0 .. tau_max`` (Eq. 1 and 2).

    ``d`` is computed through the autocorrelation by FFT: with the frame zero outside
    ``[0, N - 1]``, ``d(tau) = E(0) + E(tau) - 2 r(tau)``, where ``E(tau)`` is the energy of
    ``x[tau:]`` and ``r(tau)`` the autocorrelation at lag ``tau``. A lag with ``d = 0`` up to
    that lag (silence) gets ``d' = 1``.

    Args:
        x: one frame, shape ``(N,)``, at least two samples.
        tau_max: the largest lag, at least 1.

    Raises:
        ValueError: if ``x`` is not 1D with two samples or ``tau_max < 1``.

    Example:
        >>> d = cmndf(np.sin(2 * np.pi * np.arange(64) / 16), 24)
        >>> int(np.argmin(d[2:]) + 2)
        16
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must be a 1D frame of at least two samples")
    if tau_max < 1:
        raise ValueError(f"tau_max must be >= 1, got {tau_max}")
    n = x.size
    taus = np.arange(tau_max + 1)
    tail = np.concatenate([np.cumsum((x * x)[::-1])[::-1], [0.0]])
    nf = 1 << (2 * n - 1).bit_length()
    spec = np.fft.rfft(x, nf)
    r = np.fft.irfft(spec * np.conj(spec), nf)[:n]
    inside = taus < n
    e_tau = np.where(inside, tail[np.minimum(taus, n)], 0.0)
    r_tau = np.where(inside, r[np.minimum(taus, n - 1)], 0.0)
    d = np.maximum(tail[0] + e_tau - 2.0 * r_tau, 0.0)
    out = np.ones(tau_max + 1)
    mean = np.cumsum(d[1:]) / np.arange(1, tau_max + 1)
    ok = mean > 0.0
    out[1:][ok] = d[1:][ok] / mean[ok]
    return out


def yin_f0(
    x,
    sr: float,
    f0_min: float,
    threshold: float,
    f0_max: float | None = None,
) -> float:
    """The fundamental frequency of one frame by YIN steps 1 and 2, in Hz, or ``nan``.

    Choice stated: the paper does not say what to return when no lag is below the threshold;
    the frame is then unvoiced and the result is ``nan``. The threshold has no default: the
    paper does not give a value.

    Args:
        x: one frame, shape ``(N,)``.
        sr: sample rate in Hz.
        f0_min: the lowest F0 searched, in Hz; the lags go up to ``ceil(sr / f0_min)``.
        threshold: absolute threshold on ``d'``, positive.
        f0_max: the highest F0 searched, in Hz; the lags start at ``floor(sr / f0_max)``
            (at least 1). ``None`` starts at 1.

    Raises:
        ValueError: if ``sr``, ``f0_min`` or ``threshold`` is not positive, or ``f0_max`` is
            not above ``f0_min``.

    Example:
        >>> n = np.arange(1024)
        >>> round(yin_f0(np.sin(2 * np.pi * 440 * n / 8000), 8000, 100.0, 0.1), -1)
        440.0
    """
    if sr <= 0 or f0_min <= 0 or threshold <= 0:
        raise ValueError("sr, f0_min and threshold must be positive")
    if f0_max is not None and f0_max <= f0_min:
        raise ValueError("f0_max must be above f0_min")
    tau_max = math.ceil(sr / f0_min)
    tau_min = 1 if f0_max is None else max(1, math.floor(sr / f0_max))
    d = cmndf(x, tau_max)
    t = np.arange(tau_min, tau_max)
    hit = (d[t] < threshold) & (d[t] < d[t - 1]) & (d[t] <= d[t + 1])
    if not hit.any():
        return float("nan")
    tau = int(t[np.argmax(hit)])
    a, b, c = d[tau - 1], d[tau], d[tau + 1]
    den = a - 2.0 * b + c
    delta = 0.0 if den == 0.0 else float(np.clip(0.5 * (a - c) / den, -1.0, 1.0))
    return float(sr / (tau + delta))


class YIN(Root):
    """YIN as a primitive: ``update_many(frame)`` returns the F0 of that frame.

    Each call is one frame; nothing is kept between frames except the last result ``f0_``,
    so a block of sound given by :class:`AudioInput` is one frame.

    Args:
        sr: sample rate in Hz.
        f0_min: the lowest F0 searched, in Hz.
        threshold: absolute threshold on ``d'``.
        f0_max: the highest F0 searched, in Hz, or ``None``.

    Example:
        >>> y = YIN(8000, 100.0, 0.1)
        >>> round(float(y.update_many(np.sin(2 * np.pi * 200 * np.arange(1024) / 8000))), 0)
        200.0
    """

    budget_s: float = 4e-3
    memory_class: str = "O(1)"

    def __init__(
        self,
        sr: float,
        f0_min: float,
        threshold: float,
        f0_max: float | None = None,
    ) -> None:
        if sr <= 0 or f0_min <= 0 or threshold <= 0:
            raise ValueError("sr, f0_min and threshold must be positive")
        if f0_max is not None and f0_max <= f0_min:
            raise ValueError("f0_max must be above f0_min")
        self.sr = sr
        self.f0_min = f0_min
        self.threshold = threshold
        self.f0_max = f0_max
        self.f0_ = float("nan")

    def update_many(self, values) -> float:
        """The F0 of the frame ``values`` in Hz, ``nan`` if it is unvoiced."""
        self.f0_ = yin_f0(values, self.sr, self.f0_min, self.threshold, self.f0_max)
        return self.f0_
