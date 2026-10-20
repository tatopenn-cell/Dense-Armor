"""Drift against microstructure noise: structural change or the noise of the tick?

A tick price is noisy: bid-ask bounce and rounding make the tick return
noisier than the return over a moderate interval. A CUSUM on the raw
tick returns fires on that noise, and the alarm is not a structural
change. This pipeline compares the same detector on two statistics: the
raw tick returns, and a statistic that reduces the microstructure
noise.

Reference: Maneesoonthorn, Martin and Forbes (2017). *High-Frequency
Jump Tests: Which Test Should We Use?* arXiv:1708.09520. The two
remedies for microstructure noise are in the paper: prices averaged
over a small window (section 2.5, page 5, the LM12 test) and a moderate
sampling frequency of 30 seconds (section 5, page 10, guideline 3).
The statistic used here is the return over a small window of prices
(LM12, page 5), computed as the difference between the means of two
consecutive windows of ``k`` prices. The paper uses the LM12 test
statistic itself (a standardized return with a local volatility
estimate); this pipeline takes the return itself, which is the scalars
a streaming drift detector can take. The choice is declared in the
docstring of :class:`DriftVsNoisePipeline`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dense_armor.utility.drift.detector import CUSUMDriftDetector


def synthetic_tick(
    n: int = 5000,
    seed: int = 0,
    noise_std: float = 0.002,
    drift: float = 0.0,
    vol: float = 0.001,
    shift_at: int | None = None,
    drift_after: float = 0.0,
    vol_after: float | None = None,
    noise_kind: str = "gaussian",
    noise_df: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """A synthetic log price at tick frequency, with additive noise.

    The log price is a random walk with drift ``drift`` and volatility
    ``vol``; the tick price is the log price plus additive noise of
    standard deviation ``noise_std``. If ``shift_at`` is given, from
    that sample on the drift is ``drift_after`` and the volatility is
    ``vol_after`` (or ``vol`` if ``None``).

    Args:
        n: number of ticks.
        seed: RNG seed.
        noise_std: standard deviation of the additive noise on the
            tick price.
        drift: drift of the log price before ``shift_at``.
        vol: volatility of the log price before ``shift_at``.
        shift_at: index of the structural shift, or ``None`` for none.
        drift_after: drift from ``shift_at`` on.
        vol_after: volatility from ``shift_at`` on, or ``None`` to keep
            ``vol``.
        noise_kind: ``"gaussian"`` (default) or ``"student"``.
            The Student-t noise is the heavy-tailed model a tick
            price shows when a quote is stale or a trade prints far
            from the touch. It is this implementation's choice.
        noise_df: degrees of freedom of the Student-t noise. Default
            3.0. It is this implementation's choice.

    Returns:
        ``(true_price, tick_price)``: two arrays of length ``n``. The
        first is the log price without the noise, the second is what a
        reader sees.
    """
    rng = np.random.default_rng(seed)
    v = float(vol)
    d = float(drift)
    true = np.empty(n, dtype=float)
    true[0] = 0.0
    for i in range(1, n):
        if shift_at is not None and i >= shift_at:
            v = float(vol_after) if vol_after is not None else float(vol)
            d = float(drift_after)
        true[i] = true[i - 1] + d + v * rng.normal()
    if noise_kind == "gaussian":
        z = rng.normal(size=n)
    elif noise_kind == "student":
        z = rng.standard_t(noise_df, size=n) * np.sqrt(
            (noise_df - 2.0) / noise_df)
    else:
        raise ValueError(
            f"noise_kind must be 'gaussian' or 'student', "
            f"got {noise_kind!r}")
    tick = true + noise_std * z
    return true, tick


def raw_returns(tick_price: np.ndarray) -> np.ndarray:
    """Tick returns: the difference of consecutive tick prices."""
    x = np.asarray(tick_price, dtype=float)
    return np.diff(x)


def preaveraged_returns(tick_price: np.ndarray, window: int = 10) -> np.ndarray:
    """Returns between the means of two consecutive windows of prices.

    With a window of ``k`` prices, the mean of ``[i, i+k)`` and the mean
    of ``[i+k, i+2k)`` are computed, and the return is the difference of
    the two means. The microstructure noise averages out in the mean
    (the paper calls this "prices averaged over a small window", section
    2.5, page 5), and the return over ``k`` ticks is less noisy than the
    tick return.

    Args:
        tick_price: the tick log price.
        window: number of ticks in each window. Default 10.

    Returns:
        A 1-D array of length ``len(tick_price) // (2 * window)``.
    """
    x = np.asarray(tick_price, dtype=float)
    m = len(x) // (2 * window)
    out = np.empty(m, dtype=float)
    for i in range(m):
        a = x[i * 2 * window:(i * 2 + 1) * window].mean()
        b = x[(i * 2 + 1) * window:(i * 2 + 2) * window].mean()
        out[i] = b - a
    return out


@dataclass
class DriftVsNoiseResult:
    """Result of one run of the pipeline.

    Attributes:
        n_samples: number of returns fed to the raw detector.
        n_episodes_raw: alarm episodes on the raw statistic.
        n_episodes_robust: alarm episodes on the noise-robust statistic.
        delay_raw_samples: first episode after the shift, on the raw
            statistic, or ``None``. In samples of the raw detector.
        delay_robust_samples: first episode after the shift, on the
            noise-robust statistic, or ``None``. In samples of the
            robust detector (the pre-averaged returns).
        change_at: index of the shift, or ``None``.
        warmup: samples skipped by both detectors before the first read.
    """

    n_samples: int
    n_episodes_raw: int
    n_episodes_robust: int
    delay_raw_samples: int | None
    delay_robust_samples: int | None
    change_at: int | None
    warmup: int


class DriftVsNoisePipeline:
    """Run the same CUSUM on a raw and a noise-robust statistic.

    Args:
        window: number of ticks in each pre-averaging window. Default
            10. It is this implementation's choice; the paper uses
            window lengths that correspond to 5 seconds (section 5,
            page 10), and here the window is in ticks.
        warmup: samples skipped by both detectors before the first
            read. Default 20. It is this implementation's choice.
        cusum_kwargs: keyword arguments forwarded to
            :class:`CUSUMDriftDetector`. Default
            ``reference="fixed", k=0.5, h=20.0, reset_on_alarm=True``.

    Raises:
        ValueError: on a non-positive ``window`` or a negative
            ``warmup``.
    """

    def __init__(
        self,
        window: int = 10,
        warmup: int = 20,
        cusum_kwargs: dict | None = None,
    ) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if warmup < 0:
            raise ValueError(f"warmup must be >= 0, got {warmup}")
        self.window = int(window)
        self.warmup = int(warmup)
        self.cusum_kwargs = cusum_kwargs or {
            "reference": "fixed", "k": 0.5, "h": 20.0,
            "reset_on_alarm": True,
        }

    def run(
        self,
        tick_price: np.ndarray,
        change_at: int | None = None,
    ) -> DriftVsNoiseResult:
        """Run the pipeline on a tick log price.

        Args:
            tick_price: the tick log price.
            change_at: index of the shift in the tick series, or
                ``None`` for none.

        Returns:
            :class:`DriftVsNoiseResult`.
        """
        raw = raw_returns(tick_price)
        robust = preaveraged_returns(tick_price, self.window)
        det_raw = CUSUMDriftDetector(**self.cusum_kwargs)
        det_rob = CUSUMDriftDetector(**self.cusum_kwargs)
        raw_alarms: list[int] = []
        rob_alarms: list[int] = []
        for i, v in enumerate(raw):
            det_raw.update(float(v))
            if det_raw.drift_detected and i >= self.warmup:
                raw_alarms.append(i)
        for i, v in enumerate(robust):
            det_rob.update(float(v))
            if det_rob.drift_detected and i >= self.warmup:
                rob_alarms.append(i)
        delay_raw = None
        delay_rob = None
        if change_at is not None:
            # the robust detector is on pre-averaged returns: 2 * window
            # raw ticks per robust sample
            k = 2 * self.window
            change_rob = change_at // k
            after_raw = [a for a in raw_alarms if a >= change_at]
            if after_raw:
                delay_raw = int(after_raw[0] - change_at)
            after_rob = [a for a in rob_alarms if a >= change_rob]
            if after_rob:
                delay_rob = int(after_rob[0] - change_rob)
        return DriftVsNoiseResult(
            n_samples=len(raw),
            n_episodes_raw=len(raw_alarms),
            n_episodes_robust=len(rob_alarms),
            delay_raw_samples=delay_raw,
            delay_robust_samples=delay_rob,
            change_at=change_at,
            warmup=self.warmup,
        )
