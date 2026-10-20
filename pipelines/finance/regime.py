"""Regime pipeline: has the market moved to a different volatility regime?

The pipeline reads a daily close series, turns it into log returns, and
feeds the returns to :class:`BayesianOnlineChangepoint`
(:mod:`dense_armor.utility.drift.bocpd`, Adams and MacKay 2007,
arXiv:0710.3742). A changepoint is a day on which the volatility or the
mean of the returns changed, and the alarm is the first day the
detector reports one after the change.

Reference: Adams, R. P. and MacKay, D. J. C. (2007). *Bayesian Online
Changepoint Detection.* arXiv:0710.3742. The hazard is equation 4,
page 2; the run length recursion is Algorithm 1, pages 3-4.

The client is the risk manager or the execution desk: a regime change
means the risk model and the execution schedule are tuned for the wrong
market. The pipeline answers "when did the market change regime" and
"how many false alarms per year of calm data".

The threshold of the detector, the window of run lengths whose mass
raises the alarm, and the number of warm-up returns are choices of this
pipeline: the paper reports the run length posterior, not a threshold
on it. They are declared in the docstring of :class:`RegimePipeline`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np

from dense_armor.utility.drift.bocpd import BayesianOnlineChangepoint

_YAHOO_HEADER_ROWS = 3


def load_log_returns(path) -> np.ndarray:
    """Daily close to daily log returns from a Yahoo CSV.

    The file has three header rows (``Price``, ``Ticker``, ``Date``) and
    then one row per day with the columns in alphabetical order. The
    close is the ``Close`` column; the log return is
    ``log(close_t / close_{t-1})``.

    Args:
        path: path of the CSV.

    Returns:
        A 1-D array of log returns, length ``n_days - 1``.

    Raises:
        ValueError: if the file has fewer than four rows, or no
            ``Close`` column, or a non-positive close.
    """
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if len(rows) <= _YAHOO_HEADER_ROWS:
        raise ValueError(f"{path}: fewer than {_YAHOO_HEADER_ROWS + 1} rows")
    header = rows[0]
    if "Close" not in header:
        raise ValueError(f"{path}: no Close column, header is {header}")
    j = header.index("Close")
    closes = []
    for row in rows[_YAHOO_HEADER_ROWS:]:
        if len(row) <= j:
            continue
        try:
            c = float(row[j])
        except ValueError:
            continue
        closes.append(c)
    closes = np.asarray(closes, dtype=float)
    if np.any(closes <= 0):
        raise ValueError(f"{path}: a close is not positive")
    return np.log(closes[1:] / closes[:-1])


def synthetic_two_regime(
    n: int = 1000,
    seed: int = 0,
    change_at: int = 500,
    vol_low: float = 0.01,
    vol_high: float = 0.03,
) -> tuple[np.ndarray, int]:
    """Two Gaussian regimes with different volatility, and the true index.

    Args:
        n: number of returns.
        seed: RNG seed.
        change_at: index of the first return of the second regime.
        vol_low: standard deviation of the first regime.
        vol_high: standard deviation of the second regime.

    Returns:
        ``(returns, change_at)``: the returns, and the index of the first
        sample of the second regime.
    """
    rng = np.random.default_rng(seed)
    x = np.empty(n, dtype=float)
    x[:change_at] = rng.normal(0.0, vol_low, change_at)
    x[change_at:] = rng.normal(0.0, vol_high, n - change_at)
    return x, int(change_at)


@dataclass
class RegimeResult:
    """Result of a regime run on a return series.

    Attributes:
        n_returns: length of the series.
        alarms: indices of the alarms (0-based, on the returns).
        n_alarms: number of alarms.
        first_alarm_after_change: the first alarm at or after the planted
            change, or ``None``.
        delay_samples: ``first_alarm_after_change - change_at``, or
            ``None``. It is the delay in samples.
        years_of_calm: number of years of the calm part, at 252 returns
            per year.
        alarm_rate_per_year: alarms in the calm part divided by
            ``years_of_calm``.
        change_at: the planted change index, or ``None`` for a real
            series.
    """

    n_returns: int
    alarms: list[int]
    n_alarms: int
    first_alarm_after_change: int | None
    delay_samples: int | None
    years_of_calm: float
    alarm_rate_per_year: float
    change_at: int | None


class RegimePipeline:
    """Run the BOCPD detector on a return series.

    Args:
        lam: mean of the discrete exponential prior on the interval
            between changepoints, forwarded to the detector
            (``H(tau) = 1 / lam``). Default 252.0, about one year of
            trading days. It is this pipeline's choice.
        max_run_length: maximum run length kept by the detector.
            Default 252, one year of trading days. It is this
            pipeline's choice.
        mass_window: number of shortest run lengths whose mass raises
            the alarm, forwarded to the detector. Default 10. It is
            this pipeline's choice.
        detection_threshold: mass above which the detector alarms.
            Default 0.5. It is this pipeline's choice.
        warmup: number of returns skipped before the detector is read.
            The first alarms happen while the run length is still one
            sample long, before the detector has seen a normal run.
            Default 20. It is this pipeline's choice.

    Raises:
        ValueError: on a negative ``warmup``.
    """

    def __init__(
        self,
        lam: float = 252.0,
        max_run_length: int = 252,
        mass_window: int = 10,
        detection_threshold: float = 0.5,
        warmup: int = 20,
    ) -> None:
        if warmup < 0:
            raise ValueError(f"warmup must be >= 0, got {warmup}")
        self.lam = float(lam)
        self.max_run_length = int(max_run_length)
        self.mass_window = int(mass_window)
        self.detection_threshold = float(detection_threshold)
        self.warmup = int(warmup)

    def run(
        self,
        returns,
        change_at: int | None = None,
    ) -> RegimeResult:
        """Run the detector on ``returns``.

        Args:
            returns: a 1-D array of returns.
            change_at: the planted change index, or ``None`` for a real
                series.

        Returns:
            :class:`RegimeResult`.
        """
        x = np.asarray(returns, dtype=float)
        det = BayesianOnlineChangepoint(
            lam=self.lam,
            max_run_length=self.max_run_length,
            mass_window=self.mass_window,
            detection_threshold=self.detection_threshold,
        )
        alarms: list[int] = []
        for i, v in enumerate(x):
            det.update(float(v))
            if det.drift_detected and i >= self.warmup:
                alarms.append(i)
        first = None
        delay = None
        if change_at is not None:
            after = [a for a in alarms if a >= change_at]
            if after:
                first = after[0]
                delay = first - change_at
        if change_at is not None:
            n_calm = change_at
        else:
            n_calm = len(x) // 2
        calm_alarms = [a for a in alarms if a < n_calm]
        years = n_calm / 252.0
        rate = float(len(calm_alarms) / years) if years > 0 else float("nan")
        return RegimeResult(
            n_returns=len(x),
            alarms=alarms,
            n_alarms=len(alarms),
            first_alarm_after_change=first,
            delay_samples=delay,
            years_of_calm=years,
            alarm_rate_per_year=rate,
            change_at=change_at,
        )


def run_pipeline(
    path,
    lam: float = 252.0,
    warmup: int = 20,
) -> RegimeResult:
    """Read a Yahoo CSV and run the pipeline on its log returns."""
    returns = load_log_returns(path)
    pipe = RegimePipeline(lam=lam, warmup=warmup)
    return pipe.run(returns, change_at=None)


def run_synthetic(
    n_seeds: int = 20,
    n: int = 1000,
    change_at: int = 500,
    lam: float = 252.0,
    warmup: int = 20,
) -> dict:
    """Run the pipeline on ``n_seeds`` synthetic two-regime streams.

    Returns a dict with the per-seed delays (``None`` when a seed has no
    alarm after the change), the per-seed alarm rates per year of calm
    data, and the median and interquartile range of the delays and the
    rates over the seeds that have a delay.
    """
    delays: list[int] = []
    rates: list[float] = []
    per_seed: list[dict] = []
    for s in range(n_seeds):
        x, ch = synthetic_two_regime(n=n, seed=s, change_at=change_at)
        pipe = RegimePipeline(lam=lam, warmup=warmup)
        r = pipe.run(x, change_at=ch)
        per_seed.append({
            "seed": s,
            "delay_samples": r.delay_samples,
            "n_alarms": r.n_alarms,
            "alarm_rate_per_year": r.alarm_rate_per_year,
        })
        if r.delay_samples is not None:
            delays.append(int(r.delay_samples))
        rates.append(float(r.alarm_rate_per_year))
    d = np.asarray(delays, dtype=float)
    ra = np.asarray(rates, dtype=float)
    return {
        "n_seeds": n_seeds,
        "n": n,
        "change_at": change_at,
        "per_seed": per_seed,
        "delay_median": float(np.median(d)) if d.size else None,
        "delay_iqr": float(np.percentile(d, 75) - np.percentile(d, 25))
                      if d.size else None,
        "rate_median": float(np.median(ra)) if ra.size else None,
        "rate_iqr": float(np.percentile(ra, 75) - np.percentile(ra, 25))
                    if ra.size else None,
        "n_with_detection": int(d.size),
    }
