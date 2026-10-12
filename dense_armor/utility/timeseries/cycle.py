"""Cycle residual for a repetitive motion.

``CycleResidual`` finds the period of a repetitive signal from the
autocorrelation of the first samples, keeps a median template of the
last ``n_cycles`` cycles, aligns the phase of the current sample, and
returns the difference between the sample and the template. The raw
buffer that holds the samples is bounded by
``n_cycles * period + 2 * max_period``: after the template is built,
only the most recent window is kept.

The four steps (period search, median template, phase alignment,
residual) are promoted from ``benchmarks/casper_benchmark.py``.
"""

from typing import Any

import numpy as np

from dense_armor.roles import Transformer


class CycleResidual(Transformer):
    """Cycle residual of a repetitive scalar stream.

    The raw sample buffer is bounded: after the period is found and the
    template is built, only the most recent ``n_cycles * period + 2 *
    max_period`` samples are kept. Before the period is known, the
    buffer holds at most ``2 * max_period`` samples. Memory is therefore
    ``O(window)`` with ``window = n_cycles * period + 2 * max_period``.

    Args:
        period: known period, in samples. If ``None`` the period is
            estimated from the autocorrelation of the first
            ``max_period * 2`` samples.
        max_period: maximum period searched, in samples.
        n_cycles: number of cycles kept in the template.
        feature: key of the channel to read from a dict. If ``None``,
            the smallest key of the first dict is used.
        min_acf: minimum autocorrelation peak accepted as a period.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.timeseries import CycleResidual
        >>> cr = CycleResidual(period=20, n_cycles=3)
        >>> rng = np.random.default_rng(0)
        >>> out = {}
        >>> for i in range(120):
        ...     v = float(np.sin(2 * np.pi * i / 20)) + 0.05 * rng.normal()
        ...     out = cr.transform_one({"x": v})
        >>> abs(out["residual"]) < 1.0
        True
    """

    budget_s: float | None = 1e-3
    memory_class: str = "O(window)"

    def __init__(
        self,
        period: int | None = None,
        max_period: int = 1000,
        n_cycles: int = 4,
        feature: str | None = None,
        min_acf: float = 0.3,
    ) -> None:
        if period is not None and period < 1:
            raise ValueError(f"period must be >= 1, got {period}")
        if max_period < 2:
            raise ValueError(f"max_period must be >= 2, got {max_period}")
        if n_cycles < 1:
            raise ValueError(f"n_cycles must be >= 1, got {n_cycles}")
        self.period = period
        self.max_period = int(max_period)
        self.n_cycles = int(n_cycles)
        self.feature = feature
        self.min_acf = float(min_acf)
        self._key: str | None = None
        self._buf: list = []
        self._template: np.ndarray | None = None
        self._phase: int = 0
        self.n_missing_: int = 0

    @property
    def n_missing(self) -> int:
        """Count of samples skipped because of a missing or bad value."""
        return self.n_missing_

    def _key_of(self, x: dict) -> str:
        if self.feature is not None:
            return self.feature
        if self._key is None:
            self._key = min(x)
        return self._key

    def _read(self, x: Any) -> float | None:
        if not isinstance(x, dict):
            return None
        try:
            v = x[self._key_of(x)]
            f = float(v)
            return f if np.isfinite(f) else None
        except (TypeError, ValueError, KeyError, IndexError):
            return None

    def _find_period(self) -> int | None:
        n = len(self._buf)
        max_lag = min(self.max_period, n - 2)
        if max_lag < 2:
            return None
        y = np.asarray(self._buf, dtype=float)
        y = y - y.mean()
        F = np.fft.rfft(y, n=2 * n)
        acf = np.fft.irfft(F * np.conj(F))[:n]
        if acf[0] <= 0.0:
            return None
        acf = acf / acf[0]
        best_lag, best_acf = 0, 0.0
        for i in range(2, max_lag):
            is_peak = (
                acf[i] > acf[i - 1] and acf[i] > acf[i + 1] and acf[i] > self.min_acf
            )
            if is_peak and acf[i] > best_acf:
                best_lag, best_acf = i, float(acf[i])
        return best_lag if best_lag > 0 else None

    def _build_template(self) -> None:
        if self.period is None:
            return
        n_needed = self.period * self.n_cycles
        if len(self._buf) < n_needed:
            return
        block = np.asarray(self._buf[:n_needed], dtype=float)
        block = block.reshape(self.n_cycles, self.period)
        self._template = np.median(block, axis=0)
        self._phase = (len(self._buf) - 1) % self.period

    def transform_one(self, x: dict, t: float | None = None) -> dict:
        """One-step-ahead residual against the template."""
        v = self._read(x)
        if v is None:
            return {"residual": float("nan")}
        self._buf.append(v)
        if self.period is None and len(self._buf) >= 2 * self.max_period:
            p = self._find_period()
            if p is not None:
                self.period = p
        if self._template is None:
            self._build_template()
            if self._template is None:
                self._prune()
                return {"residual": 0.0}
        assert self.period is not None
        idx = self._phase % self.period
        residual = v - float(self._template[idx])
        self._phase = (self._phase + 1) % self.period
        self._prune()
        return {"residual": residual}

    def _prune(self) -> None:
        """Keep only the recent window of samples.

        Before the period is known, the buffer holds at most
        ``2 * max_period`` samples. After the template is built, it
        holds ``n_cycles * period`` samples (the template window) plus
        ``2 * max_period`` samples for the period-search margin: the
        autocorrelation peak of the newest part is used only if the
        period changes, and a small recent margin keeps the running
        statistics stable.
        """
        cap = 2 * self.max_period
        if self.period is not None:
            cap = self.n_cycles * self.period + 2 * self.max_period
        if len(self._buf) > cap:
            del self._buf[: len(self._buf) - cap]
