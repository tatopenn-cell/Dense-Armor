"""Cycle residual for a repetitive motion.

``CycleResidual`` finds the period of a repetitive signal from the
autocorrelation of its first samples (or takes it as given), keeps the
last ``n_cycles`` complete cycles, and returns the difference between
each sample and the median of those cycles at the same phase. The
template moves with the signal: every time a cycle is complete, the
oldest cycle leaves and the median is recomputed.

The steps (period search, median template, phase alignment, residual)
come from ``benchmarks/casper_benchmark.py``: ``strongest_period``,
``median_template``, ``template_residual`` and ``phase_jitter``.
"""

from typing import Any

import numpy as np

from dense_armor.roles import Transformer


class CycleResidual(Transformer):
    """Cycle residual of a repetitive scalar stream.

    Until the template exists the residual is ``nan`` and ``ready`` is
    ``0.0``; ``ready`` is ``1.0`` on every sample whose residual is a number. The template exists after ``n_cycles`` complete cycles, so
    with a known period the first residual appears at sample
    ``n_cycles * period``. When the period is searched, the first
    ``2 * max_period`` samples are held to find it (the default
    ``max_period = 1000`` therefore needs 2000 samples); they are then
    replayed into the cycle store and dropped.

    After that the only state is an array of ``n_cycles * period``
    values: memory is ``O(n_cycles * period)``.

    Phase alignment: with ``search > 0``, every complete cycle is
    compared with the template shifted by ``-search .. search`` samples;
    the best shift is applied to the next cycle and the cycle is stored
    back aligned, so the template stays in one phase. ``search = 0``
    disables it.

    The residual of ``x`` is computed against the template of the cycles
    before ``x``, then ``x`` is added to the store.

    Args:
        period: known period, in samples. If ``None`` it is estimated
            from the autocorrelation of the first ``2 * max_period``
            samples.
        max_period: maximum period searched, in samples.
        n_cycles: number of complete cycles in the template.
        feature: key of the channel to read from a dict. If ``None``,
            the smallest key of the first dict is used.
        min_acf: minimum autocorrelation peak accepted as a period.
        search: largest phase shift tried per cycle, in samples.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.timeseries import CycleResidual
        >>> cr = CycleResidual(period=20, n_cycles=3)
        >>> rng = np.random.default_rng(0)
        >>> out = {}
        >>> for i in range(120):
        ...     v = float(np.sin(2 * np.pi * i / 20)) + 0.05 * rng.normal()
        ...     out = cr.transform_one({"x": v})
        >>> abs(out["residual"]) < 1.0 and out["ready"] == 1.0
        True
    """

    budget_s: float | None = 1e-3
    memory_class: str = "O(n_cycles * period)"

    def __init__(
        self,
        period: int | None = None,
        max_period: int = 1000,
        n_cycles: int = 4,
        feature: str | None = None,
        min_acf: float = 0.3,
        search: int = 0,
    ) -> None:
        if period is not None and period < 1:
            raise ValueError(f"period must be >= 1, got {period}")
        if max_period < 2:
            raise ValueError(f"max_period must be >= 2, got {max_period}")
        if n_cycles < 1:
            raise ValueError(f"n_cycles must be >= 1, got {n_cycles}")
        if search < 0:
            raise ValueError(f"search must be >= 0, got {search}")
        self.period = period
        self.max_period = int(max_period)
        self.n_cycles = int(n_cycles)
        self.feature = feature
        self.min_acf = float(min_acf)
        self.search = int(search)
        self._key: str | None = None
        self._buf: list = []
        self._ring: np.ndarray | None = None
        self._slot = 0
        self._phase = 0
        self._done = 0
        self._shift = 0
        self._template: np.ndarray | None = None
        self.n_missing_: int = 0

    @property
    def n_missing(self) -> int:
        """Count of samples skipped because of a missing or bad value."""
        return self.n_missing_

    @property
    def ready(self) -> bool:
        """True once the template exists."""
        return self._template is not None

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

    def _start_store(self) -> None:
        assert self.period is not None
        self._ring = np.zeros((self.n_cycles, self.period))
        self._slot = 0
        self._phase = 0
        self._done = 0
        self._shift = 0
        self._template = None

    def _best_shift(self, row: np.ndarray) -> int:
        assert self._template is not None
        best_err, best_s = np.inf, 0
        for s in range(-self.search, self.search + 1):
            err = float(np.mean((row - np.roll(self._template, s)) ** 2))
            if err < best_err:
                best_err, best_s = err, s
        return best_s

    def _push(self, v: float) -> None:
        assert self._ring is not None and self.period is not None
        self._ring[self._slot, self._phase] = v
        self._phase += 1
        if self._phase < self.period:
            return
        row = self._ring[self._slot]
        if self.search > 0 and self._template is not None:
            shift = self._best_shift(row)
            self._ring[self._slot] = np.roll(row, -shift)
            self._shift = shift
        self._done += 1
        if self._done >= self.n_cycles:
            self._template = np.median(self._ring, axis=0)
        self._phase = 0
        self._slot = (self._slot + 1) % self.n_cycles

    def transform_one(self, x: dict, t: float | None = None) -> dict:
        """Residual of ``x`` against the template, and the ``ready`` flag."""
        v = self._read(x)
        if v is None:
            self.n_missing_ += 1
            return {"residual": float("nan"), "ready": 0.0}
        if self.period is None:
            self._buf.append(v)
            if len(self._buf) >= 2 * self.max_period:
                p = self._find_period()
                if p is not None:
                    self.period = p
                    self._start_store()
                    for u in self._buf:
                        self._push(u)
                    self._buf = []
                else:
                    del self._buf[: len(self._buf) - 2 * self.max_period + 1]
            return {"residual": float("nan"), "ready": 0.0}
        if self._ring is None:
            self._start_store()
        assert self.period is not None
        if self._template is None:
            residual = float("nan")
        else:
            idx = (self._phase - self._shift) % self.period
            residual = v - float(self._template[idx])
        self._push(v)
        return {"residual": residual, "ready": 0.0 if np.isnan(residual) else 1.0}
