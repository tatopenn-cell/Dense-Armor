# -*- coding: utf-8 -*-
"""
utility/streaming_arbiter.py
=============================
One-sample-at-a-time version of `arbiter.classify_segments` + `route_and_correct`.

The batch function decides spike vs regime by grouping consecutive deviant
points into runs and looking at what happens *after* the run (to distinguish
a genuine regime shift from a transient collapse that returns). This is
fundamentally non-causal: the decision for a point requires future context.
`StreamingArbiter` therefore decides with a bounded delay: each sample gets
a provisional label at once (deviant / normal, from the causal window) and a
final label (`clean` / `spike` / `regime`) after at most `max_delay` samples.

With `max_delay` large enough to cover both the longest run and the widest
window the batch correction uses (`radius * ref_mult`), the final labels and
corrected values equal the batch `classify_segments` + `route_and_correct`
on the same series. With smaller `max_delay`, labels near run boundaries may
change (documented in the latency table of `arbiter.md`).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def _robust_center_scale(w: np.ndarray) -> Tuple[float, float]:
    med = float(np.median(w))
    mad = float(np.median(np.abs(w - med)))
    return med, 1.4826 * mad


class StreamingArbiter:
    """Bounded-delay streaming version of `arbiter.classify_segments`.

    `update(x)` adds a sample and returns the list of samples whose final
    label was decided at this step, as `(index, label, corrected_value)`.
    `flush()` finalises everything still pending (call it at the end of a
    series).

    Parameters
    ----------
    radius, ref_mult, n_sigmas, spike_run_max, eps
        Same meaning and defaults as `arbiter.classify_segments`.
    max_delay
        Maximum number of future samples the streaming version will wait
        before committing to a final label. With ``max_delay`` at least
        ``longest_run + radius * ref_mult``, the result is identical to
        the batch function; smaller values trade exactness for lower
        latency.

    Examples
    --------
    >>> import numpy as np
    >>> from dense_armor.protect.streaming_arbiter import StreamingArbiter
    >>> rng = np.random.default_rng(0)
    >>> x = list(rng.normal(0, 1, 100)) + [50.0] + list(rng.normal(0, 1, 100))
    >>> sa = StreamingArbiter(radius=10, max_delay=200)
    >>> decided = []
    >>> for v in x:
    ...     decided.extend(sa.update(v))
    >>> decided.extend(sa.flush())
    >>> spike_idx = [i for i, lab, _ in decided if lab == "spike"]
    >>> 100 in spike_idx
    True
    """

    def __init__(self, radius: int = 10, ref_mult: int = 3,
                 n_sigmas: float = 3.0, spike_run_max: int = 2,
                 max_delay: int = 40, eps: float = 1e-9):
        self.radius = radius
        self.ref_mult = ref_mult
        self.n_sigmas = n_sigmas
        self.spike_run_max = spike_run_max
        self.max_delay = int(max_delay)
        self.eps = eps
        self._x: List[float] = []
        self._deviante: List[bool] = []
        self._med_rif: List[float] = []
        self._labels: List[Optional[str]] = []
        self._corrected: List[float] = []
        self._next_undecided: int = 0
        self._emitted: List[Tuple[int, str, float]] = []

    def _causal_window(self, i: int) -> np.ndarray:
        lo = max(0, i - self.radius * self.ref_mult)
        return np.asarray(self._x[lo:i], dtype=float)

    def _compute_deviante(self, i: int) -> None:
        w = self._causal_window(i)
        if w.size < 4:
            self._deviante.append(False)
            self._med_rif.append(0.0)
            return
        med, scale = _robust_center_scale(w)
        self._med_rif.append(med)
        scarto = abs(self._x[i] - med)
        if scale < self.eps:
            self._deviante.append(scarto > self.eps)
            return
        self._deviante.append(scarto / scale > self.n_sigmas)

    def _decide_one_run(self, s: int, j: int, H: int) -> str:
        run_len = j - s
        if run_len <= self.spike_run_max:
            return "spike"
        run_vals = np.asarray(self._x[s:j], dtype=float)
        run_spread = float(np.std(run_vals))
        run_jump = float(abs(np.median(run_vals) - self._med_rif[s]))
        coerente = run_spread < 0.5 * max(run_jump, self.eps)
        post_end = min(H, j + self.radius)
        if post_end > j:
            post_med = float(np.median(self._x[j:post_end]))
            run_med = float(np.median(run_vals))
            persiste = abs(post_med - run_med) < abs(post_med - self._med_rif[s])
        else:
            persiste = True
        return "regime" if (coerente and persiste) else "spike"

    def _assign(self, s: int, j: int, label: str, H: int) -> None:
        span = self.radius * self.ref_mult
        for i in range(s, j):
            self._labels[i] = label
            if label == "spike":
                lo = max(0, i - span)
                hi = min(H, i + span + 1)
                self._corrected[i] = float(np.median(self._x[lo:hi]))
            else:
                self._corrected[i] = self._x[i]
            self._emitted.append((i, label, self._corrected[i]))

    def _sweep(self, forced: bool) -> None:
        H = len(self._x)
        span = self.radius * self.ref_mult
        while self._next_undecided < H:
            k = self._next_undecided
            if not self._deviante[k]:
                self._labels[k] = "clean"
                self._corrected[k] = self._x[k]
                self._emitted.append((k, "clean", self._x[k]))
                self._next_undecided += 1
                continue
            s = k
            j = s
            while j < H and self._deviante[j]:
                j += 1
            run_closed = j < H
            if run_closed:
                full_post = H >= (j - 1) + span + 1
                force_ok = (H - s) >= self.max_delay
                if not (full_post or force_ok or forced):
                    break
            else:
                if not (forced or (H - s) >= self.max_delay):
                    break
            label = self._decide_one_run(s, j, H)
            self._assign(s, j, label, H)
            self._next_undecided = j

    def update(self, x: float) -> List[Tuple[int, str, float]]:
        """Add a sample and return the samples finalised at this step."""
        self._emitted = []
        self._x.append(float(x))
        self._labels.append(None)
        self._corrected.append(float(x))
        self._compute_deviante(len(self._x) - 1)
        self._sweep(forced=False)
        return list(self._emitted)

    def flush(self) -> List[Tuple[int, str, float]]:
        """Finalise every sample still pending."""
        self._emitted = []
        self._sweep(forced=True)
        return list(self._emitted)


class StreamingHealing:
    """One-sample-at-a-time version of `healing.healing_filter`.

    `healing_filter` judges each point against a wide local baseline and
    replaces it when the deviation is not shared by the majority of a
    narrow window of neighbours. It uses a symmetric wide window, so the
    streaming version decides with the same bounded-delay idea as
    `StreamingArbiter`: each sample gets a corrected value as soon as the
    wide symmetric window around it is fully available.

    Parameters
    ----------
    radius, sustain_threshold, wide_mult
        Same meaning and defaults as `healing.healing_filter`.
    max_delay
        Maximum number of samples to wait before finalising.

    Examples
    --------
    >>> import numpy as np
    >>> from dense_armor.protect.streaming_arbiter import StreamingHealing
    >>> rng = np.random.default_rng(0)
    >>> x = list(rng.normal(0, 0.1, 50)) + [5.0] + list(rng.normal(0, 0.1, 50))
    >>> sh = StreamingHealing(radius=2, wide_mult=3, max_delay=100)
    >>> out = []
    >>> for v in x:
    ...     out.extend(sh.update(v))
    >>> out.extend(sh.flush())
    >>> out = [v for _, v in sorted(out, key=lambda t: t[0])]
    >>> abs(out[50]) < 1.0
    True
    """

    def __init__(self, radius: int = 2, sustain_threshold: float = 0.7,
                 wide_mult: int = 3, max_delay: int = 30):
        self.radius = radius
        self.sustain_threshold = sustain_threshold
        self.wide_mult = wide_mult
        self.max_delay = int(max_delay)
        self._x: List[float] = []
        self._decided: List[Optional[float]] = []
        self._next: int = 0
        self._emitted: List[Tuple[int, float]] = []

    def _decide(self, i: int, H: int) -> float:
        wide = self.radius * self.wide_mult
        lo_w, hi_w = max(0, i - wide), min(H, i + wide + 1)
        baseline = float(np.median(self._x[lo_w:hi_w]))
        lo, hi = max(0, i - self.radius), min(H, i + self.radius + 1)
        x_i = self._x[i]
        dev_i = x_i - baseline
        if abs(dev_i) < 1e-9:
            return x_i
        window = np.asarray(self._x[lo:hi], dtype=float)
        devs = window - baseline
        same_sign_share = float(np.mean(
            (np.sign(devs) == np.sign(dev_i))
            & (np.abs(devs) > self.sustain_threshold * abs(dev_i))
        ))
        return x_i if same_sign_share > 0.5 else baseline

    def _sweep(self, forced: bool) -> None:
        H = len(self._x)
        wide = self.radius * self.wide_mult
        while self._next < H:
            k = self._next
            full_wide = (k + wide) < H
            force_ok = (H - k) >= self.max_delay
            if not (full_wide or force_ok or forced):
                break
            self._decided[k] = self._decide(k, H)
            self._emitted.append((k, self._decided[k]))
            self._next += 1

    def update(self, x: float) -> List[Tuple[int, float]]:
        self._emitted = []
        self._x.append(float(x))
        self._decided.append(None)
        self._sweep(forced=False)
        return list(self._emitted)

    def flush(self) -> List[Tuple[int, float]]:
        self._emitted = []
        self._sweep(forced=True)
        return list(self._emitted)
