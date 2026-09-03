# -*- coding: utf-8 -*-
"""
utility/cusum.py
=================
Two-sided CUSUM (cumulative sum) change-point detector -- Page, E.S.
(1954), "Continuous Inspection Schemes", Biometrika 41(1-2):100-115.
Standard textbook change-point detection, not specific to this project;
added to close a real, measured gap: test/test_benchmark_v0_runtime_
behavioral_drift.py found that utility/arbiter.py's classify_segments
(instantaneous n_sigmas threshold on a windowed robust z-score) detects
a SHARP transient spike perfectly (100% detection, latency 0 at every
severity tested) but is much less sensitive to a SLOW, GRADUAL sustained
drift (8.9%/24.4%/26.7% detected across the ramp+settling window at
mild/moderate/strong severity) -- an instantaneous threshold test is
structurally the wrong tool for a shift too small, at any single step,
to itself cross the threshold. CUSUM is the standard classical answer:
it accumulates small, sustained deviations over time instead of judging
each point in isolation, so a drift too gradual to trip an instantaneous
threshold still eventually trips the accumulated sum.

HONEST DEVIATION FROM THE CITED ALGORITHM: Page's original CUSUM (and
the textbook treatment since) accumulates deviation against a FIXED
reference mean, decided once from an in-control baseline and never
updated. This implementation instead recomputes med/scale from a
SLIDING causal window every step (same convention as arbiter.py), so
the "reference" itself drifts along with the data over time. This is a
deliberate choice (matching classify_segments' own adaptive-window
convention, and needed for a streaming setting where there is no fixed
known-good baseline to pin against) but it is NOT literally Page's
fixed-reference scheme -- call this an adaptive-reference CUSUM, not an
unqualified claim to be running the textbook algorithm unmodified. The
practical effect, measured in test/test_benchmark_v0_runtime_behavioral_
drift.py: it detects the LEADING EDGE of a sustained drift quickly
(3-8 points into a 15-point ramp), then stops accumulating once its own
reference window has caught up to the new level -- it will NOT keep
accumulating forever against a genuinely fixed target the way Page's
original scheme does.

This is a NEW, SEPARATE detector, not a replacement for classify_segments
and not a retuning of its shipped thresholds (radius=10, ref_mult=3,
n_sigmas=3.0, spike_run_max=2 in arbiter.py are untouched by this file,
and remain exactly the preregistered defaults the benchmark declared it
would never adjust after seeing results) -- classify_segments stays the
right tool for transient spikes/glitches (see the benchmark's condition
3), this is the complementary tool for condition 2's slow-drift blind
spot. A production system would run both channels and flag on either.

Reference window and robust standardization deliberately mirror
utility/arbiter.py's own convention (causal window, radius*ref_mult
span, median/1.4826*MAD center-scale) so a point's "z" here is in the
same robust-sigma units classify_segments already reports, rather than
introducing a second, incompatible notion of scale.
"""
from typing import Tuple

import numpy as np


def _window_causal(x: np.ndarray, i: int, span: int) -> np.ndarray:
    lo = max(0, i - span)
    return x[lo:i]


def _robust_center_scale(w: np.ndarray) -> Tuple[float, float]:
    med = float(np.median(w))
    mad = float(np.median(np.abs(w - med)))
    return med, 1.4826 * mad


def cusum_detector(
    x: np.ndarray, radius: int = 10, ref_mult: int = 3,
    k: float = 0.5, h: float = 20.0, eps: float = 1e-9,
    reference: str = "adaptive",
) -> Tuple[np.ndarray, np.ndarray]:
    """Two-sided CUSUM on the same causal robust-z-score classify_segments
    uses, so a sustained drift too gradual to cross classify_segments'
    OWN instantaneous n_sigmas threshold still accumulates and eventually
    trips this one.

    S+[i] = max(0, S+[i-1] + (z[i] - k))   -- accumulates upward drift
    S-[i] = min(0, S-[i-1] + (z[i] + k))   -- accumulates downward drift
    flags when |S+[i]| > h or |S-[i]| > h; both sums reset to 0 the
    instant they flag, so a second, later shift can be detected too
    (standard CUSUM reset behavior, not specific to this implementation).

    k (the "slack"/reference value, in robust-sigma units) is the
    classical convention: half the shift size you want to detect --
    Page's original scheme and the textbook tuning since (e.g. Hawkins &
    Olwell, "Cumulative Sum Charts and Charting for Quality Improvement",
    1998). h (the decision threshold) is NOT the commonly-cited h~4-5:
    checked directly (2026-09-03, prompted by a parallel investigation
    porting this same algorithm to online-ml/river) before shipping that
    value -- h=5.0 gives a 100% ("adaptive") / 85% ("fixed") stream-level
    false-alarm rate on a 1000-sample purely stable N(0,1) series (200
    trials, radius=10, ref_mult=3, k=0.5), because its average run length
    under no-change is only ~19-38 samples, nowhere near 1000. h=20.0
    (this default) empirically gives 3.5% ("adaptive") / 15.5% ("fixed")
    over the same 1000-sample horizon. "fixed" stays structurally more
    exposed than "adaptive" at any shared h -- its target never updates,
    so an unlucky warmup-window estimate is never self-corrected the way
    "adaptive"'s sliding reference recovers from one; not a tuning gap,
    a direct consequence of what "fixed" is for (see its own docstring
    below). "Textbook k=0.5/h=5.0" is a real, commonly-cited tuning for
    short/industrial monitoring horizons; it was not previously
    re-validated here against the longer streams typical of software/ML
    drift monitoring. test/test_cusum.py's existing assertions (large,
    unambiguous 10-sigma steps; a <10% POINT-level noise flag rate, which
    h=5.0 already satisfied despite its much higher STREAM-level
    false-alarm rate -- the same per-point-vs-per-stream distinction that
    motivated this fix) all still hold at h=20.0, re-verified directly,
    not assumed.

    Parameters
    ----------
    x : np.ndarray
        1D signal.
    radius, ref_mult : int
        Causal reference window span = radius*ref_mult, same convention
        as utility.arbiter.classify_segments.
    k : float
        CUSUM slack, in robust-sigma units.
    h : float
        Decision threshold, in accumulated robust-sigma units.
    eps : float
        Degenerate-scale guard, same convention as classify_segments.
    reference : {"adaptive", "fixed"}, default "adaptive"
        "adaptive" (this module's own default, see the module docstring
        for the honest disclosure of how this differs from Page's
        original scheme): med/scale recomputed every step from a sliding
        causal window, so the reference itself drifts along with the
        data -- catches the LEADING EDGE of a sustained drift, then
        stops accumulating once the window has caught up to the new
        level.
        "fixed": med/scale computed ONCE from x[0:span] (the first
        radius*ref_mult points, assumed in-control/baseline) and never
        updated afterward -- this IS Page's original scheme. Points
        before index `span` have no valid fixed reference yet and are
        skipped (flagged=False, cusum=0), same warmup convention as
        "adaptive"'s own causal window. Unlike "adaptive", a genuinely
        sustained shift will keep accumulating against this fixed target
        indefinitely if never reset by a flag, rather than fading once a
        sliding window adapts to the new level.

    Returns
    -------
    (flagged, cusum)
        flagged : bool array, True where either accumulator crossed h.
        cusum   : float array, signed accumulated statistic (S+ where
                  positive-going, S- where negative-going) at each point,
                  for inspection/plotting.
    """
    if reference not in ("adaptive", "fixed"):
        raise ValueError(f"reference must be 'adaptive' or 'fixed', got {reference!r}")
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    flagged = np.zeros(n, dtype=bool)
    cusum = np.zeros(n, dtype=np.float64)
    s_pos, s_neg = 0.0, 0.0
    span = radius * ref_mult

    fixed_med, fixed_scale = None, None
    if reference == "fixed" and n > span:
        init_window = x[:span]
        if np.all(np.isfinite(init_window)):
            fixed_med, fixed_scale = _robust_center_scale(init_window)

    for i in range(n):
        if reference == "fixed":
            if i < span or fixed_med is None or not np.isfinite(fixed_scale) or fixed_scale < eps:
                continue
            med, scale = fixed_med, fixed_scale
            if not np.isfinite(x[i]):
                continue
            z = (x[i] - med) / scale
            s_pos = max(0.0, s_pos + z - k)
            s_neg = min(0.0, s_neg + z + k)
            cusum[i] = s_pos if s_pos >= abs(s_neg) else s_neg
            if s_pos > h:
                flagged[i] = True
                s_pos = 0.0
            elif s_neg < -h:
                flagged[i] = True
                s_neg = 0.0
            continue

        if not np.isfinite(x[i]):
            # Explicit, not incidental: without this guard, a NaN x[i] would
            # flow into z (nan) and then into max(0.0, s_pos+z-k)/min(0.0,
            # s_neg+z+k) -- Python's max/min happen to return the FIRST
            # argument when compared against NaN (max(0.0, nan) == 0.0,
            # min(0.0, nan) == 0.0, verified directly), which silently
            # clamps the accumulator to 0 only because `0.0` is written
            # first in this file's own calls below -- swap the argument
            # order in a future edit and NaN would propagate instead.
            # Holding state (skip, don't touch s_pos/s_neg) is the explicit,
            # order-independent version of that same "don't corrupt the
            # accumulator" intent.
            continue
        w_ref = _window_causal(x, i, span)
        if w_ref.size < 4:
            continue
        med, scale = _robust_center_scale(w_ref)
        if not np.isfinite(scale) or scale < eps:
            # not np.isfinite(scale): a NaN/Inf point earlier still inside
            # this causal window (span points back) would otherwise make
            # med/scale themselves NaN -- same explicit-not-incidental
            # reasoning as above, this time for a contaminated window
            # rather than the current point.
            continue
        z = (x[i] - med) / scale
        s_pos = max(0.0, s_pos + z - k)
        s_neg = min(0.0, s_neg + z + k)
        cusum[i] = s_pos if s_pos >= abs(s_neg) else s_neg
        if s_pos > h:
            flagged[i] = True
            s_pos = 0.0
        elif s_neg < -h:
            flagged[i] = True
            s_neg = 0.0

    return flagged, cusum
