# -*- coding: utf-8 -*-
"""
utility/one_sided.py
======================
one_sided_upper_filter -- restricts any boolean flag array (from
utility.arbiter.classify_segments, utility.cusum.cusum_detector, or any
other detector returning one flag per point) to points that are
strictly ABOVE their own causal reference median, discarding flags
triggered by a point being unusually LOW.

WHY THIS EXISTS: for a signal where only an increase is ever meaningful
(latency, error rate, queue depth -- a fast response or an empty queue
is never itself an anomaly), the generic two-sided detectors in this
package (by design general-purpose, flagging deviation in EITHER
direction) over-flag by counting "unusually fast" as evidence on equal
footing with "unusually slow". Found and measured directly, not
assumed: on real Qwen2 1.8B tool-use latency telemetry (test/
agent_v2/telemetry_v2_1_fresh.jsonl), this filter alone cut
classify_segments' baseline false-positive rate from 22.5% to 12.5% and
cusum_detector's from 17.5% to 10.0%, and cut the false-reject rate on
a genuine, legitimate behavioral change (a real task-domain switch,
not an injected fault) from 24.0% to 4.0% for both -- see
test/test_benchmark_v2_1_ablation.py for the full decomposition,
including the finding that a log-transform tried ALONGSIDE this barely
moved any of these numbers and sometimes made them worse, so it was
dropped rather than kept as unproven complexity.

THE REAL COST, not hidden: the same filter also cut persistent
(sustained, gradual) drift detection substantially in that same
experiment (classify_segments 16.0%->4.0%, cusum_detector 20.0%->8.0%)
-- discarding "flagged because low" also discards some of the weaker,
ambiguous "flagged because moderately high" evidence a two-sided
threshold would have kept on the borderline. This is a real tradeoff,
not a free improvement: use this filter when false alarms are the
dominant cost for your signal (the common case for latency-style
runtime monitoring), not when catching a subtle gradual drift matters
more than avoiding false alarms.

Does not modify classify_segments/cusum_detector themselves -- both
stay general-purpose two-sided detectors, usable directly for signals
where a decrease is just as meaningful as an increase (e.g. a
similarity score, a confidence score bounded on both sides).
"""
from typing import Tuple

import numpy as np


def _causal_median(x: np.ndarray, span: int) -> np.ndarray:
    n = len(x)
    meds = np.zeros(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - span)
        w = x[lo:i]
        meds[i] = float(np.median(w)) if w.size > 0 else float(x[i])
    return meds


def one_sided_upper_filter(
    x: np.ndarray, flags: np.ndarray, radius: int = 10, ref_mult: int = 3,
) -> np.ndarray:
    """Keep a flag only where x[i] is strictly above its own causal
    reference median (same radius*ref_mult causal-window convention as
    utility.arbiter.classify_segments/utility.cusum.cusum_detector, so
    the same radius/ref_mult values used for the underlying detector
    should normally be passed here too).

    Parameters
    ----------
    x : np.ndarray
        The ORIGINAL signal (same scale the flags were computed on;
        pass the untransformed signal here even if the detector itself
        ran on a transformed copy, since "above the reference" should
        be judged on the scale that has physical meaning).
    flags : np.ndarray of bool
        The detector's own output, same length as x.
    radius, ref_mult : int
        Causal reference window span = radius*ref_mult.

    Returns
    -------
    np.ndarray of bool
        flags, with any True where x[i] <= its causal reference median
        set to False.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    flags = np.asarray(flags, dtype=bool).ravel()
    if x.shape != flags.shape:
        raise ValueError(f"x and flags must have the same shape, got {x.shape} vs {flags.shape}")
    span = radius * ref_mult
    meds = _causal_median(x, span)
    return flags & (x > meds)
