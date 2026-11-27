"""
core/online/anomaly/streaming.py
================================
Promoted from Dense-Evolution-Discovery (Experiments 48-49) after
validation on two independent real physical domains (teleoperated SO-101
robotic arm, real human IMU UCI HAR) -- same promotion discipline already
used for `stable_frame_filter.py`.

MOTIVATION: a real robot runs at 30-100 Hz in real time, it cannot
wait for an already recorded array -- but reading `arbiter.classify_
segments` line by line (not by assumption) a real constraint appears:
the final spike/regime label looks `radius` points AHEAD at the end
of a deviating run (lines 178-186 of arbiter.py, the "persiste" check)
to decide whether the run settles or goes back -- that half
cannot be zero-latency streaming. A real robotic safety loop does not
need that distinction in real time anyway (it is a triage question,
solvable afterwards); it needs
"this point is deviating NOW" -- exactly the per-point computation of
classify_segments BEFORE the run-length logic.
Only this half is brought here.

CORRECTNESS: `streamDD`, fed one point at a
time, must reproduce exactly (bit for bit) the internal `deviante` array
of `classify_segments` -- verified on real data (not only
synthetic) in Dense-Evolution-Discovery before this promotion,
re-verified here in test/test_streaming.py on the real telemetry
already frozen in this repo (test/agent_v2/telemetry_v2_frozen.jsonl).

PERFORMANCE: a simple buffer (deque) that recomputes median/MAD at every
step -- O(span), not a two-heap structure O(log span). For the windows
already used throughout this project (10-100 points), measured at
~18.6 kHz sustainable on real hardware, over 180x the real rate of a
robotic control loop (30-100 Hz) -- no real performance trade-off to
justify with greater complexity.

MULTI-CHANNEL SUPPORT: every real robotic experiment in Dense-
Evolution-Discovery (the 6 joints of the LeRobot arm, the several axes of
the IMU) required a manual per-channel loop around functions
designed for a single 1D signal. `class_seg_multi` and
`MultistreamDD` apply the SAME per-channel logic
already validated, without requiring that loop -- ergonomics, not
a new algorithm. Each channel keeps its own reference window and its own
scale independent by construction (a fast joint and an almost still one
have very different natural noise scales; a shared scale would
desensitize the slow one or give false alarms on the active one).
"""
from collections import deque

import numpy as np

from dense_armor._dtype import as_default_array, get_default_dtype
from dense_armor.core.online.protect.arbiter import (
    _robust_center_scale,
    classify_segments,
)


def _feature_value(x, feature):
    """Finite float of ``feature`` (smallest key when ``None``), else ``None``."""
    try:
        v = float(x[feature if feature is not None else min(x)])
    except (KeyError, TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


class streamDD:
    """Causal zero-latency deviation flag, one point at a time --
    same causal window (radius*ref_mult), same robust median/MAD,
    same degenerate-baseline rule as classify_segments'
    (arbiter.py), NOT the spike/regime distinction (which requires a
    look ahead, it stays a batch/offline question by design, not
    an oversight).

    Args:
        radius: half-width unit; the causal window holds `radius * ref_mult` values.
        ref_mult: window length multiplier.
        n_sigmas: flag when the deviation is above this many robust sigmas.
        eps: numerical guard for a degenerate scale.

    Examples:
    >>> det = streamDD(radius=5, ref_mult=2, n_sigmas=3.0)
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> stream = list(rng.normal(0, 1, 20)) + [50.0]
    >>> flags = [det.update(x) for x in stream]
    >>> flags[-1]
    True
    """

    def __init__(self, radius: int = 10, ref_mult: int = 3, n_sigmas: float = 3.0, eps: float = 1e-9):
        self.radius = radius
        self.ref_mult = ref_mult
        self.n_sigmas = n_sigmas
        self.eps = eps
        self._span = radius * ref_mult
        self._buffer = deque(maxlen=self._span)
        self.last_deviation = 0.0  # informative, not part of the correctness contract

    def update(self, x: float) -> bool:
        """Feeds a new point. Returns True if it DEVIATES from the
        window of the strictly PREVIOUS points (never including
        itself) -- same semantics as `_window_causal` in arbiter.py.
        A non-finite value does not enter the buffer and is not deviating."""
        if not np.isfinite(x):
            self.last_deviation = 0.0
            return False
        if len(self._buffer) < 4:
            self._buffer.append(x)
            self.last_deviation = 0.0
            return False

        med, scala = _robust_center_scale(np.array(self._buffer))
        scarto = abs(x - med)
        if scala < self.eps:
            self.last_deviation = scarto
            deviante = scarto > self.eps
        else:
            self.last_deviation = scarto / scala
            deviante = self.last_deviation > self.n_sigmas

        self._buffer.append(x)
        return deviante


class MultistreamDD:
    """N independent instances of `streamDD`, one per
    channel -- the flag of each channel is computed exactly as if
    `streamDD` ran on that channel alone (the
    channels never influence each other).

    Args:
        n_channels: number of independent channels.
        radius: half-width unit; each channel's window holds `radius * ref_mult` values.
        ref_mult: window length multiplier.
        n_sigmas: per-channel flag threshold in robust-sigma units.
        eps: numerical guard for a degenerate scale.

    Examples:
    >>> import numpy as np
    >>> det = MultistreamDD(n_channels=2, radius=5, ref_mult=2, n_sigmas=3.0)
    >>> _ = [det.update(r) for r in np.zeros((10, 2))]
    >>> det.update([50.0, 0.0]).tolist()
    [True, False]
    """

    def __init__(self, n_channels: int, radius: int = 10, ref_mult: int = 3,
                 n_sigmas: float = 3.0, eps: float = 1e-9):
        self.n_channels = n_channels
        self._detectors: list[streamDD] = [
            streamDD(radius=radius, ref_mult=ref_mult, n_sigmas=n_sigmas, eps=eps)
            for _ in range(n_channels)
        ]

    def update(self, x_vec) -> np.ndarray:
        """x_vec: array-like of `n_channels` numbers. Returns a bool array (n_channels,).

        Raises:
            ValueError: if `x_vec` does not have `n_channels` values.
        """
        x_vec = as_default_array(x_vec).ravel()
        if x_vec.size != self.n_channels:
            raise ValueError(f"expected {self.n_channels} channels, got {x_vec.size}")
        return np.array([det.update(float(v)) for det, v in zip(self._detectors, x_vec)], dtype=bool)


def class_seg_multi(
    X: np.ndarray, radius: int = 10, ref_mult: int = 3,
    n_sigmas: float = 3.0, spike_run_max: int = 2, eps: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply `classify_segments` (arbiter.py) independently to each column.

    Exactly the same result per channel as calling it individually, only
    stacked instead of requiring a manual loop from the caller.

    Args:
        X: array of shape (n_samples, n_channels).
        radius: half-width unit; the causal window holds `radius * ref_mult` values.
        ref_mult: window length multiplier.
        n_sigmas: flag threshold in robust-sigma units.
        spike_run_max: maximum run length counted as a spike.
        eps: numerical guard for a degenerate scale.

    Returns:
        A tuple `(labels, deviation, uncertainty)`, each of shape
        `(n_samples, n_channels)`.

    Examples:
        >>> import numpy as np
        >>> X = np.zeros((5, 2))
        >>> e, d, u = class_seg_multi(X, radius=2, ref_mult=2)
        >>> e.shape == d.shape == u.shape == (5, 2)
        True
    """
    X = as_default_array(X)
    n, c = X.shape
    etichette = np.empty((n, c), dtype=object)
    deviazione = np.zeros((n, c), dtype=get_default_dtype())
    incertezza = np.zeros((n, c), dtype=get_default_dtype())
    for j in range(c):
        e, d, u = classify_segments(X[:, j], radius=radius, ref_mult=ref_mult,
                                     n_sigmas=n_sigmas, spike_run_max=spike_run_max, eps=eps)
        etichette[:, j] = e
        deviazione[:, j] = d
        incertezza[:, j] = u
    return etichette, deviazione, incertezza
