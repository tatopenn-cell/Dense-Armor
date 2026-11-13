"""Dynamics: a compressor, a lookahead limiter and a soft clipper with oversampling."""

import math
from collections import deque
from typing import Any

import numpy as np

from dense_armor.core.audio._state import StatefulRoot

_FLOOR = 1e-10


def _smoothing_coefficient(time_s: float, sr: float) -> float:
    return math.exp(-1.0 / (time_s * sr)) if time_s > 0 else 0.0


def gain_curve(
    x_db: Any, threshold_db: float, ratio: float, knee_db: float = 0.0
) -> np.ndarray:
    """Static curve of a compressor: the output level in dB for an input level in dB.

    Below the threshold ``T`` the level is unchanged; above it, every ``ratio`` dB of input give
    1 dB of output. With a knee of width ``W`` the transition is a parabola over
    ``T - W/2 .. T + W/2``::

        y = x                                            if x - T < -W/2
        y = x + (1/R - 1) (x - T + W/2)**2 / (2 W)       if |x - T| <= W/2
        y = T + (x - T) / R                              if x - T > W/2

    The curve and its slope are continuous at both ends of the knee: at ``T - W/2`` the parabola
    gives ``x`` with slope 1, and at ``T + W/2`` it gives ``T + W/(2R)`` with slope ``1/R``. An
    infinite ratio is a hard limit at ``T``.

    Args:
        x_db: input level in dB, a number or an array.
        threshold_db: the threshold ``T`` in dB.
        ratio: the ratio ``R``, at least 1 (``inf`` allowed).
        knee_db: the width ``W`` of the knee in dB, at least 0.

    Returns:
        The output level in dB, the shape of ``x_db``.

    Raises:
        ValueError: if ``ratio < 1`` or ``knee_db < 0``.

    Example:
        >>> gain_curve([-30.0, -10.0], -20.0, 4.0).tolist()
        [-30.0, -17.5]

    References:
        Giannoulis, Massberg and Reiss (2012), Digital Dynamic Range Compressor Design - A
        Tutorial and Analysis, J. Audio Eng. Soc. 60(6), 399-408. Bibliographic only (not in
        ``papers/``): the formula was derived and checked by hand for the continuity of the
        curve and of its slope at both ends of the knee, and the values of the example, and the
        ten points of ``test_the_static_curve_at_ten_points``, were computed by hand.
    """
    if ratio < 1.0 or knee_db < 0.0:
        raise ValueError("ratio must be >= 1 and knee_db >= 0")
    x = np.asarray(x_db, dtype=float)
    inv = 0.0 if math.isinf(ratio) else 1.0 / ratio
    over = x - threshold_db
    above = threshold_db + over * inv
    if knee_db == 0.0:
        return np.where(over <= 0.0, x, above)
    knee = x + (inv - 1.0) * (over + knee_db / 2.0) ** 2 / (2.0 * knee_db)
    return np.where(
        2.0 * over < -knee_db, x, np.where(2.0 * np.abs(over) <= knee_db, knee, above)
    )


class Compressor(StatefulRoot):
    """Feed-forward compressor with a soft knee and a smoothed gain.

    Each sample: the level ``x_dB = 20 log10 |x|``; the gain reduction the static curve asks,
    ``g = x_dB - gain_curve(x_dB) >= 0``; the smoothed reduction ``s``, which moves towards ``g``
    with a one-pole filter of time ``attack_s`` while ``g`` is above it and ``release_s`` while it
    is below; the output ``x * 10 ** ((makeup_db - s) / 20)``. A time of 0 follows at once. The
    last gain reduction is in ``gain_reduction_db`` and the output in ``out``.

    Args:
        threshold_db: threshold in dB.
        ratio: compression ratio, at least 1.
        attack_s: attack time in seconds, at least 0.
        release_s: release time in seconds, at least 0.
        knee_db: knee width in dB, at least 0.
        makeup_db: gain added after the compression, in dB.
        sr: sampling rate in hertz, positive.

    Raises:
        ValueError: if a parameter is out of range.

    Example:
        >>> c = Compressor(threshold_db=-20.0, ratio=4.0, attack_s=0.0, release_s=0.0, sr=1000.0)
        >>> y = c.update_many(np.full(3, 10 ** (-8 / 20)))
        >>> round(c.gain_reduction_db, 6)
        9.0

    References:
        Giannoulis, Massberg and Reiss (2012), J. Audio Eng. Soc. 60(6), 399-408: a feed-forward
        compressor whose gain computer is the static curve of :func:`gain_curve` and whose gain is
        smoothed by an attack and a release. Bibliographic only (not in ``papers/``): the
        smoothing is verified against ``scipy.signal.lfilter`` (see
        ``test_the_smoothing_is_a_one_pole_filter``) and the static gain against a hand
        computation (a level 12 dB above a threshold with ratio 4 comes out 3 dB above it, a
        gain reduction of 9 dB).
    """

    budget_s: float = 2e-4
    memory_class: str = "O(1)"
    _state_attrs = ("gain_reduction_db", "out")

    def __init__(
        self,
        threshold_db: float = -20.0,
        ratio: float = 4.0,
        attack_s: float = 0.01,
        release_s: float = 0.1,
        knee_db: float = 0.0,
        makeup_db: float = 0.0,
        sr: float = 44100.0,
    ) -> None:
        if ratio < 1.0 or knee_db < 0.0 or attack_s < 0.0 or release_s < 0.0 or sr <= 0:
            raise ValueError(
                "ratio must be >= 1, knee_db, attack_s and release_s >= 0, sr > 0"
            )
        self.threshold_db = float(threshold_db)
        self.ratio = float(ratio)
        self.attack_s = float(attack_s)
        self.release_s = float(release_s)
        self.knee_db = float(knee_db)
        self.makeup_db = float(makeup_db)
        self.sr = float(sr)
        self.gain_reduction_db = 0.0
        self.out = 0.0

    def update(self, x: float) -> "Compressor":
        """Compress one sample; the result is ``out``."""
        a_att = _smoothing_coefficient(self.attack_s, self.sr)
        a_rel = _smoothing_coefficient(self.release_s, self.sr)
        x = float(x)
        level = 20.0 * math.log10(max(abs(x), _FLOOR))
        wanted = level - float(
            gain_curve(level, self.threshold_db, self.ratio, self.knee_db)
        )
        s = self.gain_reduction_db
        a = a_att if wanted > s else a_rel
        s = a * s + (1.0 - a) * wanted
        self.gain_reduction_db = s
        self.out = x * 10.0 ** ((self.makeup_db - s) / 20.0)
        return self

    def update_many(self, values: Any) -> np.ndarray:
        """Compress a block; returns the output samples."""
        v = np.asarray(values, dtype=float)
        out = np.empty(len(v))
        for i, x in enumerate(v):
            out[i] = self.update(x).out
        return out


class Limiter(StatefulRoot):
    """Brickwall limiter with lookahead: no output sample is above ``ceiling_db``.

    The gain reduction is the one of :func:`gain_curve` with an infinite ratio, as a gain: a
    sample of level ``|x|`` above the ceiling ``C`` asks for ``C / |x|``. With ``L`` the lookahead
    in samples, each step is:

    1. the asked gain ``r[n]``;
    2. ``s[n]``, the smallest of ``r`` over the last ``L + 1`` samples;
    3. ``y[n]``, which follows ``s`` down at once and up with a one-pole filter of time
       ``release_s``;
    4. ``g[n]``, the mean of ``y`` over the last ``L + 1`` samples;
    5. the output is ``x[n - L] * g[n]``.

    Every term of the mean in step 4 is at most ``r[n - L]``, so ``g[n] <= r[n - L]`` and the output
    is at most ``C`` in modulus, with no exception and no overshoot. The gain falls over ``L``
    samples before the loud sample reaches the output, and the output is delayed by ``L`` samples
    (``latency``). The last reduction is in ``gain_reduction_db`` and the output in ``out``.

    Choice stated: the attack is the lookahead window, a minimum followed by a mean, and not an
    exponential attack; an exponential one cannot give the guarantee above. The release is
    exponential.

    Args:
        ceiling_db: the ceiling in dB, at most 0 for a signal in ``[-1, 1]``.
        lookahead_s: lookahead in seconds, positive.
        release_s: release time in seconds, at least 0.
        sr: sampling rate in hertz, positive.

    Raises:
        ValueError: if ``lookahead_s`` or ``sr`` is not positive, or ``release_s`` is negative.

    Example:
        >>> lim = Limiter(ceiling_db=-6.0, lookahead_s=0.001, sr=1000.0)
        >>> y = lim.update_many(np.array([0.0] * 5 + [1.0] * 5 + [0.0] * 5))
        >>> bool(np.abs(y).max() <= 10 ** (-6 / 20) + 1e-12)
        True

    References:
        The lookahead limiter of the dynamics-processing literature, with its gain computer
        from :func:`gain_curve`: Giannoulis, Massberg and Reiss (2012), J. Audio Eng. Soc.
        60(6), 399-408, and Zolzer (ed.) (2011), DAFX: Digital Audio Effects, 2nd ed., Wiley,
        ISBN 978-0-470-66599-2. Bibliographic only (not in ``papers/``): the guarantee is a
        short proof (above) and a test on random signals with bursts (see
        ``test_no_sample_is_above_the_ceiling``).
    """

    budget_s: float = 2e-4
    memory_class: str = "O(window)"
    _state_attrs = (
        "_delay",
        "_minq",
        "_avg",
        "_total",
        "_y",
        "_n",
        "gain_reduction_db",
        "out",
    )

    def __init__(
        self,
        ceiling_db: float = -0.1,
        lookahead_s: float = 0.005,
        release_s: float = 0.05,
        sr: float = 44100.0,
    ) -> None:
        if lookahead_s <= 0.0 or sr <= 0.0 or release_s < 0.0:
            raise ValueError("lookahead_s and sr must be > 0 and release_s >= 0")
        self.ceiling_db = float(ceiling_db)
        self.lookahead_s = float(lookahead_s)
        self.release_s = float(release_s)
        self.sr = float(sr)
        n = self.latency
        self._delay: deque = deque([0.0] * n, maxlen=n)
        self._minq: deque = deque()
        self._avg: deque = deque([1.0] * (n + 1), maxlen=n + 1)
        self._total = float(n + 1)
        self._y = 1.0
        self._n = 0
        self.gain_reduction_db = 0.0
        self.out = 0.0

    @property
    def latency(self) -> int:
        """The delay of the output, in samples: the lookahead."""
        return max(1, round(self.lookahead_s * self.sr))

    def update(self, x: float) -> "Limiter":
        """Limit one sample; the result is ``out``, delayed by ``latency`` samples."""
        n_look = self.latency
        ceiling = 10.0 ** (self.ceiling_db / 20.0)
        a_rel = _smoothing_coefficient(self.release_s, self.sr)
        x = float(x)
        ax = abs(x)
        r = ceiling / ax if ax > ceiling else 1.0
        q = self._minq
        while q and q[-1][1] >= r:
            q.pop()
        q.append((self._n, r))
        while q[0][0] < self._n - n_look:
            q.popleft()
        s = q[0][1]
        y = s if s < self._y else a_rel * self._y + (1.0 - a_rel) * s
        self._y = y
        self._total += y - self._avg[0]
        self._avg.append(y)
        g = self._total / (n_look + 1)
        delayed = self._delay[0]
        self._delay.append(x)
        self._n += 1
        self.gain_reduction_db = -20.0 * math.log10(g)
        self.out = delayed * g
        return self

    def update_many(self, values: Any) -> np.ndarray:
        """Limit a block; returns the output samples, delayed by ``latency``."""
        v = np.asarray(values, dtype=float)
        out = np.empty(len(v))
        for i, x in enumerate(v):
            out[i] = self.update(x).out
        return out
