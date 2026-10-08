"""Online metrics for event detection.

Classical classification metrics assume a per-sample label. In event
detection the target is a set of *intervals* --- a fault that lasts a
few seconds, a change that starts at an unknown time --- and the
detector produces a stream of *alarms*. Two alarms close together are
one event; an alarm outside every labelled event is a false alarm.

The module implements three papers.

* **Range-based precision and recall** (Tatbul et al. 2018). Precision
  and recall are redefined over intervals: a prediction range can
  partially overlap a real one, and the overlap has a size, a position
  and a cardinality. The positional-bias functions are the four of
  Figure 2b (flat, front, back, middle), the existence reward is
  equation 5, the overlap reward is equation 6, the cardinality factor
  is equation 7. Recall uses the user-selected positional bias,
  precision uses flat (Section 5 of the paper).
* **Point-adjusted F1** (Xu et al. 2018; warning of Kim et al. 2021).
  The point-adjusted protocol marks a whole real anomaly segment as
  correctly detected as soon as one sample inside it is detected. Kim
  et al. show a *random* anomaly score reaches a high point-adjusted
  F1, so the metric cannot distinguish a detector from noise. Report
  it only next to range-based precision and recall.
* **The NAB score** (Lavin and Ahmad 2015). The scaled sigmoid of
  Figure 3, ``s(y) = 2 / (1 + exp(5 y)) - 1``, rewards early
  detections; the TP, FP and FN contributions follow Figure 3 and the
  equations 2 to 4 of the paper. The profile weights are parameters.

Ranges are treated as discrete sets of points: a window ``[start, end]``
sampled at period ``dt`` seconds contains
``round((end - start) / dt) + 1`` points. ``dt`` is a parameter of
:class:`EventMetrics` so an event of 0.5 s at ``dt = 0.01`` has 51
points, as required by the corrections.

References
----------
Tatbul, N., Lee, T. J., Zdonik, S., Alam, M., Gottschlich, J. (2018).
    Precision and recall for time series. In NeurIPS.
Lavin, A., Ahmad, S. (2015). Evaluating real-time anomaly detection
    algorithms --- the Numenta Anomaly Benchmark. In IEEE ICMLA.
Kim, S., Choi, K., Choi, H.-S., Lee, B., Yoon, S. (2021). Towards a
    rigorous evaluation of time-series anomaly detection. arXiv:2109.05257.
Xu, H. et al. (2018). Unsupervised anomaly detection via variational
    auto-encoder for seasonal KPIs in web applications. In WWW.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from dense_armor.roles import Classifier
from dense_armor.utility.metrics.base import Metric


@dataclass(frozen=True)
class EventWindow:
    """A labelled event interval.

    Attributes:
        start: start time of the event, in seconds.
        end: end time of the event, in seconds; ``end >= start``.

    Raises:
        ValueError: if ``end < start``.
    """

    start: float
    end: float

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(
                f"EventWindow: end {self.end} < start {self.start}"
            )

    def contains(self, t: float) -> bool:
        return self.start <= t <= self.end

    def n_points(self, dt: float) -> int:
        """Number of discrete points in the window at period ``dt``.

        A window ``[start, end]`` sampled at period ``dt`` covers the
        points ``start, start+dt, ..., end``; the count is
        ``round((end - start) / dt) + 1``.
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt}")
        return max(1, round((self.end - self.start) / dt) + 1)


def _overlap_seconds(a: EventWindow, b: EventWindow) -> float:
    lo = max(a.start, b.start)
    hi = min(a.end, b.end)
    return max(0.0, hi - lo)


def _has_overlap(a: EventWindow, b: EventWindow, dt: float) -> bool:
    """True if the two windows share at least one discrete point."""
    if _overlap_seconds(a, b) > 0.0:
        return True
    if a.n_points(dt) == 1 and b.contains(a.start):
        return True
    return b.n_points(dt) == 1 and a.contains(b.start)


class _PositionalBias:
    """Positional-bias functions of Tatbul et al. (2018), Figure 2b.

    ``i`` is the 1-indexed position of a point in the window of length
    ``n`` (``i = 1`` is the first point, ``i = n`` the last).
    """

    @staticmethod
    def flat(i: int, n: int) -> float:
        return 1.0

    @staticmethod
    def front(i: int, n: int) -> float:
        return float(n - i + 1)

    @staticmethod
    def back(i: int, n: int) -> float:
        return float(i)

    @staticmethod
    def middle(i: int, n: int) -> float:
        if i <= n / 2.0:
            return float(i)
        return float(n - i + 1)

    @staticmethod
    def get(name: str):
        table = {
            "flat": _PositionalBias.flat,
            "front": _PositionalBias.front,
            "back": _PositionalBias.back,
            "middle": _PositionalBias.middle,
        }
        if name not in table:
            raise ValueError(
                f"positional bias must be one of {sorted(table)}, "
                f"got {name!r}"
            )
        return table[name]


def _nab_sigmoid(y: float) -> float:
    """Scaled sigmoid of Lavin and Ahmad (2015), Figure 3."""
    return 2.0 / (1.0 + math.exp(5.0 * y)) - 1.0


class EventMetrics:
    """Collect the event metrics for a stream of alarms.

    Feed the labelled windows once, then each alarm as it happens.
    ``report()`` returns all the metrics in one dict.

    Args:
        events: the labelled event windows, as :class:`EventWindow` or
            as ``(start, end)`` pairs.
        dt: sample period in seconds, used to count the discrete points
            of a window. Default ``1.0`` (one point per second).
        alpha: existence-reward weight of Tatbul et al. (2018), in
            ``[0, 1]``. ``0`` ignores existence and rewards only overlap.
        positional_bias: one of ``"flat"``, ``"front"``, ``"back"``,
            ``"middle"``. Used for recall; precision always uses flat.
        cardinality: whether the cardinality factor of equation 7
            penalises a real event split across several predictions.
        normal_time_s: total normal time in seconds, used as the
            denominator of false alarms per hour. If ``None``, it is
            inferred from the first and last alarm timestamps minus the
            event durations.
        threshold_window: two alarms within this many seconds are the
            same detection. Default ``0.0``.
        a_tp: NAB weight of a true positive (Lavin and Ahmad 2015), in
            ``[0, 1]``; standard profile ``1.0``.
        a_fp: NAB weight of a false positive, in ``[-1, 0]``; standard
            profile ``-0.11``.
        a_fn: NAB weight of a missed window, in ``[-1, 0]``; standard
            profile ``-1.0``.

    Raises:
        ValueError: on a bad bias, an empty ``events``, or out-of-range
            weights.
    """

    def __init__(
        self,
        events: Iterable[EventWindow | tuple[float, float]],
        dt: float = 1.0,
        alpha: float = 0.0,
        positional_bias: str = "front",
        cardinality: bool = True,
        normal_time_s: float | None = None,
        threshold_window: float = 0.0,
        a_tp: float = 1.0,
        a_fp: float = -0.11,
        a_fn: float = -1.0,
    ) -> None:
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt}")
        self.events: list[EventWindow] = []
        for e in events:
            if isinstance(e, EventWindow):
                self.events.append(e)
            else:
                self.events.append(EventWindow(float(e[0]), float(e[1])))
        if not self.events:
            raise ValueError("EventMetrics needs at least one event window")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if not 0.0 <= a_tp <= 1.0:
            raise ValueError(f"a_tp must be in [0, 1], got {a_tp}")
        if not -1.0 <= a_fp <= 0.0:
            raise ValueError(f"a_fp must be in [-1, 0], got {a_fp}")
        if not -1.0 <= a_fn <= 0.0:
            raise ValueError(f"a_fn must be in [-1, 0], got {a_fn}")
        self.dt = dt
        self.alpha = alpha
        self.positional_bias = positional_bias
        self._bias_recall = _PositionalBias.get(positional_bias)
        self._bias_precision = _PositionalBias.get("flat")
        self.cardinality = cardinality
        self.normal_time_s = normal_time_s
        self.threshold_window = threshold_window
        self.a_tp = a_tp
        self.a_fp = a_fp
        self.a_fn = a_fn
        self.alarms_: list[float] = []

    def add_alarm(self, t: float) -> "EventMetrics":
        """Register an alarm at time ``t`` (seconds)."""
        self.alarms_.append(float(t))
        return self

    def _merged_alarms(self) -> list[EventWindow]:
        if not self.alarms_:
            return []
        times = sorted(self.alarms_)
        out: list[EventWindow] = []
        cur_s, cur_e = times[0], times[0]
        for t in times[1:]:
            if t - cur_e <= self.threshold_window:
                cur_e = t
            else:
                out.append(EventWindow(cur_s, cur_e))
                cur_s, cur_e = t, t
        out.append(EventWindow(cur_s, cur_e))
        return out

    def _is_false_alarm(self, alarm_t: float) -> bool:
        return not any(e.contains(alarm_t) for e in self.events)

    def _first_alarm_after(self, start: float) -> float | None:
        for t in sorted(self.alarms_):
            if t >= start:
                return t
        return None

    def detection_delays_s(self) -> list[float | None]:
        """Delay in seconds for every event, ``None`` if it was missed."""
        return [self._delay_for(e) for e in self.events]

    def _delay_for(self, e: EventWindow) -> float | None:
        t = self._first_alarm_after(e.start)
        if t is None or t > e.end:
            return None
        return t - e.start

    def missed_events(self) -> int:
        """Number of event windows with no alarm inside."""
        return sum(1 for d in self.detection_delays_s() if d is None)

    def false_alarms(self) -> int:
        """Number of alarms outside every event window."""
        return sum(1 for t in self.alarms_ if self._is_false_alarm(t))

    def false_alarms_per_hour(self) -> float:
        """False alarms divided by the observed normal time in hours."""
        fa = self.false_alarms()
        if self.normal_time_s is not None:
            hours = self.normal_time_s / 3600.0
        elif not self.alarms_:
            return 0.0
        else:
            t_lo = min(self.alarms_)
            t_hi = max(self.alarms_)
            span = t_hi - t_lo
            in_events = sum(
                max(0.0, min(e.end, t_hi) - max(e.start, t_lo))
                for e in self.events
            )
            hours = max(0.0, span - in_events) / 3600.0
        if hours <= 0.0:
            return 0.0
        return fa / hours

    def _existence_reward(self, e: EventWindow) -> float:
        return 1.0 if any(e.contains(t) for t in self.alarms_) else 0.0

    @staticmethod
    def _bias_positions(e: EventWindow, a: EventWindow, dt: float) -> list[int]:
        """1-indexed positions in ``e`` covered by the overlap with ``a``."""
        lo = max(e.start, a.start)
        hi = min(e.end, a.end)
        if hi < lo:
            return []
        n = e.n_points(dt)
        first = round((lo - e.start) / dt)
        last = round((hi - e.start) / dt)
        first = max(0, min(first, n - 1))
        last = max(0, min(last, n - 1))
        if last < first:
            return []
        return list(range(first + 1, last + 2))

    def _overlap_reward(
        self,
        e: EventWindow,
        alarms: list[EventWindow],
        bias_fn,
    ) -> float:
        n = e.n_points(self.dt)
        max_v = sum(bias_fn(i + 1, n) for i in range(n))
        if max_v <= 0:
            return 0.0
        my = 0.0
        seen: set[int] = set()
        for a in alarms:
            for i in self._bias_positions(e, a, self.dt):
                if i not in seen:
                    seen.add(i)
                    my += bias_fn(i, n)
        return my / max_v

    def _cardinality_factor(
        self, e: EventWindow, alarms: list[EventWindow]
    ) -> float:
        if not self.cardinality:
            return 1.0
        overlapping = sum(
            1 for a in alarms if _has_overlap(e, a, self.dt)
        )
        if overlapping <= 1:
            return 1.0
        return 1.0 / overlapping

    def range_recall(self) -> float:
        """Range-based recall of Tatbul et al. (2018), equation 3."""
        alarms = self._merged_alarms()
        if not self.events:
            return 0.0
        total = 0.0
        for e in self.events:
            ex = self._existence_reward(e)
            ov = self._overlap_reward(e, alarms, self._bias_recall)
            total += self.alpha * ex + (1.0 - self.alpha) * (
                self._cardinality_factor(e, alarms) * ov
            )
        return total / len(self.events)

    def range_precision(self) -> float:
        """Range-based precision of Tatbul et al. (2018), equation 8.

        Precision always uses the flat positional bias (Section 5).
        """
        alarms = self._merged_alarms()
        if not alarms:
            return 0.0
        total = 0.0
        for a in alarms:
            ov = 0.0
            for e in self.events:
                if _has_overlap(a, e, self.dt):
                    ov += self._overlap_reward(a, [e], self._bias_precision)
            card = 1.0
            if self.cardinality:
                n_overlap = sum(
                    1 for e in self.events if _has_overlap(a, e, self.dt)
                )
                if n_overlap > 1:
                    card = 1.0 / n_overlap
            total += card * ov
        return total / len(alarms)

    def range_f1(self) -> float:
        """Harmonic mean of :meth:`range_precision` and :meth:`range_recall`."""
        p = self.range_precision()
        r = self.range_recall()
        return 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0

    def _nab_y_tp(self, e: EventWindow, t_detect: float) -> float:
        """Relative position in ``[-1, 0]`` of a detection inside ``e``."""
        length = max(1e-12, e.end - e.start)
        return (t_detect - e.start) / length - 1.0

    def _nab_y_fp(self, t_fp: float) -> float | None:
        """Position of a false alarm after the preceding window.

        ``(t_fp - end) / length`` of the closest window that ends before
        ``t_fp``; ``None`` when no window precedes the alarm.
        """
        preceding: EventWindow | None = None
        for e in self.events:
            if e.end <= t_fp and (preceding is None or e.end > preceding.end):
                preceding = e
        if preceding is None:
            return None
        length = max(1e-12, preceding.end - preceding.start)
        return (t_fp - preceding.end) / length

    def nab_score(self) -> float:
        """Normalised NAB score of Lavin and Ahmad (2015), equation 4.

        The score uses the scaled sigmoid of Figure 3,
        ``s(y) = 2 / (1 + exp(5 y)) - 1``, with ``y = -1`` at the
        window start and ``y = 0`` at the window end. Each window
        contributes ``a_tp * s(y)`` for its earliest detection, or
        ``a_fn`` if missed; each false alarm contributes
        ``|a_fp| * s(y)`` with ``y`` measured after the preceding
        window, and ``-|a_fp|`` when no window precedes it (Figure 3).

        Returns:
            The score; 100 for the perfect detector, 0 for the null one.
        """
        raw = 0.0
        for e in self.events:
            hits = [t for t in self.alarms_ if e.contains(t)]
            if hits:
                first = min(hits)
                raw += self.a_tp * _nab_sigmoid(self._nab_y_tp(e, first))
            else:
                raw += self.a_fn
        for t in self.alarms_:
            if self._is_false_alarm(t):
                y = self._nab_y_fp(t)
                s_fp = -1.0 if y is None else _nab_sigmoid(y)
                raw += abs(self.a_fp) * s_fp
        s_null = self.a_fn * len(self.events)
        s_perfect = self.a_tp * _nab_sigmoid(-1.0) * len(self.events)
        denom = s_perfect - s_null
        if denom == 0.0:
            return 0.0
        return 100.0 * (raw - s_null) / denom

    def report(self) -> dict:
        """Return all event metrics in one dict.

        Returns:
            A dict with ``detection_delays_s`` (list, one per event,
            ``None`` for missed), ``mean_delay_s``, ``missed``,
            ``false_alarms``, ``false_alarms_per_hour``,
            ``range_precision``, ``range_recall``, ``range_f1``, and
            ``nab_score``.
        """
        delays = self.detection_delays_s()
        seen = [d for d in delays if d is not None]
        return {
            "detection_delays_s": delays,
            "mean_delay_s": sum(seen) / len(seen) if seen else None,
            "missed": self.missed_events(),
            "false_alarms": self.false_alarms(),
            "false_alarms_per_hour": self.false_alarms_per_hour(),
            "range_precision": self.range_precision(),
            "range_recall": self.range_recall(),
            "range_f1": self.range_f1(),
            "nab_score": self.nab_score(),
        }


class PointAdjustedF1(Metric):
    """Point-adjusted F1 (Xu et al. 2018).

    The protocol marks the whole real anomaly segment as correctly
    detected as soon as one sample inside it is detected. It was
    proposed on the basis that a single alert inside an anomaly is
    enough to act; Kim et al. (2021) showed that a *random* anomaly
    score reaches a high point-adjusted F1, so the metric cannot
    distinguish a real detector from noise.

    Use this metric only as a comparison, next to the range-based
    precision and recall of Tatbul et al. (2018).

    Args:
        positive: the label treated as the positive class.
        warn: whether to emit a ``UserWarning`` the first time the
            metric is scored. Default ``True``.

    References:
        Xu, H. et al. (2018). In WWW.
        Kim, S. et al. (2021). arXiv:2109.05257.
    """

    role_ = Classifier
    bigger_is_better = True

    def __init__(self, positive: Any = 1, warn: bool = True) -> None:
        super().__init__()
        self.positive = positive
        self.warn = warn
        self._warned = False
        self.y_true_: list[int] = []
        self.y_pred_: list[int] = []

    def _update(self, y_true: Any, y_pred: Any, w: float) -> None:
        self.y_true_.append(1 if y_true == self.positive else 0)
        self.y_pred_.append(1 if y_pred == self.positive else 0)

    def get(self) -> float:
        if not self._warned and self.warn:
            import warnings

            warnings.warn(
                "PointAdjustedF1 applies the point-adjusted protocol; a "
                "random anomaly score reaches a high value. Report it "
                "only next to the range-based PRF of Tatbul et al. 2018.",
                stacklevel=2,
            )
            self._warned = True
        n = len(self.y_true_)
        if n == 0:
            return 0.0
        segments: list[tuple[int, int]] = []
        i = 0
        while i < n:
            if self.y_true_[i] == 1:
                j = i
                while j < n and self.y_true_[j] == 1:
                    j += 1
                segments.append((i, j))
                i = j
            else:
                i += 1
        adjusted = list(self.y_pred_)
        for s, e in segments:
            if any(self.y_pred_[k] == 1 for k in range(s, e)):
                for k in range(s, e):
                    adjusted[k] = 1
        tp = sum(
            1 for a, b in zip(self.y_true_, adjusted) if a == 1 and b == 1
        )
        fp = sum(
            1 for a, b in zip(self.y_true_, adjusted) if a == 0 and b == 1
        )
        fn = sum(
            1 for a, b in zip(self.y_true_, adjusted) if a == 1 and b == 0
        )
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0
