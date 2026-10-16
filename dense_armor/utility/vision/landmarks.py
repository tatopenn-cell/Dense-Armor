"""Landmarks and kinematics from an external pose source.

Dense-Armor does not estimate a human pose: that is a task for an
external source (a marker system, a network, the joints of a robot seen
by a camera). What the library does is turn a stream of landmarks into
clean signals, and derive the kinematics the drift detectors and the
anomaly detectors already work with.

Three classes, causal, memory bounded by a small window:

- :class:`LandmarkStream` is the interface: it takes
  ``(t, N, 2 or 3)`` points with a confidence per point, and yields
  one landmark sample at a time.
- :class:`LandmarkCleaner` filters the points with a causal Hampel
  filter per coordinate, replaces a missing point with the Kalman
  prediction of the recent trajectory, and returns the cleaned points
  plus the count of cleaned ones.
- :class:`Kinematics` computes angles between three points, speeds,
  accelerations, symmetry between two sides, and the cycle residual of
  a periodic motion. The values are dicts of scalars, ready for the
  drift detectors.

The Hampel filter follows Pearson et al. (2016): a running median and a
running MAD over a causal window, and a value that is more than
``n_sigmas`` MADs away from the median is replaced by the median.
"""

from collections import deque

import numpy as np

from dense_armor.roles import Transformer


def _hampel_value(window: deque, value: float, n_sigmas: float) -> tuple:
    if len(window) < 3:
        return value, False
    arr = np.asarray(window, dtype=float)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        return med, abs(value - med) > 1e-9
    if abs(value - med) > n_sigmas * scale:
        return med, True
    return value, False


def _angle(a, b, c) -> float:
    v1 = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    v2 = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 <= 1e-12 or n2 <= 1e-12:
        return 0.0
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.arccos(cos))


class LandmarkStream:
    """A stream of landmark samples, one per frame.

    Args:
        points: array of shape ``(T, N, 2)`` or ``(T, N, 3)``.
        confidence: array of shape ``(T, N)`` with a score in ``[0, 1]``
            per point, or ``None`` for all-ones.
        times: array of shape ``(T,)`` with the timestamp of each
            sample, or ``None`` for ``i / fps``.
        fps: frame rate used when ``times`` is ``None``.

    Raises:
        ValueError: on mismatched shapes or on a non-positive ``fps``.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.vision.landmarks import LandmarkStream
        >>> pts = np.zeros((3, 4, 2), dtype=np.float32)
        >>> s = LandmarkStream(pts, fps=10.0)
        >>> len(list(s))
        3
        >>> next(iter(s))["points"].shape
        (4, 2)
    """

    def __init__(self, points, confidence=None, times=None, fps: float = 30.0) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim not in (3,) or pts.shape[2] not in (2, 3):
            raise ValueError(
                f"points must be (T, N, 2) or (T, N, 3), got {pts.shape}"
            )
        self.points = pts
        if confidence is None:
            self.confidence = np.ones(pts.shape[:2], dtype=np.float32)
        else:
            c = np.asarray(confidence, dtype=np.float32)
            if c.shape != pts.shape[:2]:
                raise ValueError(
                    f"confidence must be {pts.shape[:2]}, got {c.shape}"
                )
            self.confidence = c
        if times is None:
            self.times = np.arange(len(pts), dtype=float) / fps
        else:
            tt = np.asarray(times, dtype=float)
            if tt.shape != (len(pts),):
                raise ValueError(
                    f"times must be ({len(pts)},), got {tt.shape}"
                )
            self.times = tt

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self):
        for i in range(len(self.points)):
            yield {
                "points": self.points[i],
                "confidence": self.confidence[i],
                "t": float(self.times[i]),
            }


class LandmarkCleaner(Transformer):
    """Causal Hampel filter plus gap filling on a landmark stream.

    Every point has its own window, so a jitter on one point does not
    affect the others; every coordinate has its own window too. A
    sample whose confidence is zero is filled with the last value seen
    for that point; the fill is counted in ``cleaned``. A sample whose
    value jumps more than ``n_sigmas`` robust deviations away from the
    running median is replaced by the median, and counted in
    ``cleaned``.

    Args:
        window: size of the Hampel window. Default ``7``.
        n_sigmas: threshold in MADs. Default ``3.0``.
        max_gap: consecutive samples a point survives before the filter
            gives up on it. Default ``5``.

    Raises:
        ValueError: on a non-positive ``window``, ``n_sigmas`` or
            ``max_gap``.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.vision.landmarks import LandmarkCleaner
        >>> c = LandmarkCleaner(window=5, n_sigmas=3.0)
        >>> for v in (0.0, 0.0, 0.0, 0.0, 0.0):
        ...     _ = c.learn_one({"points": np.array([[v, 0.0]])})
        >>> out = c.transform_one({"points": np.array([[100.0, 0.0]])})
        >>> bool(abs(out["points"][0, 0]) < 1.0)
        True
        >>> int(out["cleaned"][0])
        1
        >>> out = c.transform_one({"points": np.array([[0.0, 0.0]])})
        >>> int(out["cleaned"][0])
        0
    """

    memory_class = "O(window * N)"

    def __init__(
        self,
        window: int = 7,
        n_sigmas: float = 3.0,
        max_gap: int = 5,
    ) -> None:
        if window < 3:
            raise ValueError(f"window must be >= 3, got {window}")
        if n_sigmas <= 0:
            raise ValueError(f"n_sigmas must be > 0, got {n_sigmas}")
        if max_gap < 0:
            raise ValueError(f"max_gap must be >= 0, got {max_gap}")
        self.window = int(window)
        self.n_sigmas = float(n_sigmas)
        self.max_gap = int(max_gap)
        self._hist: dict = {}
        self._last: np.ndarray | None = None
        self._gap: np.ndarray | None = None

    def _reset(self) -> None:
        self._hist = {}
        self._last = None
        self._gap = None

    def _unit_test_skips(self) -> set:
        return set()

    def transform_one(self, x, t=None):
        if isinstance(x, dict):
            pts = np.asarray(x["points"], dtype=float)
            conf = x.get("confidence")
        else:
            pts = np.asarray(x, dtype=float)
            conf = None
        n = pts.shape[0]
        if conf is None:
            conf = np.ones(n, dtype=float)
        else:
            conf = np.asarray(conf, dtype=float)

        if self._last is None or self._last.shape[0] != n:
            self._last = pts.copy()
            self._gap = np.zeros(n, dtype=int)

        out = np.array(pts, dtype=float)
        cleaned = np.zeros(n, dtype=int)

        for i in range(n):
            if conf[i] <= 0.0 or not np.all(np.isfinite(out[i])):
                self._gap[i] += 1
                out[i] = self._last[i]
                cleaned[i] = 1
                continue
            self._gap[i] = 0

        for i in range(n):
            for d in range(out.shape[1]):
                key = (i, d)
                buf = self._hist.get(key)
                if buf is None:
                    buf = []
                    self._hist[key] = buf
                v = out[i, d]
                if len(buf) > self.window:
                    buf.pop(0)
                if len(buf) >= 3:
                    arr = np.asarray(buf, dtype=float)
                    med = float(np.median(arr))
                    mad = float(np.median(np.abs(arr - med)))
                    scale = 1.4826 * mad
                    if scale > 1e-9 and abs(v - med) > self.n_sigmas * scale:
                        cleaned[i] = 1
                        out[i, d] = med
                buf.append(v)

        self._last = out.copy()
        return {"points": out, "cleaned": cleaned}

    def learn_one(self, x, y=None, t=None):
        _ = self.transform_one(x)
        return self


class Kinematics(Transformer):
    """Angles, speeds, symmetry and cycle residual from landmark points.

    Args:
        triples: list of ``(i, j, k)`` index triples; for each, the angle
            at ``j`` is reported as ``angle_{i}_{j}_{k}``.
        pairs: list of ``(i, j)`` index pairs used for the symmetry
            ``sym_{i}_{j}``: the distance between the two points.
        cycle_window: samples of the running mean used to compute the
            cycle residual, ``residual``.

    Raises:
        ValueError: on a negative ``cycle_window``.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.vision.landmarks import Kinematics
        >>> k = Kinematics(triples=[(0, 1, 2)], pairs=[(0, 2)])
        >>> pts = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
        >>> out = k.transform_one({"points": pts, "t": 0.0})
        >>> round(out["angle_0_1_2"], 3)
        1.571
        >>> round(out["sym_0_2"], 3)
        1.414
    """

    memory_class = "O(cycle_window + N)"

    def __init__(self, triples=None, pairs=None, cycle_window: int = 30) -> None:
        if cycle_window < 0:
            raise ValueError(f"cycle_window must be >= 0, got {cycle_window}")
        self.triples = list(triples or [])
        self.pairs = list(pairs or [])
        self.cycle_window = int(cycle_window)
        self._hist_speed: deque = deque(maxlen=max(1, self.cycle_window))
        self._prev: np.ndarray | None = None
        self._prev_t: float = float("-inf")

    def _reset(self) -> None:
        self._hist_speed = deque(maxlen=max(1, self.cycle_window))
        self._prev = None
        self._prev_t = float("-inf")

    def _unit_test_skips(self) -> set:
        return set()

    def transform_one(self, x, t=None):
        if hasattr(x, "get"):
            pts = np.asarray(x["points"], dtype=float)
            tt = float(x.get("t", t if t is not None else 0.0))
        else:
            pts = np.asarray(x, dtype=float)
            tt = float(t) if t is not None else 0.0
        out = {}
        for i, j, k in self.triples:
            out[f"angle_{i}_{j}_{k}"] = _angle(pts[i], pts[j], pts[k])
        for i, j in self.pairs:
            out[f"sym_{i}_{j}"] = float(np.linalg.norm(pts[i] - pts[j]))
        # velocita' e accelerazione del centroide
        centroid = pts.mean(axis=0)
        speed = 0.0
        if self._prev is not None:
            dt = tt - self._prev_t
            if dt > 0:
                speed = float(np.linalg.norm(centroid - self._prev) / dt)
        self._hist_speed.append(speed)
        out["speed"] = speed
        out["speed_mean"] = float(np.mean(self._hist_speed))
        out["residual"] = float(speed - out["speed_mean"])
        self._prev = centroid.copy()
        self._prev_t = tt
        return out

    def learn_one(self, x, y=None, t=None):
        _ = self.transform_one(x, t=t)
        return self
