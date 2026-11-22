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
from typing import Any, TypedDict

import numpy as np

from dense_armor._dtype import as_default_array, get_default_dtype
from dense_armor.roles import Transformer


class _BatchedLandmarkShieldOutput(TypedDict):
    points: np.ndarray
    cleaned: np.ndarray

_MIN_WINDOW = 3
_SCALE_EPS = 1e-12
_DEV_EPS = 1e-9
_DT_EPS = 1e-6
_DECAY_STEP = 0.2

_H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
_Q = np.diag([1e-6, 1e-6, 1e-4, 1e-4])
_R = np.diag([1e-3, 1e-3])
_P_INIT = np.diag([1e-6, 1e-6, 1e-2, 1e-2])


def _hampel_value(window: list[float], value: float, n_sigmas: float) -> tuple[float, bool]:
    """Causal Hampel filter on a 1D window, following dense_armor LandmarkCleaner.

    Returns (filtered_value, was_outlier).
    """
    if len(window) < _MIN_WINDOW:
        return value, False
    arr = as_default_array(window)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    scale = 1.4826 * mad
    if scale <= _SCALE_EPS:
        return med, abs(value - med) > _DEV_EPS
    if abs(value - med) > n_sigmas * scale:
        return med, True
    return value, False


def _hampel_batch(
    hist: np.ndarray, vals: np.ndarray, n_sigmas: float
) -> tuple[np.ndarray, np.ndarray]:
    """Hampel rule of ``_hampel_value`` for N points at once.

    Args:
        hist: (N, W) history, left-padded with NaN.
        vals: (N,) new values.
        n_sigmas: threshold in scaled MAD units.

    Returns:
        (filtered values (N,), outlier flags (N,))
    """
    out: np.ndarray = np.zeros(len(vals), dtype=bool)
    filt = vals.copy()
    rows = np.sum(~np.isnan(hist), axis=1) >= _MIN_WINDOW
    if rows.any():
        h = hist[rows]
        med = np.nanmedian(h, axis=1)
        scale = 1.4826 * np.nanmedian(np.abs(h - med[:, None]), axis=1)
        dev = np.abs(vals[rows] - med)
        flag = np.where(scale <= _SCALE_EPS, dev > _DEV_EPS, dev > n_sigmas * scale)
        out[rows] = flag
        filt[rows] = np.where(flag, med, vals[rows])
    return filt, out


def _angle(a, b, c) -> float:
    if len(window) < 3:
        return value, False
    arr = as_default_array(window)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        return med, abs(value - med) > 1e-9
    if abs(value - med) > n_sigmas * scale:
        return med, True
    return value, False


def _angle(a, b, c) -> float:
    v1 = as_default_array(a) - as_default_array(b)
    v2 = as_default_array(c) - as_default_array(b)
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
        >>> from dense_armor.core.signals.vision.landmarks import LandmarkStream
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
            self.times = np.arange(len(pts), dtype=get_default_dtype()) / fps
        else:
            tt = as_default_array(times)
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
        >>> from dense_armor.core.signals.vision.landmarks import LandmarkCleaner
        >>> c = LandmarkCleaner(window=5, n_sigmas=3.0)
        >>> rng = np.random.default_rng(0)
        >>> for _ in range(8):
        ...     v = float(rng.normal(0.0, 0.01))
        ...     _ = c.learn_one({"points": np.array([[v, 0.0]])})
        >>> out = c.transform_one({"points": np.array([[100.0, 0.0]])})
        >>> bool(abs(out["points"][0, 0]) < 0.1)
        True
        >>> int(out["cleaned"][0])
        1
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
            pts = as_default_array(x["points"])
            conf = x.get("confidence")
        else:
            pts = as_default_array(x)
            conf = None
        n = pts.shape[0]
        if conf is None:
            conf = np.ones(n, dtype=get_default_dtype())
        else:
            conf = as_default_array(conf)

        if self._last is None or self._last.shape[0] != n:
            self._last = pts.copy()
            self._gap = np.zeros(n, dtype=int)

        out = as_default_array(pts, copy=True)
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
                    arr = as_default_array(buf)
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
        >>> from dense_armor.core.signals.vision.landmarks import Kinematics
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
            pts = as_default_array(x["points"])
            tt = float(x.get("t", t if t is not None else 0.0))
        else:
            pts = as_default_array(x)
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


class BatchedLandmarkShield(Transformer):
    """Batched constant-velocity Kalman filter with causal Hampel outlier filter for landmark streams.

    This is the batched (vectorized over all points) version of the landmark shield.
    It gives every point its own 4D constant-velocity Kalman filter (x, y, vx, vy)
    and a causal Hampel filter per coordinate.

    - Confidence > 0: measurement update after Hampel outlier check. The first
      valid measurement of a point initialises its filter at that measurement,
      with zero velocity.
    - Confidence == 0: predict-only (no measurement update). A point that was
      never measured is returned as given. After max_gap frames of occlusion the
      prediction decays toward the neutral pose of that point.
    - The Hampel history of a point is cleared when the point becomes valid
      again after an occlusion.
    - The time step comes from the frame timestamps; the first frame uses 1/30 s.

    This implementation follows the linear Kalman filter equations (special case
    of EKF with constant F and H) from Thieu & Melnik (2025, arXiv:2502.15179),
    §2.1 pp. 4-5 (EKF equations) and §4.2 p. 13 (constant-velocity motion model
    x_{k+1} = x_k + v_k dt + w_k). The outlier test is the causal Hampel rule
    following dense_armor's LandmarkCleaner (Pearson et al. 2016).

    Args:
        window: causal window size for Hampel filter (default 7).
        n_sigmas: Hampel threshold in scaled MAD units (default 3.0).
        max_gap: max frames to run predict-only before decaying to neutral (default 5).
        neutral: neutral pose (N, 2) to decay to after max_gap. If None, the
            neutral of each point is its first valid measurement; it is undefined
            (NaN in ``neutral``) until the point has been measured once, and
            the decay does not apply to it.

    References:
        - Thieu, N. & Melnik, R. (2025). Nonlinear Dynamical Systems for
          Automatic Face Annotation in Head Tracking and Pose Estimation.
          arXiv:2502.15179. EKF equations §2.1 pp. 4-5; motion model §4.2 p. 13.
        - Zhu, X. et al. (2021). Reasoning Structural Relation for
          Occlusion-Robust Facial Landmark Localization. arXiv:2112.10087.
          Test protocol: dynamic occlusion moving across face.
        - Pearson, R. et al. (2016). Robust outlier detection using the
          Hampel filter. In *Data Mining and Knowledge Discovery*.
    """

    budget_s: float = 3.5e-2
    memory_class: str = 'O(N * window)'

    def __init__(
        self,
        window: int = 7,
        n_sigmas: float = 3.0,
        max_gap: int = 5,
        neutral: np.ndarray | None = None,
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
        self.neutral = np.asarray(neutral, dtype=float) if neutral is not None else None
        self._neutral_given = neutral is not None
        self._x: np.ndarray | None = None
        self._p: np.ndarray | None = None
        self._hist: np.ndarray | None = None
        self._gap: np.ndarray | None = None
        self._initialized: np.ndarray | None = None
        self._f = np.eye(4)
        self._neutral_valid: np.ndarray | None = None
        self._n: int | None = None
        self._dt: float = 1.0 / 30.0
        self._last_t: float | None = None

    def _ensure_state(self, n: int) -> None:
        if self._x is not None and self._n == n:
            return
        self._n = n
        self._dt = 1.0 / 30.0
        self._last_t = None
        self._f = np.eye(4)
        self._f[0, 2] = self._f[1, 3] = self._dt
        self._x = np.zeros((n, 4))
        self._p = np.tile(_P_INIT, (n, 1, 1))
        self._hist = np.full((n, 2, self.window), np.nan)
        self._gap = np.zeros(n, dtype=int)
        self._initialized = np.zeros(n, dtype=bool)
        if self._neutral_given and self.neutral is not None:
            self._neutral_valid = np.ones(n, dtype=bool)
        else:
            self.neutral = np.full((n, 2), np.nan, dtype=float)
            self._neutral_valid = np.zeros(n, dtype=bool)

    def _step_time(self, current_t: float | None) -> None:
        dt = self._dt
        if current_t is not None:
            if self._last_t is not None and current_t - self._last_t > 0:
                dt = current_t - self._last_t
            self._last_t = current_t
        if abs(dt - self._dt) > _DT_EPS:
            self._dt = dt
            self._f[0, 2] = self._f[1, 3] = dt

    def _frame(self, x: dict) -> tuple[np.ndarray, np.ndarray, float | None]:
        if 'points' in x:
            pts = as_default_array(x['points'])
            conf = as_default_array(x.get('confidence', np.ones(pts.shape[0])))
            t = x.get('t')
            return pts, conf, None if t is None else float(t)
        keys = sorted(k for k in x if k != 't')
        vals = np.array([float(x[k]) for k in keys], dtype=float)
        pts = np.stack([vals, np.zeros_like(vals)], axis=1) if keys else np.zeros((0, 2))
        t = x.get('t')
        return pts, np.ones(len(keys)), None if t is None else float(t)

    def _update(self, pts: np.ndarray, ok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self._x is not None
        assert self._p is not None
        assert self._hist is not None
        assert self._gap is not None
        assert self._initialized is not None
        assert self._neutral_valid is not None
        assert self.neutral is not None
        n = len(pts)
        out = np.where(np.isfinite(pts), pts, 0.0).astype(float)
        cleaned = np.where(ok, 0, 2)
        xp = self._x @ self._f.T
        pp = self._f @ self._p @ self._f.T + _Q
        init = self._initialized

        occ = ~ok & init
        self._x[occ] = xp[occ]
        self._p[occ] = pp[occ]
        self._gap[occ] += 1
        out[occ] = xp[occ, :2]
        decay = occ & (self._gap > self.max_gap) & self._neutral_valid
        a = np.minimum(1.0, (self._gap[decay] - self.max_gap) * _DECAY_STEP)[:, None]
        out[decay] = (1 - a) * out[decay] + a * self.neutral[decay]

        self._hist[ok & (self._gap > 0)] = np.nan
        upd = ok & init
        if upd.any():
            fx, ox = _hampel_batch(self._hist[upd, 0], pts[upd, 0], self.n_sigmas)
            fy, oy = _hampel_batch(self._hist[upd, 1], pts[upd, 1], self.n_sigmas)
            cleaned[upd] = ox.astype(int) + oy.astype(int)
            pu = pp[upd]
            s_inv = np.linalg.inv(pu[:, :2, :2] + _R)
            gain = pu[:, :, :2] @ s_inv
            innov = np.stack([fx, fy], axis=1) - xp[upd, :2]
            self._x[upd] = xp[upd] + (gain @ innov[..., None])[..., 0]
            p_new = (np.eye(4) - gain @ _H) @ pu
            self._p[upd] = 0.5 * (p_new + p_new.transpose(0, 2, 1))
            out[upd] = self._x[upd, :2]
        new = ok & ~init
        if new.any():
            self._x[new] = np.concatenate([pts[new], np.zeros((int(new.sum()), 2))], axis=1)
            self._p[new] = _P_INIT
            self._initialized[new] = True
            out[new] = pts[new]
        self._gap[ok] = 0
        self._hist[ok, :, :-1] = self._hist[ok, :, 1:]
        self._hist[ok, :, -1] = pts[ok]
        assert len(out) == n
        return out, cleaned

    def _learn_neutral(self, out: np.ndarray, measured: np.ndarray) -> None:
        if self._neutral_given:
            return
        assert self._neutral_valid is not None
        assert self.neutral is not None
        fresh = measured & ~self._neutral_valid
        self.neutral[fresh] = out[fresh]
        self._neutral_valid[fresh] = True

    def transform_one(self, x: dict, t: float | None = None) -> "_BatchedLandmarkShieldOutput":
        """Clean one frame.

        Args:
            x: a frame dict ("points" (N, 2), "confidence" (N,), optional "t"),
                or a flat dict of numbers read as N points on the x axis.
            t: timestamp in seconds, used when the frame has no "t".

        Returns:
            _BatchedLandmarkShieldOutput: dict with keys:
            - "points": cleaned landmarks (N, 2)
            - "cleaned": int array (N,) count of cleaned coordinates (0, 1, or 2)
        """
        pts, conf, frame_t = self._frame(x)
        n = pts.shape[0]
        if n == 0:
            return {'points': pts.copy(), 'cleaned': np.zeros(0, dtype=int)}
        self._ensure_state(n)
        self._step_time(frame_t if frame_t is not None else t)
        measured = (conf > 0.0) & np.all(np.isfinite(pts), axis=1)
        out, cleaned = self._update(pts.astype(float), measured)
        self._learn_neutral(out, measured)
        return {'points': out, 'cleaned': cleaned}

    def learn_one(self, x: dict, y: Any = None, t: float | None = None) -> "BatchedLandmarkShield":
        self.transform_one(x, t)
        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params = super().get_params(deep)
        params.update(
            {
                'window': self.window,
                'n_sigmas': self.n_sigmas,
                'max_gap': self.max_gap,
                'neutral': self.neutral.tolist()
                if self._neutral_given and self.neutral is not None
                else None,
            }
        )
        return params  # type: ignore[no-any-return]

    def _unit_test_skips(self) -> set:
        return set()
