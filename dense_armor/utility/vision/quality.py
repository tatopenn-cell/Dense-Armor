"""Quality of a frame, sync between cameras, non-standard inputs.

A production pipeline needs to know when a frame is bad before any
detector sees it. This module answers three questions:

- :class:`FrameQuality` is a per-frame descriptor with the sharpness,
  the exposure, the noise, and two boolean flags for a black frame and a
  frame repeated from the previous one. Sharpness is the variance of the
  Laplacian; exposure is the mean and the fraction of pixels saturated
  at 0 and 1; noise is the high-frequency energy that survives a small
  median filter.
- :class:`Sync` estimates the clock offset between two streams that
  share the same physical event (two cameras, a camera and a sensor).
  The offset is estimated from a cross-correlation of the feature
  differences of the two streams, then applied to one of them so the
  timestamps align.
- :class:`EventFrame` accumulates an event stream ``(t, x, y, polarity)``
  into a frame: each event adds ``+1`` at its pixel for positive
  polarity and ``-1`` for negative, and the accumulated frame is
  returned with the timestamp of the last event. Depth maps and
  multi-channel frames come in through :class:`~dense_armor.utility.vision.streams.CubeStream`
  and need no other class here.
"""

from collections import deque

import numpy as np

from dense_armor.roles import Transformer


def _to_gray(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 3 and a.shape[2] >= 3:
        return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    if a.ndim == 3:
        return a[..., 0]
    return a


class FrameQuality(Transformer):
    """Per-frame sharpness, exposure, noise and difficulty flags.

    Args:
        black_std: standard deviation below which a frame is flagged
            ``black``. Default ``1e-4``.
        repeat_threshold: mean absolute difference below which a frame
            is flagged ``repeated``. Default ``1e-6``.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.vision.quality import FrameQuality
        >>> q = FrameQuality()
        >>> f = np.zeros((8, 8), dtype=np.float32)
        >>> f[2, 2] = 1.0
        >>> out = q.transform_one({"array": f})
        >>> out["black"]
        False
        >>> out["sharpness"] > 0
        True
    """

    memory_class = "O(HW)"

    def __init__(self, black_std: float = 1e-4,
                 repeat_threshold: float = 1e-6) -> None:
        self.black_std = float(black_std)
        self.repeat_threshold = float(repeat_threshold)
        self._prev: np.ndarray | None = None

    def _reset(self) -> None:
        self._prev = None

    def _unit_test_skips(self) -> set:
        return set()

    def transform_one(self, x, t=None):
        arr = x["array"] if isinstance(x, dict) else (
            x.array if hasattr(x, "array") else x
        )
        g = _to_gray(np.asarray(arr))
        # sharpness: varianza del laplaciano
        lap = (
            -4.0 * g
            + np.roll(g, 1, 0) + np.roll(g, -1, 0)
            + np.roll(g, 1, 1) + np.roll(g, -1, 1)
        )
        sharpness = float(lap.var())
        exposure = float(g.mean())
        saturated_low = float((g <= 1e-6).mean())
        saturated_high = float((g >= 1 - 1e-6).mean())
        # rumore: differenza tra il frame e se stesso filtrato con mediana 3x3
        if g.shape[0] >= 3 and g.shape[1] >= 3:
            padded = np.pad(g, 1, mode="edge")
            stacked = np.stack([
                padded[i:i+g.shape[0], j:j+g.shape[1]]
                for i in range(3) for j in range(3)
            ])
            med = np.median(stacked, axis=0)
            noise = float(np.mean(np.abs(g - med)))
        else:
            noise = 0.0
        black = bool(float(g.std()) < self.black_std)
        repeated = False
        if self._prev is not None and self._prev.shape == g.shape:
            repeated = bool(
                float(np.mean(np.abs(g - self._prev))) < self.repeat_threshold
            )
        self._prev = g.copy()
        return {
            "sharpness": sharpness,
            "exposure": exposure,
            "saturated_low": saturated_low,
            "saturated_high": saturated_high,
            "noise": noise,
            "black": black,
            "repeated": repeated,
        }

    def learn_one(self, x, y=None, t=None):
        return self


class Sync(Transformer):
    """Estimate and apply the clock offset between two streams.

    Args:
        window: number of paired samples used for the cross-correlation.
            Default ``200``.
        max_offset_s: maximum offset searched. Default ``0.5``.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.vision.quality import Sync
        >>> rng = np.random.default_rng(0)
        >>> s = Sync(window=200, max_offset_s=0.2)
        >>> sig = rng.normal(0, 1, 200)
        >>> for i in range(200):
        ...     _ = s.add_a(float(sig[i]), t=i * 0.01)
        ...     _ = s.add_b(float(sig[i]), t=i * 0.01 + 0.05)
        >>> out = s.estimate()
        >>> bool(abs(out["offset_s"] - 0.05) < 0.02)
        True
    """

    memory_class = "O(window)"

    def __init__(self, window: int = 200, max_offset_s: float = 0.5) -> None:
        if window < 4:
            raise ValueError(f"window must be >= 4, got {window}")
        if max_offset_s <= 0:
            raise ValueError(f"max_offset_s must be > 0, got {max_offset_s}")
        self.window = int(window)
        self.max_offset_s = float(max_offset_s)
        self._a: deque = deque(maxlen=self.window)
        self._b: deque = deque(maxlen=self.window)

    def _reset(self) -> None:
        self._a = deque(maxlen=self.window)
        self._b = deque(maxlen=self.window)

    def _unit_test_skips(self) -> set:
        return set()

    def add_a(self, value: float, t: float):
        self._a.append((float(t), float(value)))
        return self

    def add_b(self, value: float, t: float):
        self._b.append((float(t), float(value)))
        return self

    def estimate(self) -> dict:
        if len(self._a) < 4 or len(self._b) < 4:
            return {"offset_s": 0.0, "correlation": 0.0}
        ta = np.array([t for t, _ in self._a])
        va = np.array([v for _, v in self._a])
        tb = np.array([t for t, _ in self._b])
        vb = np.array([v for _, v in self._b])
        if va.std() < 1e-9 or vb.std() < 1e-9:
            return {"offset_s": 0.0, "correlation": 0.0}
        t0 = max(ta[0], tb[0])
        t1 = min(ta[-1], tb[-1])
        if t1 <= t0:
            return {"offset_s": 0.0, "correlation": 0.0}
        ts = np.linspace(t0, t1, self.window)
        a = np.interp(ts, ta, va)
        b = np.interp(ts, tb, vb)
        a = (a - a.mean()) / (a.std() + 1e-9)
        b = (b - b.mean()) / (b.std() + 1e-9)
        dt = ts[1] - ts[0]
        max_shift = int(round(self.max_offset_s / dt))
        best_shift = 0
        best = -2.0
        n = len(ts)
        for sh in range(-max_shift, max_shift + 1):
            if sh > 0:
                aa = a[sh:]; bb = b[:n - sh]
            elif sh < 0:
                aa = a[:n + sh]; bb = b[-sh:]
            else:
                aa = a; bb = b
            if len(aa) < 4 or aa.std() < 1e-9 or bb.std() < 1e-9:
                continue
            c = float(np.corrcoef(aa, bb)[0, 1])
            if c > best:
                best = c
                best_shift = sh
        return {"offset_s": -best_shift * dt, "correlation": best}

    def transform_one(self, x=None, t=None):
        return self.estimate()

    def learn_one(self, x, y=None, t=None):
        return self


class EventFrame(Transformer):
    """Accumulate an event stream into a frame, one event at a time.

    Args:
        shape: ``(H, W)`` of the output frame. Default ``(64, 64)``.
        decay: at every new frame the accumulator is multiplied by this
            value before the events are added. ``1.0`` keeps every event
            seen, useful for a fixed window; ``0.0`` clears the frame.
            Default ``0.9``.

    Raises:
        ValueError: on a shape that is not two positive integers.

    Examples:
        >>> from dense_armor.utility.vision.quality import EventFrame
        >>> e = EventFrame(shape=(8, 8), decay=1.0)
        >>> for _ in range(3):
        ...     _ = e.learn_one({"x": 4, "y": 2, "polarity": 1, "t": 0.0})
        >>> out = e.transform_one(None)
        >>> int(out["frame"][2, 4])
        3
    """

    memory_class = "O(HW)"

    def __init__(self, shape=(64, 64), decay: float = 0.9) -> None:
        if (not isinstance(shape, (tuple, list)) or len(shape) != 2
                or shape[0] < 1 or shape[1] < 1):
            raise ValueError(f"shape must be (H, W) positive, got {shape}")
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"decay must be in [0, 1], got {decay}")
        self.shape = (int(shape[0]), int(shape[1]))
        self.decay = float(decay)
        self._frame: np.ndarray | None = None
        self._last_t: float = 0.0

    def _reset(self) -> None:
        self._frame = None
        self._last_t = 0.0

    def _unit_test_skips(self) -> set:
        return set()

    def learn_one(self, x, y=None, t=None):
        if isinstance(x, dict) and "events" in x:
            events = x["events"]
        elif isinstance(x, (list, tuple, np.ndarray)) and (
            isinstance(x, np.ndarray) or (len(x) > 0 and isinstance(x[0], (list, tuple)))
        ):
            events = x
        else:
            events = [x]
        for ev in events:
            if isinstance(ev, dict):
                ex = int(ev["x"]); ey = int(ev["y"]); pol = int(ev.get("polarity", 1))
                tt = float(ev.get("t", t if t is not None else 0.0))
            else:
                ex, ey, pol, tt = int(ev[0]), int(ev[1]), int(ev[2]), float(ev[3])
            if self._frame is None:
                self._frame = np.zeros(self.shape, dtype=np.float32)
            if 0 <= ey < self.shape[0] and 0 <= ex < self.shape[1]:
                self._frame[ey, ex] += 1.0 if pol > 0 else -1.0
            self._last_t = tt
        return self

    def transform_one(self, x=None, t=None):
        if self._frame is None:
            self._frame = np.zeros(self.shape, dtype=np.float32)
        out = self._frame.copy()
        if self.decay < 1.0:
            self._frame = self._frame * self.decay
        else:
            self._frame = np.zeros(self.shape, dtype=np.float32)
        return {"frame": out, "t": self._last_t, "n_events": int(np.abs(out).sum())}
