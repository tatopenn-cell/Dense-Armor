"""Sliding window of frames with streaming statistics.

:class:`FrameWindow` keeps the last ``n`` frames (or the last ``s``
seconds) in a ring buffer, and summarises the frames it holds. The
descriptor is a flat dict of numbers: the mean and standard deviation of
the pixel intensity over the window, the mean of the per-frame motion
energy, and a small set of quantiles of the intensity, tracked with the
relative-error sketch of :class:`~dense_armor.utility.stats.quantiles.DDSketch`.

The memory is ``O(n)`` in the number of frames, not in the length of the
stream, so a stream of any length keeps the same footprint. When the
window is given in seconds, the timestamps of the frames decide when an
old frame leaves; the timestamps must be non-decreasing and a timestamp
that goes back in time is clamped to the last one (a monotone clock is a
requirement of any causal estimate).
"""

import numpy as np

from dense_armor.roles import Transformer
from dense_armor.utility.stats.quantiles import DDSketch




def _unpack(x):
    """Return (array, t) from a Frame or from a dict with ``array`` and ``t``."""
    if hasattr(x, "array"):
        return np.asarray(x.array), float(getattr(x, "t", 0.0))
    if isinstance(x, dict):
        return np.asarray(x["array"]), float(x.get("t", 0.0))
    return np.asarray(x), 0.0

def _to_gray(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 3 and a.shape[2] >= 3:
        return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    if a.ndim == 3:
        return a[..., 0]
    return a


class FrameWindow(Transformer):
    """Ring buffer of the last ``n`` frames or ``s`` seconds, with statistics.

    Args:
        n: number of frames in the window. Exactly one of ``n`` and
            ``seconds`` must be given.
        seconds: length of the window in seconds; the window slides on
            the timestamps.
        quantiles: quantiles of the frame intensity to track with
            :class:`DDSketch`. Default ``(0.5, 0.9)``.

    Raises:
        ValueError: if neither or both of ``n`` and ``seconds`` are
            given, if ``n < 1``, if ``seconds <= 0``, or if a quantile is
            outside ``[0, 1]``.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.vision.streams import ArrayStream
        >>> from dense_armor.utility.vision.window import FrameWindow
        >>> frames = np.zeros((5, 4, 4), dtype=np.float32)
        >>> frames[4, :, :] = 1.0
        >>> w = FrameWindow(n=3)
        >>> out = []
        >>> for fr in ArrayStream(frames, fps=10.0):
        ...     _ = w.learn_one(fr)
        ...     out.append(w.transform_one(fr))
        >>> round(out[3]["mean"], 4)
        0.0
        >>> 0.0 < out[4]["mean"] < 1.0
        True
    """

    memory_class = "O(n)"

    def __init__(
        self,
        n: int | None = None,
        seconds: float | None = None,
        quantiles=(0.5, 0.9),
    ) -> None:
        if (n is None) == (seconds is None):
            raise ValueError(
                "exactly one of n and seconds must be given"
            )
        if n is not None and n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if seconds is not None and seconds <= 0.0:
            raise ValueError(f"seconds must be > 0, got {seconds}")
        for q in quantiles:
            if not 0.0 <= q <= 1.0:
                raise ValueError(f"quantiles must be in [0, 1], got {q}")
        self.n = None if n is None else int(n)
        self.seconds = None if seconds is None else float(seconds)
        self.quantiles = tuple(float(q) for q in quantiles)
        self._times: list[float] = []
        self._arrays: list[np.ndarray] = []
        self._sketch = DDSketch(alpha=0.05)
        self._motion: list[float] = []
        self._prev: np.ndarray | None = None

    def _reset(self) -> None:
        self._times = []
        self._arrays = []
        self._sketch = DDSketch(alpha=0.05)
        self._motion = []
        self._prev = None

    def _unit_test_skips(self) -> set:
        return set()

    def _drop_old(self, now: float) -> None:
        if self.n is not None:
            while len(self._arrays) > self.n:
                self._arrays.pop(0)
                self._motion.pop(0)
                self._times.pop(0)
        else:
            while self._times and now - self._times[0] > self.seconds:
                self._arrays.pop(0)
                self._motion.pop(0)
                self._times.pop(0)

    def _motion_energy(self, g: np.ndarray) -> float:
        if self._prev is None:
            return 0.0
        f = 2
        h, w = g.shape[0] // f, g.shape[1] // f
        if h < 1 or w < 1:
            d = np.abs(g - self._prev)
        else:
            a = g[: h * f, : w * f].reshape(h, f, w, f).mean(axis=(1, 3))
            b = self._prev[: h * f, : w * f].reshape(h, f, w, f).mean(axis=(1, 3))
            d = np.abs(a - b)
        return float(d.mean())

    def _sketch_values(self) -> None:
        s = DDSketch(alpha=0.05)
        for a in self._arrays:
            flat = a.reshape(-1)
            step = max(1, len(flat) // 64)
            for v in flat[::step]:
                s.learn_one({"x": float(v)})
        self._sketch = s

    def learn_one(self, x, y=None, t=None):
        arr, tt = _unpack(x)
        if t is not None and not hasattr(x, "t") and not isinstance(x, dict):
            tt = float(t)
        g = _to_gray(arr)
        if self._times and tt < self._times[-1]:
            tt = self._times[-1]
        m = self._motion_energy(g)
        self._prev = g
        self._arrays.append(g)
        self._motion.append(m)
        self._times.append(float(tt))
        self._drop_old(float(tt))
        return self

    def transform_one(self, x, t=None):
        if not self._arrays:
            return {
                "mean": 0.0,
                "std": 0.0,
                "motion": 0.0,
                **{f"q{int(q * 100)}": 0.0 for q in self.quantiles},
                "n_frames": 0,
            }
        stack = np.stack(self._arrays)
        mean = float(stack.mean())
        std = float(stack.std())
        motion = float(np.mean(self._motion))
        self._sketch_values()
        out = {"mean": mean, "std": std, "motion": motion,
               "n_frames": len(self._arrays)}
        for q in self.quantiles:
            out[f"q{int(q * 100)}"] = float(self._sketch.quantile(q))
        return out
