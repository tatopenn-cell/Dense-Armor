"""Robustness for difficult videos: exposure, gain, black frames, bad clocks.

A real video is never a clean sequence of frames. The light changes, the
camera adjusts its exposure, some frames arrive black, the clock of the
container is not always monotone. Every module of the pipeline has to
keep working on these inputs and say what it saw instead of crashing.

:class:`FrameNormalizer` is the one place where all of that is handled:

- exposure and gain are compensated with a per-frame affine stretch that
  maps a low and a high percentile of the frame to the unit interval. It
  does not depend on the content: the same rule applies to a thermal
  frame, to a low-light shot, and to a colour frame from a webcam;
- a black or constant frame is flagged in the output and left as it is,
  so a downstream module can decide to skip it;
- a frame repeated from the previous one is flagged, which is what a
  streaming source does when it stalls;
- a timestamp that goes back in time is clamped to the last one and
  flagged, so a causal pipeline never sees time travel.

The class is a :class:`Transformer`: it returns the normalised frame plus
the flags, in one dict, for the caller that wants them.
"""

import numpy as np

from dense_armor.roles import Transformer


def _to_gray(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 3 and a.shape[2] >= 3:
        return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    if a.ndim == 3:
        return a[..., 0]
    return a


class FrameNormalizer(Transformer):
    """Bring a frame into ``[0, 1]`` and flag the difficult cases.

    Args:
        low_pct: low percentile mapped to 0. Default ``1.0``.
        high_pct: high percentile mapped to 1. Default ``99.0``.
        black_std: a frame whose standard deviation is below this value
            is flagged ``black``. Default ``1e-4``.
        repeat_threshold: a frame whose mean absolute difference from the
            previous is below this value is flagged ``repeated``.
            Default ``1e-6``.
        log_scale: apply ``log1p`` to the frame before the percentile
            stretch. Use it for a signal with a wide dynamic range, for
            example a thermal sensor in linear units. Default ``False``.

    Raises:
        ValueError: if ``low_pct`` is not below ``high_pct`` or if any of
            the two is outside ``[0, 100]``.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.vision.robustness import FrameNormalizer
        >>> n = FrameNormalizer()
        >>> f = np.linspace(0.1, 0.3, 100, dtype=np.float32).reshape(10, 10)
        >>> out = n.transform_one({"array": f, "t": 0.0})
        >>> round(float(out["frame"].min()), 3), round(float(out["frame"].max()), 3)
        (0.0, 1.0)
        >>> out["black"]
        False
        >>> black = np.zeros((10, 10), dtype=np.float32)
        >>> n.transform_one({"array": black, "t": 0.1})["black"]
        True
    """

    memory_class = "O(HW)"

    def __init__(
        self,
        low_pct: float = 1.0,
        high_pct: float = 99.0,
        black_std: float = 1e-4,
        repeat_threshold: float = 1e-6,
        log_scale: bool = False,
    ) -> None:
        if not 0.0 <= low_pct < high_pct <= 100.0:
            raise ValueError(
                f"need 0 <= low_pct < high_pct <= 100, got {low_pct}, {high_pct}"
            )
        self.low_pct = float(low_pct)
        self.high_pct = float(high_pct)
        self.black_std = float(black_std)
        self.repeat_threshold = float(repeat_threshold)
        self.log_scale = bool(log_scale)
        self._prev: np.ndarray | None = None
        self._last_t: float = float("-inf")
        self.n_missing_: int = 0

    def _reset(self) -> None:
        self._prev = None
        self._last_t = float("-inf")
        self.n_missing_ = 0

    def _unit_test_skips(self) -> set:
        return set()

    def _normalize(self, g: np.ndarray) -> np.ndarray:
        if self.log_scale:
            g = np.log1p(np.maximum(g, 0.0))
        lo = float(np.percentile(g, self.low_pct))
        hi = float(np.percentile(g, self.high_pct))
        span = hi - lo
        if span <= 1e-12:
            return np.zeros_like(g, dtype=np.float32)
        out = (g - lo) / span
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def transform_one(self, x, t=None):
        if hasattr(x, "array"):
            arr = x.array
            tt = float(x.t) if x.t is not None else 0.0
        elif isinstance(x, dict) and "array" in x:
            arr = x["array"]
            tt = float(x.get("t", t if t is not None else 0.0))
        else:
            arr = x
            tt = float(t) if t is not None else 0.0

        g = _to_gray(np.asarray(arr))
        black = bool(float(g.std()) < self.black_std)
        time_back = bool(tt < self._last_t)
        tt_eff = max(tt, self._last_t) if self._last_t != float("-inf") else tt
        self._last_t = tt_eff

        if black:
            norm = np.zeros_like(g, dtype=np.float32)
            repeated = False
        else:
            norm = self._normalize(g)
            if self._prev is None or self._prev.shape != norm.shape:
                repeated = False
            else:
                repeated = bool(
                    float(np.mean(np.abs(norm - self._prev))) < self.repeat_threshold
                )
        self._prev = norm

        return {
            "frame": norm,
            "t": tt_eff,
            "black": black,
            "repeated": repeated,
            "time_back": time_back,
        }

    def learn_one(self, x, y=None, t=None):
        return self
