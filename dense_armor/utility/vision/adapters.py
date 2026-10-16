"""Adapters from video features to a :class:`Signal`.

The library already has every online estimator a robot needs (drift
detectors, anomaly detectors, metrics, the Armature protection). They
all take a :class:`~dense_armor.roles.Signal`, a dict-like reading with
named channels, units and a timestamp. A video is not a Signal: it is an
image, or a list of boxes, or a dict of features. The two classes here
close that gap.

- :class:`TrackToSignal` turns one track of
  :class:`~dense_armor.utility.vision.tracking.MovingObjectTracker` into
  a Signal with the box centre, the box size and the speed as channels.
- :class:`FeatureToSignal` turns any dict of scalar features (the output
  of :class:`~dense_armor.utility.vision.features.FrameFeatures`, of
  :class:`~dense_armor.utility.vision.motion.MotionEnergy`, or of any
  transformer) into a Signal with one channel per feature.

Both keep the timestamp of the frame, so a detector that uses time
(``Rolling``, ``PageHinkley`` with a warm-up in seconds) sees the true
video clock.
"""

import numpy as np

from dense_armor.roles import Signal, Transformer


class TrackToSignal(Transformer):
    """Turn one track of the tracker into a Signal.

    Args:
        track_id: id of the track to read, or ``None`` for the first
            track of the frame. Default ``None``.
        units: ``{channel: unit}`` mapping for the output. Any channel
            not listed gets ``""``.

    Raises:
        ValueError: if ``track_id`` is negative.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.vision.adapters import TrackToSignal
        >>> a = TrackToSignal(track_id=7, units={"cx": "px", "cy": "px"})
        >>> _ = a.learn_one({"tracks": [{"id": 7, "box": (10, 20, 14, 26), "missing": 0}]})
        >>> s = a.transform_one(None)
        >>> s["cx"], s["cy"]
        (12.0, 23.0)
        >>> s.units[:2]
        ['px', 'px']
    """

    memory_class = "O(1)"

    def __init__(self, track_id: int | None = None, units=None) -> None:
        if track_id is not None and track_id < 0:
            raise ValueError(f"track_id must be >= 0, got {track_id}")
        self.track_id = track_id
        self.units = dict(units or {})
        self._prev: dict | None = None

    def _reset(self) -> None:
        self._prev = None

    def _unit_test_skips(self) -> set:
        return set()

    def learn_one(self, x, y=None, t=None):
        tracks = []
        if isinstance(x, dict) and "tracks" in x:
            tracks = x["tracks"]
        elif isinstance(x, list):
            tracks = x
        chosen = None
        if self.track_id is not None:
            for tr in tracks:
                if tr.get("id") == self.track_id:
                    chosen = tr
                    break
        elif tracks:
            chosen = tracks[0]
        self._prev = chosen
        return self

    def transform_one(self, x, t=None):
        tr = self._prev
        if tr is None:
            return Signal(
                values=np.zeros(5, dtype=float),
                names=["cx", "cy", "w", "h", "speed"],
                units=[self.units.get(k, "") for k in
                       ("cx", "cy", "w", "h", "speed")],
                t=float(t) if t is not None else 0.0,
            )
        box = tr["box"]
        x0, y0, x1, y1 = box
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        w = x1 - x0
        h = y1 - y0
        speed = 0.0
        if hasattr(self, "_last") and self._last is not None:
            dt = (float(t) if t is not None else 0.0) - self._last[2]
            if dt > 0:
                dx = cx - self._last[0]
                dy = cy - self._last[1]
                speed = float(np.hypot(dx, dy) / dt)
        self._last = (cx, cy, float(t) if t is not None else 0.0)
        names = ["cx", "cy", "w", "h", "speed"]
        return Signal(
            values=np.array([cx, cy, w, h, speed], dtype=float),
            names=names,
            units=[self.units.get(k, "") for k in names],
            t=float(t) if t is not None else 0.0,
        )


class FeatureToSignal(Transformer):
    """Turn a dict of scalar features into a Signal.

    Args:
        units: ``{feature: unit}`` mapping. Any feature not listed gets
            ``""``.
        include: if given, only these features are kept, in this order.
            ``None`` keeps every feature in the dict, sorted.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.vision.adapters import FeatureToSignal
        >>> a = FeatureToSignal(units={"mean": "1"})
        >>> s = a.transform_one({"mean": 0.5, "std": 0.1})
        >>> round(s["mean"], 4), round(s["std"], 4)
        (0.5, 0.1)
        >>> s.units
        ['1', '']
    """

    memory_class = "O(1)"

    def __init__(self, units=None, include=None) -> None:
        self.units = dict(units or {})
        self.include = None if include is None else list(include)

    def _reset(self) -> None:
        pass

    def _unit_test_skips(self) -> set:
        return set()

    def learn_one(self, x, y=None, t=None):
        return self

    def transform_one(self, x, t=None):
        if self.include is not None:
            keys = self.include
        else:
            if hasattr(x, "items"):
                keys = sorted(k for k in x.keys())
            else:
                keys = []
        vals = []
        for k in keys:
            v = x.get(k, 0.0) if hasattr(x, "get") else 0.0
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                vals.append(0.0)
        if not keys:
            return Signal(
                values=np.zeros(1, dtype=float),
                names=["empty"], units=[""],
                t=float(t) if t is not None else 0.0,
            )
        return Signal(
            values=np.array(vals, dtype=float),
            names=list(keys),
            units=[self.units.get(k, "") for k in keys],
            t=float(t) if t is not None else 0.0,
        )
