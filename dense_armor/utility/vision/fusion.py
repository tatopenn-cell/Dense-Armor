"""Fuse a frame stream with a joint stream, in time, with a latency estimate.

A robot sees its own arm: the joint readings arrive from the
encoders, the frames arrive from the camera, and the two are not on the
same clock. The video pipeline of the library works with
:class:`~dense_armor.roles.Signal` objects, timestamped in seconds; the
joint readings are Signals too. This module aligns them.

- :func:`fuse_frames_joints` merges two iterables of ``(Signal, y)``
  pairs with :func:`~dense_armor.utility.stream.merge_by_time`, so the
  caller gets one stream in time order and can feed any detector.
- :class:`FrameJointFusion` estimates the latency of the camera, frame
  by frame: the motion seen in the video (the global shift of
  :class:`~dense_armor.utility.vision.motion.GlobalMotion`) is compared
  with the joint velocity of the same instant, and the lag that
  maximises the correlation between the two is the estimated latency.
  The estimate is causal, kept in a small ring buffer.

The paper that the tracker uses for the adaptive noise of the Kalman
filter is the same one that the fusion uses to keep the latency estimate
stable over time; see the module docstring of
:mod:`dense_armor.utility.vision.tracking`.
"""

from collections import deque

import numpy as np

from dense_armor.roles import Transformer
from dense_armor.utility.stream.merge import merge_by_time
from dense_armor.utility.vision.motion import GlobalMotion


def fuse_frames_joints(frames, joints):
    """Merge a frame stream and a joint stream in time order.

    Args:
        frames: iterable of ``(Signal, y)`` pairs, ``Signal.t`` in
            seconds, non-decreasing.
        joints: the same, for the joint readings.

    Yields:
        ``(signal, y)`` pairs in time order. The signal is the one of
        whichever stream fired; the caller distinguishes them by the
        channel names.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.roles import Signal
        >>> from dense_armor.utility.vision.fusion import fuse_frames_joints
        >>> def a():
        ...     for t in (0.0, 0.2):
        ...         yield Signal(np.array([1.0]), ["frame"], [""], t=t), None
        >>> def b():
        ...     for t in (0.1, 0.3):
        ...         yield Signal(np.array([2.0]), ["q0"], ["rad"], t=t), None
        >>> [round(s.t, 2) for s, _ in fuse_frames_joints(a(), b())]
        [0.0, 0.1, 0.2, 0.3]
    """
    return merge_by_time(frames, joints)


class FrameJointFusion(Transformer):
    """Estimate the camera latency by correlating video motion and joint speed.

    The class keeps a short history of the video motion magnitude and of
    the joint speed, and computes the lag that maximises their
    correlation. Positive lag means the video is *behind* the joints, so
    the video at time ``t`` corresponds to the joint state at
    ``t - lag``. The estimate is causal: only the history up to the
    current sample is used.

    Args:
        max_lag_s: maximum latency searched for, in seconds. Default
            ``0.1``.
        window_s: length of the history used in the correlation, in
            seconds. Default ``1.0``.
        frame_rate_hz: frame rate used to sample the joint speed at
            frame times. Default ``30``.

    Raises:
        ValueError: on non-positive parameters.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.vision.streams import ArrayStream
        >>> from dense_armor.utility.vision.fusion import FrameJointFusion
        >>> f = FrameJointFusion(max_lag_s=0.05, window_s=0.2)
        >>> _ = f
    """

    memory_class = "O(window)"

    def __init__(
        self,
        max_lag_s: float = 0.1,
        window_s: float = 1.0,
        frame_rate_hz: float = 30.0,
    ) -> None:
        if max_lag_s <= 0:
            raise ValueError(f"max_lag_s must be > 0, got {max_lag_s}")
        if window_s <= 0:
            raise ValueError(f"window_s must be > 0, got {window_s}")
        if frame_rate_hz <= 0:
            raise ValueError(f"frame_rate_hz must be > 0, got {frame_rate_hz}")
        self.max_lag_s = float(max_lag_s)
        self.window_s = float(window_s)
        self.frame_rate_hz = float(frame_rate_hz)
        self._frames: deque = deque()
        self._joints: deque = deque()
        self._motion = GlobalMotion()

    def _reset(self) -> None:
        self._frames = deque()
        self._joints = deque()
        self._motion = GlobalMotion()

    def _unit_test_skips(self) -> set:
        return set()

    def add_frame(self, frame, t: float) -> None:
        """Register a frame at time ``t``."""
        arr = frame["array"] if isinstance(frame, dict) else (
            frame.array if hasattr(frame, "array") else frame
        )
        tt = float(t)
        out = self._motion.transform_one(arr, t=tt)
        _ = self._motion.learn_one(arr)
        vel = float(out["dx"]) * self.frame_rate_hz
        self._frames.append((tt, vel))
        self._trim(tt)

    def add_video_velocity(self, velocity: float, t: float) -> None:
        """Register the video velocity at time ``t``, without a frame.

        Used when the velocity of the scene has already been measured by
        :class:`~dense_armor.utility.vision.motion.GlobalMotion` or by
        the tracker; the class then only does the correlation with the
        joint speed.
        """
        self._frames.append((float(t), float(velocity)))
        self._trim(float(t))

    def add_joint_speed(self, speed: float, t: float) -> None:
        """Register the joint speed at time ``t``."""
        self._joints.append((float(t), float(speed)))
        self._trim(float(t))

    def _trim(self, now: float) -> None:
        cut = now - self.window_s
        while self._frames and self._frames[0][0] < cut:
            self._frames.popleft()
        while self._joints and self._joints[0][0] < cut:
            self._joints.popleft()

    def estimate(self) -> dict:
        """Return the current estimate of the latency, in seconds.

        The video velocity and the joint velocity are resampled on a
        common grid at the video frame rate and cross-correlated: the
        lag that maximises the correlation is the latency. Positive lag
        means the video is behind the joints.
        """
        if len(self._frames) < 8 or len(self._joints) < 8:
            return {"lag_s": 0.0, "correlation": 0.0, "n": 0}
        ft = np.array([t for t, _ in self._frames])
        fv = np.array([v for _, v in self._frames])
        jt = np.array([t for t, _ in self._joints])
        jv = np.array([v for _, v in self._joints])
        if fv.std() < 1e-9 or jv.std() < 1e-9:
            return {"lag_s": 0.0, "correlation": 0.0, "n": len(fv)}
        # interpoliamo la velocita' del giunto sui tempi dei frame
        jv_at_f = np.interp(ft, jt, jv)
        # normalizzo
        fv_n = (fv - fv.mean()) / (fv.std() + 1e-9)
        jv_n = (jv_at_f - jv_at_f.mean()) / (jv_at_f.std() + 1e-9)
        dt = float(np.median(np.diff(ft))) if len(ft) > 1 else 1.0 / self.frame_rate_hz
        max_shift = int(round(self.max_lag_s / dt))
        n = len(fv_n)
        best_shift = 0
        best_corr = -2.0
        for shift in range(max_shift + 1):
            a = fv_n[shift:]
            b = jv_n[:n - shift]
            if len(a) < 4 or a.std() < 1e-9 or b.std() < 1e-9:
                continue
            c = float(np.corrcoef(a, b)[0, 1])
            if c > best_corr:
                best_corr = c
                best_shift = shift
        return {"lag_s": best_shift * dt, "correlation": best_corr, "n": n}

    def transform_one(self, x, t=None):
        return self.estimate()

    def learn_one(self, x, y=None, t=None):
        if hasattr(x, "array") or (hasattr(x, "get") and "frame" in x):
            frame = x if hasattr(x, "array") else x["frame"]
            tt = float(t) if t is not None else float(getattr(x, "t", 0.0))
            self.add_frame(frame, tt)
        return self
