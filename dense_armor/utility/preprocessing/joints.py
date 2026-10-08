"""Joint kinematics by finite differences on real timestamps.

``JointDerivatives`` turns joint positions ``q`` into velocity,
acceleration and jerk. The derivatives are computed on the real
timestamps ``t``, so non-uniform sampling is handled exactly. Missing
samples (NaN in any element of ``q`` or in ``t``) are skipped and
counted; the next derivative uses the last valid sample with its own
``t``.

The formulas come from Newton divided differences over the last
``order + 1`` valid points, which are exact on any polynomial of
degree at most ``order``. ``transform_one(x, t)`` uses the current
sample ``x`` at time ``t`` together with the buffer, so in a pipeline
(transform first, then learn) the derivative returned at time ``t``
refers to the sample at time ``t``, not to the last stored one.
``learn_one`` stores the sample in the buffer.

``JointPower`` gives the mechanical power ``tau * qd`` per joint and
its total.
"""
import math
from collections import deque
from typing import Any

import numpy as np

from dense_armor.roles import Transformer


def _as_dict(x: Any) -> dict:
    if hasattr(x, "to_dict"):
        return dict(x.to_dict())
    return dict(x)


def _derivatives_at_last(
    ts: list[float], qs: list[float]
) -> tuple[float, float, float]:
    """Return the first, second and third derivatives at the last point."""
    n = len(ts)
    if n < 2:
        return 0.0, 0.0, 0.0
    d = [float(v) for v in qs]
    for order in range(1, n):
        for i in range(n - 1, order - 1, -1):
            d[i] = (d[i] - d[i - 1]) / (ts[i] - ts[i - order])
    tl = ts[-1]
    v1 = d[1]
    if n >= 3:
        v1 += d[2] * ((tl - ts[1]) + (tl - ts[0]))
    if n >= 4:
        v1 += d[3] * (
            (tl - ts[1]) * (tl - ts[2])
            + (tl - ts[0]) * (tl - ts[2])
            + (tl - ts[0]) * (tl - ts[1])
        )
    a = 0.0
    if n >= 3:
        a = 2.0 * d[2]
    if n >= 4:
        a += 2.0 * d[3] * (
            (tl - ts[1]) + (tl - ts[2]) + (tl - ts[0])
        )
    j = 6.0 * d[3] if n >= 4 else 0.0
    return v1, a, j


class JointDerivatives(Transformer):
    """Velocity, acceleration and jerk from joint positions.

    Input: a dict with the joint positions under ``q_key`` (a vector)
    and the timestamp ``t`` passed to ``learn_one`` and to
    ``transform_one``. Output: a dict with ``qd``, ``qdd`` and ``jerk``
    keys, each a vector of the same length as ``q``. ``order`` selects
    how many derivatives are produced (1, 2 or 3).

    In a pipeline, call ``transform_one(x, t)`` first and
    ``learn_one(x, t)`` after: the derivative returned at time ``t``
    refers to the sample at time ``t``.

    Memory grows with the number of joints and the derivative order:
    the buffer holds ``order + 1`` past samples, one per joint, so the
    total is ``O(d * order)``.

    Args:
        order: number of derivatives to emit.
        q_key: input key holding the joint positions.

    Examples:
        >>> from dense_armor.utility.preprocessing.joints import JointDerivatives
        >>> jd = JointDerivatives(order=3)
        >>> for t in [0.0, 1.0, 2.0, 3.0]:
        ...     _ = jd.learn_one({"q": [t ** 3]}, t=t)
        >>> round(jd.transform_one({})["qd"][0], 6)
        27.0
        >>> round(jd.transform_one({})["qdd"][0], 6)
        18.0
    """

    budget_s = 1e-3
    memory_class = "O(1)"

    def __init__(self, order: int = 3, q_key: str = "q") -> None:
        self.order = order
        self.q_key = q_key
        self.buf_: deque[tuple[float, np.ndarray]] = deque(maxlen=order + 1)
        self.n_missing_ = 0

    def _extract(
        self, x: dict, t: float | None
    ) -> tuple[float, np.ndarray] | None:
        if t is None or self.q_key not in x:
            return None
        try:
            qv = np.asarray(x[self.q_key], dtype=float)
        except (TypeError, ValueError):
            return None
        if qv.ndim == 0:
            qv = qv.reshape(1)
        tf = float(t)
        if not np.all(np.isfinite(qv)) or not math.isfinite(tf):
            return None
        return tf, qv

    def learn_one(
        self, x: Any, y: Any = None, t: float | None = None
    ) -> "JointDerivatives":
        """Append one valid pair of timestamp and positions to the buffer."""
        self._time_step(t)
        res = self._extract(_as_dict(x), t)
        if res is None:
            self.n_missing_ += 1
            return self
        tf, qv = res
        self.buf_.append((tf, qv.copy()))
        return self

    def transform_one(self, x: Any, t: float | None = None) -> dict:
        """Return the derivative vectors at time ``t``.

        Uses the buffer together with the current sample ``x`` at time
        ``t``. If ``t`` is None, or ``x`` lacks the positions, the
        last stored derivatives are returned.
        """
        out = _as_dict(x)
        res = self._extract(_as_dict(x), t)
        if res is None:
            hist = list(self.buf_)
        else:
            tf, qv = res
            hist = list(self.buf_) + [(tf, qv.copy())]
        if len(hist) < 2:
            return out
        hist = hist[-(self.order + 1):]
        ts = [p[0] for p in hist]
        qs = np.stack([p[1] for p in hist], axis=0)
        _, d = qs.shape
        v = np.zeros(d)
        a = np.zeros(d)
        j = np.zeros(d)
        for k in range(d):
            vk, ak, jk = _derivatives_at_last(ts, qs[:, k].tolist())
            v[k], a[k], j[k] = vk, ak, jk
        if self.order >= 1:
            out["qd"] = v.tolist()
        if self.order >= 2:
            out["qdd"] = a.tolist()
        if self.order >= 3:
            out["jerk"] = j.tolist()
        return out
        hist = hist[-(self.order + 1):]
        ts = [p[0] for p in hist]
        qs = np.stack([p[1] for p in hist], axis=0)
        _, d = qs.shape
        v = np.zeros(d)
        a = np.zeros(d)
        j = np.zeros(d)
        for k in range(d):
            vk, ak, jk = _derivatives_at_last(ts, qs[:, k].tolist())
            v[k], a[k], j[k] = vk, ak, jk
        if self.order >= 1:
            out["qd"] = v.tolist()
        if self.order >= 2:
            out["qdd"] = a.tolist()
        if self.order >= 3:
            out["jerk"] = j.tolist()
        return out


class JointPower(Transformer):
    """Mechanical power ``tau * qd`` per joint and its total.

    Memory grows with the number of joints only, since no state is
    kept.

    Args:
        tau_key: input key holding the joint torques.
        qd_key: input key holding the joint velocities.

    Examples:
        >>> from dense_armor.utility.preprocessing.joints import JointPower
        >>> jp = JointPower()
        >>> out = jp.transform_one({"tau": [1.0, 2.0], "qd": [3.0, 4.0]})
        >>> out["power"], out["power_total"]
        ([3.0, 8.0], 11.0)
    """

    budget_s = 1e-3
    memory_class = "O(1)"

    def __init__(self, tau_key: str = "tau", qd_key: str = "qd") -> None:
        self.tau_key = tau_key
        self.qd_key = qd_key
        self.n_missing_ = 0

    def learn_one(
        self, x: Any, y: Any = None, t: float | None = None
    ) -> "JointPower":
        """No state to update; kept for the ``Transformer`` interface."""
        self._time_step(t)
        return self

    def transform_one(self, x: Any, t: float | None = None) -> dict:
        """Return the per-joint power and the total power."""
        d = _as_dict(x)
        out = dict(d)
        if self.tau_key not in d or self.qd_key not in d:
            self.n_missing_ += 1
            return out
        tau = np.asarray(d[self.tau_key], dtype=float)
        qd = np.asarray(d[self.qd_key], dtype=float)
        if tau.shape != qd.shape:
            self.n_missing_ += 1
            return out
        p = tau * qd
        out["power"] = p.tolist()
        out["power_total"] = float(p.sum())
        return out
