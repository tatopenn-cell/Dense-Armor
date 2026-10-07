# -*- coding: utf-8 -*-
"""
dynamics/online_dynamics.py
===========================
Online learning of the residual between a URDF-nominal rigid-body model and
the real robot's measured joint torques, one sample at a time, plus a
CUSUM/Hampel-triggered drift-aware variant that lowers the RLS forgetting
factor when the robot changes (new payload, contact).

The nominal model comes from ``urdf_dynamics.RigidBodyModel``. The residual
is learned by one ``RecursiveLeastSquares`` per joint, with features
``[qdd_j, qd_j, sign(qd_j), g_j, 1]`` and target
``tau_j - (M(q) qdd + C(q,qd) qd + g(q))_j``.

Performance note: ``RigidBodyModel`` methods are pure-Python recursive walks
and retrace the JAX graph on every call. In a control loop, JIT-patch the
model once (``model.mass_matrix = jax.jit(model.mass_matrix)`` and likewise
``bias_forces``/``gravity_forces``) or the learner's per-sample cost is
milliseconds instead of tens of microseconds.

References
----------
Ljung, L., Soderstrom, T. (1983). Theory and Practice of Recursive
Identification. MIT Press. (RLS with exponential forgetting.)
Page, E. S. (1954). Continuous inspection schemes. Biometrika 41, 100-114.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

try:
    from river import base
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "dense_armor.utility.learn.online_dynamics needs river: "
        "pip install dense-armor[river]"
    ) from exc

import jax.numpy as jnp

from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
from dense_armor.utility.drift.detector import CUSUMDriftDetector
from dense_armor.utility.anomaly.filters import HampelScorer


class RecursiveLeastSquares(base.Regressor):
    """Recursive least squares with exponential forgetting.

    After each step the weights ``w`` minimize the exponentially weighted
    sum of squared residuals ``sum_i lam^{n-i} (y_i - x_i^T w)^2``.

    Update, one sample at a time:

        K = P x / (lam + x^T P x)
        w <- w + K (y - x^T w)
        P <- (P - K x^T P) / lam

    with ``P_0 = delta * I`` and ``w_0 = 0``. P is symmetrised after each
    step to keep numerical errors from breaking the symmetry the derivation
    assumes. With ``lam = 1`` and ``delta`` large, the weights after n
    samples equal the ordinary least-squares solution on those n samples.

    Parameters
    ----------
    lam : float
        Forgetting factor in ``(0, 1]``. ``lam = 1`` weights all past samples
        equally. Smaller values weight recent samples more, letting the model
        track a parameter that changes.
    delta : float
        Initial covariance scale: ``P_0 = delta * I``.
    feature_keys : sequence of str or None
        Explicit feature key order. ``None`` sorts the keys of the first
        dict seen.

    Examples
    --------
    >>> import numpy as np
    >>> from dense_armor.utility.learn.online_dynamics import RecursiveLeastSquares
    >>> rls = RecursiveLeastSquares(lam=1.0, delta=1e10,
    ...                              feature_keys=["a", "b"])
    >>> X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    >>> y = X @ np.array([2.0, -3.0])
    >>> for i in range(3):
    ...     _ = rls.learn_one({"a": float(X[i, 0]), "b": float(X[i, 1])},
    ...                       float(y[i]))
    >>> bool(np.allclose(rls._w, np.array([2.0, -3.0]), atol=1e-6))
    True

    References
    ----------
    Ljung, L., Soderstrom, T. (1983). MIT Press.
    """

    def __init__(self, lam: float = 1.0, delta: float = 1e6,
                 feature_keys: Optional[Sequence[str]] = None):
        self.lam = lam
        self.delta = delta
        self.feature_keys = feature_keys
        self._keys: Optional[list] = None
        self._w: Optional[np.ndarray] = None
        self._P: Optional[np.ndarray] = None

    def _resolve_keys(self, x) -> list:
        if self._keys is not None:
            return self._keys
        if self.feature_keys is not None:
            self._keys = list(self.feature_keys)
        else:
            self._keys = sorted(x)
        return self._keys

    def _to_vec(self, x) -> np.ndarray:
        keys = self._resolve_keys(x)
        return np.array([float(x[k]) for k in keys], dtype=float)

    def _init(self, d: int) -> None:
        self._w = np.zeros(d)
        self._P = self.delta * np.eye(d)

    def learn_one(self, x, y):
        xv = self._to_vec(x)
        if self._w is None:
            self._init(xv.size)
        Px = self._P @ xv
        K = Px / (self.lam + xv @ Px)
        err = float(y) - float(self._w @ xv)
        self._w = self._w + K * err
        self._P = (self._P - np.outer(K, Px)) / self.lam
        self._P = 0.5 * (self._P + self._P.T)
        return self

    def predict_one(self, x) -> float:
        if self._w is None:
            return 0.0
        xv = self._to_vec(x)
        return float(self._w @ xv)

    def _unit_test_skips(self):
        return {"check_roc_auc"}


_FEATURE_NAMES = ("qdd", "qd", "sign_qd", "g", "one")


class ResidualDynamicsLearner:
    """Online residual torque learner over a nominal ``RigidBodyModel``.

    Features per joint j: ``[qdd_j, qd_j, sign(qd_j), g_j, 1]``. Target per
    joint j: ``tau_j - (M(q) qdd + C(q,qd) qd + g(q))_j``. One RLS per joint,
    learned independently.

    Parameters
    ----------
    model : RigidBodyModel
    lam : float
        Forgetting factor, passed to each per-joint RLS.
    delta : float
        RLS initial covariance scale.

    Examples
    --------
    >>> import numpy as np
    >>> from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
    >>> from dense_armor.utility.learn.online_dynamics import (
    ...     ResidualDynamicsLearner, write_minimal_urdf,
    ... )
    >>> path = write_minimal_urdf()
    >>> m = RigidBodyModel(path)
    >>> learner = ResidualDynamicsLearner(m)
    >>> q = np.zeros(m.n)
    >>> tau = learner.predict_torque(q, q, q)
    >>> tau.shape
    (2,)
    """

    def __init__(self, model: RigidBodyModel, lam: float = 1.0,
                 delta: float = 1e6):
        self.model = model
        self.n = model.n
        self.lam = lam
        self.delta = delta
        self._rls = [
            RecursiveLeastSquares(lam=lam, delta=delta,
                                  feature_keys=list(_FEATURE_NAMES))
            for _ in range(self.n)
        ]

    def _nominal_terms(self, q, qd, qdd):
        q_j = jnp.asarray(q)
        qd_j = jnp.asarray(qd)
        M = np.asarray(self.model.mass_matrix(q_j))
        C = np.asarray(self.model.bias_forces(q_j, qd_j))
        g = np.asarray(self.model.gravity_forces(q_j))
        tau_nominal = M @ np.asarray(qdd) + C + g
        return tau_nominal, g

    def _joint_features(self, j, qd, qdd, g):
        return {
            "qdd": float(qdd[j]),
            "qd": float(qd[j]),
            "sign_qd": float(np.sign(qd[j])),
            "g": float(g[j]),
            "one": 1.0,
        }

    def learn_one(self, q, qd, qdd, tau_measured):
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        qdd = np.asarray(qdd, dtype=float)
        tau = np.asarray(tau_measured, dtype=float)
        tau_nominal, g = self._nominal_terms(q, qd, qdd)
        residual = tau - tau_nominal
        for j in range(self.n):
            self._rls[j].learn_one(
                self._joint_features(j, qd, qdd, g), float(residual[j])
            )
        return self

    def predict_torque(self, q, qd, qdd) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        qdd = np.asarray(qdd, dtype=float)
        tau_nominal, g = self._nominal_terms(q, qd, qdd)
        residual = np.zeros(self.n)
        for j in range(self.n):
            residual[j] = self._rls[j].predict_one(
                self._joint_features(j, qd, qdd, g)
            )
        return tau_nominal + residual


class DriftAwareResidualDynamicsLearner(ResidualDynamicsLearner):
    """``ResidualDynamicsLearner`` with a per-joint drift trigger on the
    standardized prediction error.

    Trigger options: ``trigger="hampel"`` (default, Hampel on the causal
    window, reference-free) or ``trigger="cusum"``. When any joint's trigger
    fires, the RLS forgetting factor on all joints is temporarily lowered to
    ``fast_lam`` for ``warmup_steps`` steps, then restored to ``lam``.

    Honest result on the 2-link benchmark (payload/friction step at sample
    200 of 400, ``lam=0.99``): the Hampel trigger fires on pre-change RLS
    convergence noise around sample 154-157 (a false positive), the
    disturbance in the residual window then masks the actual change at 200,
    and the apparent improvement in the 10-60-sample post-change window is
    a timing artefact, not a mechanism. On the CUSUM path the trigger either
    never fires (``h=20``, ``h=10``) or fires as a false positive
    (``h=5``, events at 151 and 207). This class is kept as an API so users
    can try their own parameters on their own data, but for issue #40 the
    plain ``ResidualDynamicsLearner`` is the recommended default.

    Parameters
    ----------
    model : RigidBodyModel
    lam : float
        Slow (steady-state) forgetting factor.
    delta : float
        RLS initial covariance scale.
    fast_lam : float
        Fast forgetting factor used during ``warmup_steps`` after a drift.
    warmup_steps : int
        Number of samples after a drift during which ``fast_lam`` is used.
    trigger : str
        ``"hampel"`` (default) or ``"cusum"``.
    hampel_radius, hampel_n_sigmas
        Hampel trigger parameters.
    drift_radius, drift_ref_mult, drift_k, drift_h, drift_two_sided,
    drift_reference
        CUSUM trigger parameters.
    detector_warmup : int
        Number of initial samples during which the trigger is not fed.

    Examples
    --------
    >>> import numpy as np
    >>> from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
    >>> from dense_armor.utility.learn.online_dynamics import (
    ...     DriftAwareResidualDynamicsLearner, write_minimal_urdf,
    ... )
    >>> path = write_minimal_urdf()
    >>> m = RigidBodyModel(path)
    >>> learner = DriftAwareResidualDynamicsLearner(m)
    >>> tau = learner.predict_torque(np.zeros(m.n), np.zeros(m.n),
    ...                              np.zeros(m.n))
    >>> tau.shape
    (2,)
    """

    def __init__(self, model: RigidBodyModel, lam: float = 0.99,
                 delta: float = 1e6, fast_lam: float = 0.9,
                 warmup_steps: int = 20, drift_radius: int = 10,
                 drift_ref_mult: int = 3, drift_k: float = 0.5,
                 drift_h: float = 20.0, drift_two_sided: bool = True,
                 drift_reference: str = "adaptive",
                 detector_warmup: int = 0,
                 trigger: str = "hampel",
                 hampel_radius: int = 20,
                 hampel_n_sigmas: float = 3.0):
        super().__init__(model, lam=lam, delta=delta)
        if trigger not in ("hampel", "cusum"):
            raise ValueError(
                f"trigger must be 'hampel' or 'cusum', got {trigger!r}"
            )
        self.fast_lam = fast_lam
        self.warmup_steps = warmup_steps
        self.drift_radius = drift_radius
        self.drift_ref_mult = drift_ref_mult
        self.drift_k = drift_k
        self.drift_h = drift_h
        self.drift_two_sided = drift_two_sided
        self.drift_reference = drift_reference
        self.detector_warmup = detector_warmup
        self.trigger = trigger
        self.hampel_radius = hampel_radius
        self.hampel_n_sigmas = hampel_n_sigmas
        self._countdown: int = 0
        self._seen: int = 0
        if trigger == "hampel":
            self._drift_detectors = [
                HampelScorer(radius=hampel_radius, n_sigmas=hampel_n_sigmas)
                for _ in range(self.n)
            ]
        else:
            self._drift_detectors = [
                CUSUMDriftDetector(radius=drift_radius,
                                   ref_mult=drift_ref_mult,
                                   k=drift_k, h=drift_h,
                                   two_sided=drift_two_sided,
                                   reference=drift_reference)
                for _ in range(self.n)
            ]

    def _set_lam(self, value: float) -> None:
        for rls in self._rls:
            rls.lam = value

    def learn_one(self, q, qd, qdd, tau_measured):
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        qdd = np.asarray(qdd, dtype=float)
        tau = np.asarray(tau_measured, dtype=float)
        tau_nominal, g = self._nominal_terms(q, qd, qdd)
        residual = tau - tau_nominal
        self._seen += 1
        drift_seen = False
        for j in range(self.n):
            feats = self._joint_features(j, qd, qdd, g)
            pred = self._rls[j].predict_one(feats)
            err = float(residual[j]) - pred
            if self._seen > self.detector_warmup:
                det = self._drift_detectors[j]
                if self.trigger == "hampel":
                    if det.is_outlier({"v": err}):
                        drift_seen = True
                    det.learn_one({"v": err})
                else:
                    det.update(err)
                    if det.drift_detected:
                        drift_seen = True
            self._rls[j].learn_one(feats, float(residual[j]))
        if drift_seen:
            self._countdown = self.warmup_steps
        if self._countdown > 0:
            self._set_lam(self.fast_lam)
            self._countdown -= 1
        else:
            self._set_lam(self.lam)
        return self


def write_minimal_urdf(path: Optional[str] = None,
                       payload_mass: float = 0.0) -> str:
    """Write a minimal 2-link revolute URDF, optionally with an extra point
    mass at link2's own frame origin. Returns the path.
    """
    if path is None:
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".urdf")
        os.close(fd)
    m2 = 0.5 + payload_mass
    xml = f"""<robot name="two_link">
  <link name="base">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <link name="link1">
    <inertial>
      <origin xyz="0.5 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/>
    </inertial>
  </link>
  <link name="link2">
    <inertial>
      <origin xyz="0.5 0 0" rpy="0 0 0"/>
      <mass value="{m2}"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/>
    </inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base"/>
    <child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="10.0"/>
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="10.0"/>
  </joint>
</robot>"""
    Path(path).write_text(xml, encoding="utf-8")
    return path
