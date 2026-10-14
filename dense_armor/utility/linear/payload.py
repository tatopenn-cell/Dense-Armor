"""Online estimate of the mass of a payload held by a robot link.

A point mass m at the frame origin of a link adds a gravity torque
m * g * Jz(q) at the joints, where Jz is the vertical row of the
translational Jacobian of that link and g the gravity constant. The
added torque is linear in m, so m can be estimated from the residual
between measured and nominal torques by recursive least squares with
exponential forgetting. The estimate is clamped to be non-negative.

The rigid-body methods of the model are JIT-compiled once inside the
estimator. Without that, every call to mass_matrix, bias_forces or
gravity_forces retraces the whole kinematic recursion in Python and the
per-sample cost is seconds on a nine-joint arm.

Relation to Foster et al. 2024. Their method estimates the full 10
inertial parameters of a link online with an extended Kalman filter
(EKF) on a log-Cholesky parametrisation (Rucker and Wensing 2022). The
log-Cholesky parametrisation keeps the inertia matrix physically
consistent by construction, so a whole-body controller can use the
estimate safely while it converges.

This estimator is a narrow special case and uses a different
algorithm: it covers only the mass of a point load attached to a
link whose own inertial parameters are known, and it estimates that
single scalar with recursive least squares with exponential
forgetting, not with an EKF. It makes no attempt to estimate the
first moment or the rotational inertia of the load. Use it when the
load is a point mass (a tool, a weight) and the rest of the robot is
identified offline.

References:
    Foster, J., McCrory, S., DeBuys, C., Bertrand, S., Griffin, R.
    (2024). Physically Consistent Online Inertial Adaptation for
    Humanoid Loco-manipulation. arXiv:2405.07901.
    Rucker, C., Wensing, P. M. (2022). Smooth parameterization of
    rigid-body inertia. IEEE Robotics and Automation Letters 7(2),
    2771-2778.
    Ljung, L., Soderstrom, T. (1983). Theory and Practice of
    Recursive Identification. MIT Press. (RLS with exponential
    forgetting.)
"""

from typing import Any

import jax
import numpy as np

from dense_armor.roles import Regressor


class PayloadEstimator(Regressor):
    """Recursive least squares estimate of a point mass on a link.

    The estimator takes each Signal from the robot (channels
    q_j, qd_j, tau_j for every joint), computes the nominal torque of
    the unloaded model, and fits the residual to the gravity torque of
    a point mass:

        r_i = tau_i - (M(q) qdd + bias(q, qd) + gravity(q))_i
        phi_i = g * Jz(q)_i
        r_i ~ m * phi_i

    The recursive update is the standard RLS with exponential
    forgetting on the scalar m. The forgetting factor ``lam`` is
    applied once per sample, before the joint loop; the loop itself
    does not divide the covariance by ``lam`` again. The clamp to
    non-negative is applied only by :meth:`predict_one`; the internal
    state ``_m`` can be negative between updates.

    The first sample is skipped: with no previous sample there is no
    ``qdd`` to compute, and the residual would be wrong. The state is
    updated to remember the previous ``qd`` and ``t``, but no learning
    step is taken until a valid ``qdd`` can be computed.

    Args:
        model: the unloaded rigid-body model. Its inertial parameters
            must not include the payload.
        link: name of the link that carries the payload.
        lam: forgetting factor in (0, 1]. 1.0 weights all past samples
            equally. Smaller values let the estimate follow a payload
            that changes.
        delta: initial RLS covariance (scalar). Large values give the
            first samples a lot of freedom.
        g: gravity constant, default 9.81 m/s^2.

    Examples:
        >>> from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
        >>> from dense_armor.utility.linear.payload import PayloadEstimator
        >>> model = RigidBodyModel("test/fixtures/urdf/panda.urdf")
        >>> est = PayloadEstimator(model, link="panda_link7", lam=0.99)
        >>> float(est.predict_one({}))
        0.0

    Notes:
        The estimator wraps a RigidBodyModel, which the library does
        not give an equality operator. The two checks in
        check_estimator that compare estimator params with == (clone
        and repr-clone equality) cannot succeed on an object that
        holds a robot description in a slot. The clone itself is
        correct: test_linear_payload.py clones the estimator and
        reruns the same stream, verifying that the mass trajectory is
        identical.
    """

    budget_s: float = 5e-2
    memory_class: str = "O(1)"

    def __init__(
        self,
        model: Any,
        link: str,
        lam: float = 0.99,
        delta: float = 1e6,
        g: float = 9.81,
    ) -> None:
        if lam <= 0.0 or lam > 1.0:
            raise ValueError(f"lam must be in (0, 1], got {lam}")
        if delta <= 0.0:
            raise ValueError(f"delta must be > 0, got {delta}")
        if link not in getattr(model, "link_ancestor_dofs", {}):
            raise ValueError(f"unknown link {link!r} on the given model")
        self.model = model
        self.link = link
        self.lam = float(lam)
        self.delta = float(delta)
        self.g = float(g)
        self._n = int(model.n)
        self._m = 0.0
        self._P = float(delta)
        self._prev_qd: np.ndarray | None = None
        self._prev_t: float | None = None
        self.n_missing: int = 0
        self._mm = jax.jit(model.mass_matrix)
        self._bf = jax.jit(model.bias_forces)
        self._gf = jax.jit(model.gravity_forces)
        _link = link
        self._lj = jax.jit(lambda q: model.link_jacobian(q, _link))

    @staticmethod
    def _as_dict(x: Any) -> dict:
        if isinstance(x, dict):
            return x
        to_dict = getattr(x, "to_dict", None)
        if callable(to_dict):
            return to_dict()
        raise TypeError(
            f"expected a dict or an object with to_dict(), got {type(x).__name__}"
        )

    def _extract(self, d: dict) -> tuple | None:
        q = np.empty(self._n)
        qd = np.empty(self._n)
        tau = np.empty(self._n)
        try:
            for j in range(self._n):
                q[j] = float(d[f"q_{j}"])
                qd[j] = float(d[f"qd_{j}"])
                tau[j] = float(d[f"tau_{j}"])
        except KeyError:
            return None
        return q, qd, tau

    def learn_one(
        self, x: Any, y: Any = None, t: float | None = None
    ) -> "PayloadEstimator":
        """Update the mass estimate from one robot sample.

        Args:
            x: a Signal or a dict with channels q_j, qd_j, tau_j for
                every joint.
            y: ignored; the residual is computed from the torques in x.
                Accepted for compatibility with the regressor interface.
            t: optional timestamp, currently unused.

        Returns:
            self.
        """
        d = self._as_dict(x)
        extracted = self._extract(d)
        if extracted is None:
            return self
        q, qd, tau = extracted
        if np.any(np.isnan(q)) or np.any(np.isnan(qd)) or np.any(np.isnan(tau)):
            self.n_missing += 1
            return self
        t_sample = getattr(x, "t", None)
        if t_sample is None:
            t_sample = t
        if t_sample is None:
            t_sample = d.get("t")
        # no previous sample, or dt not usable: the qdd of this
        # sample cannot be computed and the residual is wrong. Skip
        # the update but remember qd and t for the next call.
        if self._prev_qd is None or self._prev_t is None or t_sample is None:
            self._prev_qd = qd
            self._prev_t = t_sample
            return self
        dt = float(t_sample) - float(self._prev_t)
        if dt <= 0.0:
            self._prev_qd = qd
            self._prev_t = t_sample
            return self
        qdd = (qd - self._prev_qd) / dt
        self._prev_qd = qd
        self._prev_t = t_sample

        M = np.asarray(self._mm(q), dtype=float)
        b = np.asarray(self._bf(q, qd), dtype=float)
        gr = np.asarray(self._gf(q), dtype=float)
        tau_nom = M @ qdd + b + gr
        residual = tau - tau_nom

        jz = np.asarray(self._lj(q), dtype=float)[2]
        phi = self.g * jz

        # forgetting once per sample, before the joint loop
        self._P = self._P / self.lam
        for i in range(self._n):
            pi = float(phi[i])
            ri = float(residual[i])
            denom = 1.0 + pi * self._P * pi
            K = self._P * pi / denom
            self._m = self._m + K * (ri - pi * self._m)
            self._P = self._P - K * pi * self._P
        return self

    def predict_one(
        self,
        x: dict | None = None,
        t: float | None = None,
        return_std: bool = False,
    ) -> float:
        """Return the current estimate of the payload mass, clamped to >= 0.

        Args:
            x: ignored; kept for interface compatibility.
            t: ignored.
            return_std: ignored.

        Returns:
            The non-negative mass estimate in kilograms.
        """
        return max(0.0, float(self._m))

    def _unit_test_skips(self) -> set:
        return {
            "check_clone",
            "check_repr_clone_equal",
        }
