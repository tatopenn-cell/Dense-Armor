"""Labelled joint stream from a real URDF, with a fault after a chosen time.

The arm follows a periodic sinusoidal trajectory inside the joint
limits of the URDF. Torques come from inverse dynamics,

    tau = mass_matrix(q) @ qdd + bias_forces(q, qd) + gravity_forces(q),

computed by :class:`~dense_armor.dynamics.urdf_dynamics.RigidBodyModel`
from the same URDF file. Gaussian sensor noise is added to ``q``, ``qd``
and ``tau``. From ``fault_at_s`` a fault adds a torque:

- ``"friction"``: a viscous term ``b * qd[j]`` on one joint;
- ``"payload"``: an extra mass ``m`` at the origin of the last link of
  the chain, whose gravity torque is ``m * g * J_z(q)``, with ``J_z`` the
  vertical row of the link's translational Jacobian (the same
  ``V = m g z`` convention as ``gravity_forces``).

The label ``y`` is ``0`` before ``fault_at_s`` and ``1`` after.

Examples:
    >>> from dense_armor.utility.datasets import SyntheticArm
    >>> ds = SyntheticArm(  # doctest: +SKIP
    ...     "test/fixtures/urdf/panda.urdf",
    ...     period_s=1.0, rate_hz=50.0,
    ...     fault_at_s=0.5, fault="payload", seed=0,
    ... )
"""

from collections.abc import Iterator
from pathlib import Path

import jax
import numpy as np

from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
from dense_armor.roles import Signal

_DYNAMICS: dict[tuple[str, str], tuple] = {}


def _dynamics(urdf_path: Path, payload_link: str | None) -> tuple:
    """Model and jitted dynamics for a URDF, compiled once per process."""
    model = RigidBodyModel(urdf_path)
    if payload_link is None:
        anc = model.link_ancestor_dofs

        def depth(name: str) -> tuple[int, int]:
            rev = sum(1 for j, _, _ in anc[name] if j["type"] != "prismatic")
            return rev, -len(anc[name])

        payload_link = max(anc, key=depth)
    if payload_link not in model.link_ancestor_dofs:
        raise ValueError(f"unknown payload_link {payload_link!r}")
    key = (str(Path(urdf_path).resolve()), payload_link)
    if key not in _DYNAMICS:
        link = payload_link
        _DYNAMICS[key] = (
            model,
            link,
            jax.jit(model.mass_matrix),
            jax.jit(model.bias_forces),
            jax.jit(model.gravity_forces),
            jax.jit(lambda q: model.link_jacobian(q, link)),
        )
    return _DYNAMICS[key]


def _joint_limits(urdf_path: str | Path, n: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        from dense_armor.roles import limits_from_urdf

        lim = limits_from_urdf(urdf_path)
        lower = np.asarray(getattr(lim, "lower", None), dtype=float)
        upper = np.asarray(getattr(lim, "upper", None), dtype=float)
        if lower.shape == (n,) and upper.shape == (n,):
            return lower, upper
    except (AttributeError, TypeError, ValueError):
        pass
    return np.full(n, -0.5), np.full(n, 0.5)


class SyntheticArm:
    """Periodic joint stream from a URDF with an optional fault.

    Args:
        urdf_path: URDF readable by
            :class:`~dense_armor.dynamics.urdf_dynamics.RigidBodyModel`.
        period_s: period of one sinusoidal cycle, in seconds.
        rate_hz: sampling rate, in Hz.
        n_cycles: number of cycles generated.
        fault_at_s: time from the start when the fault begins, in
            seconds. ``None`` means no fault.
        fault: ``"friction"`` or ``"payload"``. Required when
            ``fault_at_s`` is given.
        fault_joint: index of the joint with the ``"friction"`` fault. Default: the
            last joint.
        friction_b: viscous coefficient for ``"friction"``, in N m s.
        payload_mass: mass added for ``"payload"``, in kg.
        payload_link: link carrying the payload. Default: the last link
            of the longest chain.
        noise_std: standard deviation of the Gaussian sensor noise on
            ``q`` (rad), ``qd`` (rad/s) and ``tau`` (N m).
        seed: seed of the random generator.

    Raises:
        ValueError: on bad arguments.

    Examples:
        >>> from dense_armor.utility.datasets import SyntheticArm
        >>> ds = SyntheticArm(  # doctest: +SKIP
        ...     "test/fixtures/urdf/panda.urdf",
        ...     period_s=1.0, rate_hz=50.0,
        ...     fault_at_s=0.5, fault="payload", seed=0,
        ... )
    """

    def __init__(
        self,
        urdf_path: str | Path,
        period_s: float,
        rate_hz: float,
        n_cycles: int = 4,
        fault_at_s: float | None = None,
        fault: str | None = None,
        fault_joint: int | None = None,
        friction_b: float = 20.0,
        payload_mass: float = 2.0,
        payload_link: str | None = None,
        noise_std: tuple[float, float, float] = (1e-3, 5e-3, 5e-2),
        seed: int | None = 0,
    ) -> None:
        if period_s <= 0:
            raise ValueError(f"period_s must be > 0, got {period_s}")
        if rate_hz <= 0:
            raise ValueError(f"rate_hz must be > 0, got {rate_hz}")
        if n_cycles < 1:
            raise ValueError(f"n_cycles must be >= 1, got {n_cycles}")
        if fault_at_s is not None and fault not in ("friction", "payload"):
            raise ValueError(f"fault must be 'friction' or 'payload', got {fault!r}")
        self.urdf_path = Path(urdf_path)
        self.period_s = float(period_s)
        self.rate_hz = float(rate_hz)
        self.n_cycles = n_cycles
        self.fault_at_s = fault_at_s
        self.fault = fault
        self.friction_b = float(friction_b)
        self.payload_mass = float(payload_mass)
        self.noise_std = tuple(noise_std)
        self.seed = seed
        (
            self.model,
            self.payload_link,
            self._mass_matrix,
            self._bias,
            self._gravity,
            self._jac,
        ) = _dynamics(self.urdf_path, payload_link)
        self.n_joints = int(self.model.n)
        if fault_joint is None:
            fault_joint = self.n_joints - 1
        if not 0 <= fault_joint < self.n_joints:
            raise ValueError(
                f"fault_joint must be in [0, {self.n_joints}), got {fault_joint}"
            )
        self.fault_joint = int(fault_joint)
        self.q_min, self.q_max = _joint_limits(self.urdf_path, self.n_joints)
        span = np.maximum(self.q_max - self.q_min, 1e-3)
        self.amplitude = 0.3 * span
        self.mid = 0.5 * (self.q_min + self.q_max)
        freqs = 1.0 + 0.25 * np.arange(self.n_joints)
        phases = 0.5 * np.arange(self.n_joints)
        self.freqs = freqs
        self.phases = phases
        self.n_samples = round(n_cycles * period_s * rate_hz)
        self.dt = 1.0 / rate_hz

    def trajectory(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sinusoidal position, velocity and acceleration at ``t``.

        Args:
            t: time from the start, in seconds.

        Returns:
            ``(q, qd, qdd)``, three arrays of shape ``(n_joints,)``.
        """
        w = 2.0 * np.pi / self.period_s
        phase = w * self.freqs * t + self.phases
        q = self.mid + self.amplitude * np.sin(phase)
        qd = self.amplitude * self.freqs * w * np.cos(phase)
        qdd = -self.amplitude * self.freqs**2 * w**2 * np.sin(phase)
        return q, qd, qdd

    def nominal_torque(
        self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray
    ) -> np.ndarray:
        """Inverse dynamics of the arm, without noise or fault.

        Args:
            q: joint positions.
            qd: joint velocities.
            qdd: joint accelerations.

        Returns:
            ``mass_matrix(q) @ qdd + bias_forces(q, qd)
            + gravity_forces(q)``.
        """
        M = np.asarray(self._mass_matrix(q), dtype=float)
        b = np.asarray(self._bias(q, qd), dtype=float)
        g = np.asarray(self._gravity(q), dtype=float)
        return M @ qdd + b + g

    def _fault_term(self, q: np.ndarray, qd: np.ndarray, t: float) -> np.ndarray:
        out = np.zeros(self.n_joints)
        if self.fault_at_s is None or t < self.fault_at_s:
            return out
        if self.fault == "friction":
            out[self.fault_joint] = self.friction_b * qd[self.fault_joint]
            return out
        return self.payload_torque(q)

    def payload_torque(self, q: np.ndarray) -> np.ndarray:
        """Gravity torque of the payload mass, ``m * g * J_z(q)``."""
        jz = np.asarray(self._jac(q), dtype=float)[2]
        return self.payload_mass * 9.81 * jz

    def stream(self) -> Iterator[tuple[Signal, int]]:
        """Yield ``(Signal, y)`` pairs at ``rate_hz``.

        Each :class:`Signal` has channels ``q_j``, ``qd_j``, ``tau_j``
        for every joint; ``t`` is the sample time in seconds; ``y`` is
        ``0`` before ``fault_at_s`` and ``1`` after.

        Yields:
            ``(signal, label)`` pairs in time order.
        """
        rng = np.random.default_rng(self.seed)
        names: list[str] = []
        units: list[str] = []
        for j in range(self.n_joints):
            names += [f"q_{j}", f"qd_{j}", f"tau_{j}"]
            units += ["rad", "rad/s", "N*m"]
        for i in range(self.n_samples):
            t = i * self.dt
            q, qd, qdd = self.trajectory(t)
            tau = self.nominal_torque(q, qd, qdd) + self._fault_term(q, qd, t)
            vals: list[float] = []
            for j in range(self.n_joints):
                vals.append(q[j] + rng.normal(0.0, self.noise_std[0]))
                vals.append(qd[j] + rng.normal(0.0, self.noise_std[1]))
                vals.append(tau[j] + rng.normal(0.0, self.noise_std[2]))
            sig = Signal(
                values=np.asarray(vals, dtype=float),
                names=names,
                units=units,
                t=t,
            )
            y = 0
            if self.fault_at_s is not None and t >= self.fault_at_s:
                y = 1
            yield sig, y
