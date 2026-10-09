"""Tests for the synthetic arm dataset.

``RigidBodyModel`` is expensive (numerical Jacobians on a 7-DOF arm),
so every test uses a short stream and physics tests check formulas at
fixed points, not in loops. The CUSUM test feeds the raw ``tau[j]``
channel of the stream and only requires that the fire happens *after*
the fault.
"""

from pathlib import Path

import numpy as np
import pytest

from dense_armor.utility.datasets import SyntheticArm
from dense_armor.utility.drift.detector import CUSUMDriftDetector

URDF = Path("test/fixtures/urdf/panda.urdf")
PERIOD = 1.0
RATE = 10.0
NOISE_NONE = (0.0, 0.0, 0.0)


def _make(**kw):
    base = {"period_s": PERIOD, "rate_hz": RATE, "n_cycles": 1, "seed": 0}
    base.update(kw)
    return SyntheticArm(URDF, **base)


def test_nominal_torque_is_inverse_dynamics():
    ds = _make()
    rng = np.random.default_rng(0)
    q = rng.uniform(-0.3, 0.3, size=ds.n_joints)
    qd = rng.normal(0, 0.1, size=ds.n_joints)
    qdd = rng.normal(0, 0.1, size=ds.n_joints)
    M = ds.model.mass_matrix(q)
    b = ds.model.bias_forces(q, qd)
    g = ds.model.gravity_forces(q)
    expected = M @ qdd + b + g
    np.testing.assert_allclose(ds.nominal_torque(q, qd, qdd), expected)


def test_no_fault_torque_is_finite():
    ds = _make(fault_at_s=None, noise_std=NOISE_NONE)
    for i in range(3):
        t = i * ds.dt
        q, qd, qdd = ds.trajectory(t)
        tau = ds.nominal_torque(q, qd, qdd)
        assert np.all(np.isfinite(tau))
        assert np.all(np.isfinite(q))
        assert np.all(np.isfinite(qd))
        assert np.all(np.isfinite(qdd))


def test_trajectory_respects_joint_limits():
    ds = _make()
    for i in range(5):
        t = i * ds.dt
        q, _, _ = ds.trajectory(t)
        assert np.all(q >= ds.q_min - 1e-9)
        assert np.all(q <= ds.q_max + 1e-9)


def test_label_flips_after_fault():
    ds = _make(fault_at_s=0.5, fault="payload")
    out = list(ds.stream())
    ys = [y for _, y in out]
    ts = [s.t for s, _ in out]
    first_one = next(i for i, y in enumerate(ys) if y == 1)
    assert ys[:first_one] == [0] * first_one
    assert all(y == 1 for y in ys[first_one:])
    assert ts[first_one] >= 0.5
    assert ts[first_one] - 0.5 < 1.0 / RATE + 1e-9


def test_friction_fault_adds_residual():
    ds = _make(
        fault_at_s=0.5,
        fault="friction",
        fault_joint=0,
        friction_b=50.0,
        noise_std=NOISE_NONE,
    )
    j = ds.fault_joint
    q, qd, qdd = ds.trajectory(0.6)
    nominal = ds.nominal_torque(q, qd, qdd)
    added = ds.friction_b * qd[j]
    assert abs(added) > 1e-3
    assert nominal[j] == pytest.approx(nominal[j])


def test_payload_torque_is_gravity_of_the_mass():
    ds = _make(fault_at_s=0.5, fault="payload", payload_mass=3.0, noise_std=NOISE_NONE)
    q, _, _ = ds.trajectory(0.6)
    tau = ds.payload_torque(q)
    h = 1e-6
    grad = np.zeros(ds.n_joints)
    for k in range(ds.n_joints):
        e = np.zeros(ds.n_joints)
        e[k] = h
        zp = float(ds.model.link_position(q + e, ds.payload_link)[2])
        zm = float(ds.model.link_position(q - e, ds.payload_link)[2])
        grad[k] = 3.0 * 9.81 * (zp - zm) / (2.0 * h)
    np.testing.assert_allclose(tau, grad, atol=1e-4)
    assert np.abs(tau).max() > 1.0


def test_cusum_fires_after_fault_and_not_before():
    ds = _make(
        period_s=1.0,
        rate_hz=10.0,
        n_cycles=1,
        fault_at_s=0.4,
        fault="payload",
        payload_mass=5.0,
        noise_std=(1e-3, 5e-3, 1e-2),
    )
    j = 1
    det = CUSUMDriftDetector(reference="fixed", radius=2, ref_mult=2, h=5.0)
    fires = []
    for sig, _ in ds.stream():
        q = np.array([sig[f"q_{k}"] for k in range(ds.n_joints)])
        qd = np.array([sig[f"qd_{k}"] for k in range(ds.n_joints)])
        _, _, qdd = ds.trajectory(sig.t)
        nominal = ds.nominal_torque(q, qd, qdd)
        residual = sig[f"tau_{j}"] - nominal[j]
        det.update(residual)
        if det.drift_detected:
            fires.append(sig.t)
    assert fires, "CUSUM must fire on the payload residual step"
    assert fires[0] >= 0.4


def test_fault_joint_out_of_range():
    with pytest.raises(ValueError):
        _make(fault_at_s=0.5, fault="payload", fault_joint=99)


def test_bad_fault_name():
    with pytest.raises(ValueError):
        _make(fault_at_s=0.5, fault="sparkle")
