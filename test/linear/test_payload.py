"""Tests for the point-mass payload estimator."""

import doctest
from pathlib import Path

import numpy as np
import pytest

from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
from dense_armor.utility.datasets.synthetic_arm import SyntheticArm
from dense_armor.utility.linear import payload as payload_mod
from dense_armor.utility.linear.payload import PayloadEstimator

URDF = str(Path(__file__).parent.parent / "fixtures/urdf/panda.urdf")


def test_doctests():
    r = doctest.testmod(payload_mod)
    assert r.failed == 0


@pytest.mark.slow
def test_synthetic_payload_fault():
    ds = SyntheticArm(
        URDF,
        period_s=2.0,
        rate_hz=50.0,
        n_cycles=4,
        fault_at_s=2.0,
        fault="payload",
        payload_mass=5.0,
        seed=0,
    )
    nominal = RigidBodyModel(URDF)
    est = PayloadEstimator(nominal, link=ds.payload_link, lam=0.99, delta=1e6)

    times, masses = [], []
    for sig, _ in ds.stream():
        est.learn_one(sig)
        times.append(sig.t)
        masses.append(est.predict_one({}))
    times = np.asarray(times)
    masses = np.asarray(masses)

    after = masses[times >= 2.0]
    assert after.size > 0
    assert 0.5 * 5.0 < after.mean() < 1.5 * 5.0

    settled = np.where((times >= 2.0) & (np.abs(masses - 5.0) <= 0.5))[0]
    assert settled.size > 0
    t_settle = float(times[settled[0]] - 2.0)
    assert t_settle < 5.0


@pytest.mark.slow
def test_clone_runs_same_stream():
    ds = SyntheticArm(
        URDF,
        period_s=2.0,
        rate_hz=50.0,
        n_cycles=1,
        fault_at_s=1.0,
        fault="payload",
        payload_mass=3.0,
        seed=1,
    )
    nominal = RigidBodyModel(URDF)
    est = PayloadEstimator(nominal, link=ds.payload_link, lam=0.99)

    def run(e):
        out = []
        for sig, _ in ds.stream():
            e.learn_one(sig)
            out.append(e.predict_one({}))
        return np.asarray(out)

    a = run(est)
    b = run(est.clone())
    np.testing.assert_allclose(a, b)
