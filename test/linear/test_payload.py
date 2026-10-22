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


@pytest.mark.heavy
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

    # First sample: the estimator skips it, no qdd yet.
    assert masses[0] == 0.0

    # Before the fault the estimate stays at the clamp floor.
    before = masses[times < 2.0]
    assert np.abs(before).max() < 0.01

    # After the fault. lam=0.99 now means 100 samples of memory
    # (2 s at 50 Hz), not 11, so the convergence is slower than with
    # the old per-joint forgetting. Measured on this stream:
    #   final       4.9555
    #   mean        3.7974
    #   min         0.0000
    #   max         5.0894
    #   within +-10% of 5 kg: 36.7% of the post-fault samples
    #   stays within +-10% from t = 3.80 s after the fault
    after = masses[times >= 2.0]
    assert after.size > 0
    assert 3.0 < after.mean() < 5.0
    assert 4.5 < after[-1] < 5.5

    within_10 = np.abs(after - 5.0) <= 0.5
    stay_start = None
    for k in range(len(after)):
        if within_10[k:].all():
            stay_start = k
            break
    assert stay_start is not None
    t_stay = float(times[times >= 2.0][stay_start] - 2.0)
    assert t_stay < 5.0


@pytest.mark.heavy
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
