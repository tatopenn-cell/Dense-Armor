"""Forgetting per sample and first-sample skip in PayloadEstimator."""

from pathlib import Path

import numpy as np

from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
from dense_armor.utility.datasets.synthetic_arm import SyntheticArm
from dense_armor.utility.linear.payload import PayloadEstimator
import pytest

URDF = str(Path(__file__).parent.parent / "fixtures/urdf/panda.urdf")


@pytest.mark.heavy
def test_lam_is_applied_once_per_sample_not_once_per_joint():
    """With lam=0.99 the effective memory is ~100 samples, not ~12.

    Before the fix the covariance P was divided by lam inside the
    joint loop, so with n joints the effective factor per sample was
    lam**n. On the panda (9 joints) lam**9 = 0.9135 and the memory
    was about 11 samples. After the fix the factor per sample is lam,
    and the memory is 1 / (1 - lam) = 100 samples.
    """
    ds = SyntheticArm(
        URDF,
        period_s=2.0,
        rate_hz=50.0,
        n_cycles=1,
        fault_at_s=0.5,
        fault="payload",
        payload_mass=5.0,
        seed=0,
    )
    nominal = RigidBodyModel(URDF)
    est = PayloadEstimator(nominal, link=ds.payload_link, lam=0.99, delta=1e6)

    for sig, _ in ds.stream():
        est.learn_one(sig)

    n_joints = est._n
    factor_per_sample = est.lam
    memory_samples = 1.0 / (1.0 - factor_per_sample)

    assert n_joints == 9
    assert 0.98 < factor_per_sample < 1.0
    assert 90.0 < memory_samples < 110.0


def test_first_sample_does_not_update_the_mass():
    """The first sample has no previous qd: no qdd, so no update.

    Before the fix the first sample was used with qdd = 0 and the
    estimate jumped to 4.40 kg at t = 0 on the payload stream. After
    the fix the estimate at the first sample is 0.
    """
    ds = SyntheticArm(
        URDF,
        period_s=2.0,
        rate_hz=50.0,
        n_cycles=1,
        fault_at_s=0.5,
        fault="payload",
        payload_mass=5.0,
        seed=0,
    )
    nominal = RigidBodyModel(URDF)
    est = PayloadEstimator(nominal, link=ds.payload_link, lam=0.99, delta=1e6)

    first_sig = None
    for sig, _ in ds.stream():
        first_sig = sig
        break

    est.learn_one(first_sig)
    assert est._m == 0.0
    assert est.predict_one({}) == 0.0
    # the state was still updated to remember qd and t
    assert est._prev_qd is not None
    assert est._prev_t is not None


@pytest.mark.heavy
def test_estimate_stays_zero_before_the_fault():
    """Before the fault the internal _m can be negative, predict_one clamps."""
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

    before = masses[times < 2.0]
    assert np.abs(before).max() < 0.01
