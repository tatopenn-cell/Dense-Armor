"""NaN handling: skip the sample and count it in n_missing.

Every estimator with the pattern: a NaN in any feature or in the
target does not change the state; the sample is skipped and
n_missing is incremented. Prediction after the NaN is the same as
before.
"""

from pathlib import Path

import numpy as np

from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
from dense_armor.utility.datasets.synthetic_arm import SyntheticArm
from dense_armor.utility.linear.bayes import BayesianLinearRegression
from dense_armor.utility.linear.glm import LinearRegression, LogisticRegression
from dense_armor.utility.linear.payload import PayloadEstimator
from dense_armor.utility.optim import SGD, Adam

URDF = str(Path(__file__).parent.parent / "fixtures/urdf/panda.urdf")


def test_linear_regression_nan_is_skipped_and_counted():
    reg = LinearRegression(optimizer=SGD(lr=0.1), intercept=True)
    for _ in range(10):
        reg.learn_one({"a": 1.0, "b": 2.0}, 3.0)
    pred_before = reg.predict_one({"a": 1.0, "b": 2.0})
    w_before = reg._w.copy()

    reg.learn_one({"a": float("nan"), "b": 2.0}, 3.0)
    reg.learn_one({"a": 1.0, "b": float("nan")}, 3.0)
    reg.learn_one({"a": 1.0, "b": 2.0}, float("nan"))

    assert reg.n_missing == 3
    assert np.array_equal(reg._w, w_before)
    assert reg.predict_one({"a": 1.0, "b": 2.0}) == pred_before


def test_logistic_regression_nan_is_skipped_and_counted():
    clf = LogisticRegression(optimizer=SGD(lr=0.1), intercept=True)
    for _ in range(10):
        clf.learn_one({"a": 1.0, "b": 2.0}, 1)
    p_before = clf.predict_proba_one({"a": 1.0, "b": 2.0})
    w_before = clf._w.copy()

    clf.learn_one({"a": float("nan"), "b": 2.0}, 1)
    clf.learn_one({"a": 1.0, "b": 2.0}, float("nan"))

    assert clf.n_missing == 2
    assert np.array_equal(clf._w, w_before)
    assert clf.predict_proba_one({"a": 1.0, "b": 2.0}) == p_before


def test_bayesian_linear_regression_nan_is_skipped_and_counted():
    m = BayesianLinearRegression(alpha=1.0, beta=1.0, intercept=True)
    for _ in range(10):
        m.learn_one({"a": 1.0, "b": 2.0}, 3.0)
    pred_before = m.predict_one({"a": 1.0, "b": 2.0})
    m_before = m._m.copy()
    S_before = m._S.copy()

    m.learn_one({"a": float("nan"), "b": 2.0}, 3.0)
    m.learn_one({"a": 1.0, "b": 2.0}, float("nan"))

    assert m.n_missing == 2
    assert np.array_equal(m._m, m_before)
    assert np.array_equal(m._S, S_before)
    assert m.predict_one({"a": 1.0, "b": 2.0}) == pred_before


def test_pure_step_nan_is_skipped():
    reg = LinearRegression(optimizer=Adam(lr=0.05), intercept=True)
    state = reg.init_state(dim=2)

    state, _ = reg.step(state, {"a": 1.0, "b": 2.0, "y": 3.0})
    w_after_valid = np.asarray(state[0]).copy()

    state, _ = reg.step(state, {"a": float("nan"), "b": 2.0, "y": 3.0})
    w_after_nan = np.asarray(state[0])

    assert np.array_equal(w_after_valid, w_after_nan)


def test_payload_nan_is_skipped_and_counted():
    ds = SyntheticArm(
        URDF,
        period_s=2.0,
        rate_hz=50.0,
        n_cycles=1,
        fault_at_s=10.0,
        fault="payload",
        payload_mass=5.0,
        seed=0,
    )
    nominal = RigidBodyModel(URDF)
    est = PayloadEstimator(nominal, link=ds.payload_link, lam=0.99)

    sigs = []
    for sig, _ in ds.stream():
        sigs.append(sig)
        if len(sigs) >= 5:
            break

    for sig in sigs:
        est.learn_one(sig)
    mass_before = est.predict_one({})
    n_before = est.n_missing

    d = sigs[-1].to_dict()
    d["q_0"] = float("nan")
    est.learn_one(d)
    assert est.n_missing == n_before + 1
    assert est.predict_one({}) == mass_before

    d2 = sigs[-1].to_dict()
    d2["tau_3"] = float("nan")
    est.learn_one(d2)
    assert est.n_missing == n_before + 2
    assert est.predict_one({}) == mass_before
