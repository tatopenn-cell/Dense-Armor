import pytest

from dense_armor.checks import check_estimator
from dense_armor.utility.preprocessing.joints import (
    JointDerivatives,
    JointPower,
)


def test_derivatives_uniform_cubic():
    jd = JointDerivatives(order=3)
    for i in range(4):
        jd.learn_one({"q": [float(i) ** 3]}, t=float(i))
    out = jd.transform_one({})
    assert out["qd"][0] == pytest.approx(27.0, rel=1e-9)
    assert out["qdd"][0] == pytest.approx(18.0, rel=1e-9)
    assert out["jerk"][0] == pytest.approx(6.0, rel=1e-9)


def test_derivatives_nonuniform_cubic():
    ts = [0.0, 0.3, 1.1, 2.0]
    jd = JointDerivatives(order=3)
    for t in ts:
        jd.learn_one({"q": [t ** 3]}, t=t)
    out = jd.transform_one({})
    assert out["qd"][0] == pytest.approx(3 * ts[-1] ** 2, rel=1e-9)
    assert out["qdd"][0] == pytest.approx(6 * ts[-1], rel=1e-9)
    assert out["jerk"][0] == pytest.approx(6.0, rel=1e-9)


def test_derivatives_pipeline_uniform():
    jd = JointDerivatives(order=3)
    last = None
    for t in [0.0, 1.0, 2.0, 3.0]:
        last = jd.transform_one({"q": [t ** 3]}, t=t)
        _ = jd.learn_one({"q": [t ** 3]}, t=t)
    assert last is not None
    assert last["qd"][0] == pytest.approx(27.0, rel=1e-9)
    assert last["qdd"][0] == pytest.approx(18.0, rel=1e-9)


def test_derivatives_pipeline_nonuniform():
    ts = [0.0, 0.3, 1.1, 2.0]
    jd = JointDerivatives(order=3)
    last = None
    for t in ts:
        last = jd.transform_one({"q": [t ** 3]}, t=t)
        _ = jd.learn_one({"q": [t ** 3]}, t=t)
    assert last is not None
    assert last["qd"][0] == pytest.approx(3 * ts[-1] ** 2, rel=1e-9)
    assert last["qdd"][0] == pytest.approx(6 * ts[-1], rel=1e-9)


def test_derivatives_skip_nan():
    jd = JointDerivatives(order=2)
    jd.learn_one({"q": [0.0]}, t=0.0)
    jd.learn_one({"q": [float("nan")]}, t=0.5)
    jd.learn_one({"q": [4.0]}, t=2.0)
    assert jd.n_missing_ == 1
    out = jd.transform_one({})
    assert out["qd"][0] == pytest.approx(2.0, rel=1e-9)


def test_power():
    jp = JointPower()
    out = jp.transform_one({"tau": [1.0, 2.0], "qd": [3.0, 4.0]})
    assert out["power"] == [3.0, 8.0]
    assert out["power_total"] == 11.0


def test_check_estimator_derivatives():
    check_estimator(JointDerivatives())


def test_check_estimator_power():
    check_estimator(JointPower())
