"""Float32 path of OnlineGaussianNB: default bit for bit, float32 agrees.

Reference of the float32 comparison is the float64 mean and variance of
the same float32 values, never the float64 values they came from.
"""

import numpy as np
import pytest

from dense_armor.checks import check_estimator
from dense_armor.utility.learn.online_classifiers import OnlineGaussianNB


def test_default_matches_hand_welford_bit_for_bit():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    mean = 0.0
    m2 = 0.0
    n = 0.0
    for v in xs:
        n_new = n + 1.0
        delta = v - mean
        mean = mean + delta / n_new
        m2 = m2 + delta * (v - mean)
        n = n_new
    nb = OnlineGaussianNB()
    for v in xs:
        nb.learn_one({"a": v}, 0)
    assert nb._mu[0][0] == mean
    assert nb._m2[0][0] == m2
    assert nb._n[0] == n


def test_default_old_signature_positional():
    nb = OnlineGaussianNB(0.1, 1e-9, 1e-12)
    assert nb.alpha == 0.1
    assert nb.var_smoothing == 1e-9
    assert nb.eps == 1e-12


def test_dtype_none_is_default_path():
    a = OnlineGaussianNB(dtype=None)
    b = OnlineGaussianNB()
    for v in (1.0, 2.0, 3.0):
        a.learn_one({"a": v}, 0)
        b.learn_one({"a": v}, 0)
    assert a._mu[0][0] == b._mu[0][0]
    assert a._m2[0][0] == b._m2[0][0]


def test_dtype_float64_goes_through_default_path():
    a = OnlineGaussianNB(dtype=np.float64)
    b = OnlineGaussianNB()
    for v in (1.0, 2.0, 3.0):
        a.learn_one({"a": v}, 0)
        b.learn_one({"a": v}, 0)
    assert a._mu[0][0] == b._mu[0][0]
    assert a._m2[0][0] == b._m2[0][0]


def test_float32_agrees_with_float64_on_the_decision_stream():
    rng = np.random.default_rng(0)
    xs = (1e3 + 0.01 * rng.standard_normal(100_000)).astype(np.float32)
    nb = OnlineGaussianNB(dtype=np.float32)
    for v in xs:
        nb.learn_one({"a": float(v)}, 0)
    ref_var = float(xs.astype(np.float64).var(ddof=0))
    ref_mean = float(xs.astype(np.float64).mean())
    var32 = nb._m2[0][0] / nb._n[0]
    assert abs(nb._mu[0][0] - ref_mean) / abs(ref_mean) <= 1e-6
    assert abs(var32 - ref_var) / ref_var <= 1e-6
    assert nb._s1[0].dtype == np.float32
    assert nb._s2[0].dtype == np.float32


@pytest.mark.parametrize("big", [1e10, 1e38])
def test_float32_constant_stream_is_exactly_zero(big):
    nb = OnlineGaussianNB(dtype=np.float32)
    for _ in range(5):
        nb.learn_one({"a": float(big)}, 0)
    assert nb._m2[0][0] == 0.0


def test_float32_alpha_positive_raises():
    nb = OnlineGaussianNB(alpha=0.1, dtype=np.float32)
    with pytest.raises(ValueError):
        nb.learn_one({"a": 1.0}, 0)


def test_float32_predicts_two_classes_correctly():
    nb = OnlineGaussianNB(dtype=np.float32)
    for _ in range(200):
        nb.learn_one({"a": 0.0}, 0)
        nb.learn_one({"a": 1.0}, 1)
    assert nb.predict_one({"a": 0.05}) == 0
    assert nb.predict_one({"a": 0.95}) == 1


def test_float32_predict_before_learning_returns_none():
    nb = OnlineGaussianNB(dtype=np.float32)
    assert nb.predict_proba_one({"a": 0.0}) == {}


def test_float32_multiclass_three():
    nb = OnlineGaussianNB(dtype=np.float32)
    for _ in range(200):
        nb.learn_one({"a": 0.0}, 0)
        nb.learn_one({"a": 1.0}, 1)
        nb.learn_one({"a": 2.0}, 2)
    assert nb.predict_one({"a": 0.05}) == 0
    assert nb.predict_one({"a": 1.05}) == 1
    assert nb.predict_one({"a": 1.95}) == 2


def test_check_estimator_default_dtype():
    out = check_estimator(OnlineGaussianNB())
    assert len(out["not_applicable"]) == 0
    assert OnlineGaussianNB()._unit_test_skips() == set()


def test_check_estimator_float32_dtype():
    out = check_estimator(OnlineGaussianNB(dtype=np.float32))
    assert len(out["not_applicable"]) == 0
    assert OnlineGaussianNB(dtype=np.float32)._unit_test_skips() == set()
