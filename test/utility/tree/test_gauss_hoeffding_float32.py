"""Float32 path of _Gauss and HoeffdingTreeClassifier.

Reference of the float32 comparison is the float64 result of the same
class on the same float32 values.
"""

import numpy as np
import pytest

from dense_armor.checks import check_estimator
from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier, _Gauss


def _stream(n=100_000, seed=0):
    rng = np.random.default_rng(seed)
    return (1e3 + 0.01 * rng.standard_normal(n)).astype(np.float32)


def test_gauss_default_matches_hand_welford_bit_for_bit():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    mean = 0.0
    m2 = 0.0
    n = 0
    for v in xs:
        n += 1
        d = v - mean
        mean += d / n
        m2 += d * (v - mean)
    g = _Gauss()
    for v in xs:
        g.update(v)
    assert g.n_ == n
    assert g.mean_ == mean
    assert g.m2_ == m2


def test_gauss_default_dtype_none_is_default_path():
    a = _Gauss(dtype=None)
    b = _Gauss()
    for v in [1.0, 2.0, 3.0]:
        a.update(v)
        b.update(v)
    assert a.var == b.var
    assert a.mean_ == b.mean_


def test_gauss_float32_agrees_with_float64_on_the_decision_stream():
    xs = _stream()
    ref = float(xs.astype(np.float64).var(ddof=0))
    g = _Gauss(dtype=np.float32)
    for v in xs:
        g.update(float(v))
    assert abs(g.var - ref) / ref <= 1e-6
    assert g._m2_y.dtype == np.float32


@pytest.mark.parametrize("big", [1e10, 1e38])
def test_gauss_float32_constant_stream_is_exactly_zero(big):
    g = _Gauss(dtype=np.float32)
    for _ in range(5):
        g.update(float(big))
    assert g.var == 0.0


def test_gauss_float32_mean_is_finite_and_close():
    xs = _stream(n=1000)
    ref = float(xs.astype(np.float64).mean())
    g = _Gauss(dtype=np.float32)
    for v in xs:
        g.update(float(v))
    assert abs(g.mean_ - ref) / abs(ref) <= 1e-6


def test_hoeffding_default_bit_for_bit():
    rng = np.random.default_rng(0)
    X = (1e3 + 0.01 * rng.standard_normal((2000, 2))).astype(np.float32)
    yb = (X[:, 0] > 1e3).astype(int)
    h1 = HoeffdingTreeClassifier(grace_period=50)
    h2 = HoeffdingTreeClassifier(grace_period=50)
    for i in range(2000):
        h1.learn_one({"f0": float(X[i, 0]), "f1": float(X[i, 1])}, int(yb[i]))
        h2.learn_one({"f0": float(X[i, 0]), "f1": float(X[i, 1])}, int(yb[i]))
    g1 = next(iter(next(iter(h1.root_.feature_stats_.values())).values()))
    g2 = next(iter(next(iter(h2.root_.feature_stats_.values())).values()))
    assert g1.var == g2.var
    assert g1.mean_ == g2.mean_


def test_hoeffding_float32_agrees_with_float64():
    rng = np.random.default_rng(0)
    X = (1e3 + 0.01 * rng.standard_normal((2000, 2))).astype(np.float32)
    yb = (X[:, 0] > 1e3).astype(int)
    h64 = HoeffdingTreeClassifier(grace_period=50)
    h32 = HoeffdingTreeClassifier(grace_period=50, dtype=np.float32)
    for i in range(2000):
        h64.learn_one({"f0": float(X[i, 0]), "f1": float(X[i, 1])}, int(yb[i]))
        h32.learn_one({"f0": float(X[i, 0]), "f1": float(X[i, 1])}, int(yb[i]))
    g64 = next(iter(next(iter(h64.root_.feature_stats_.values())).values()))
    g32 = next(iter(next(iter(h32.root_.feature_stats_.values())).values()))
    assert abs(g32.var - g64.var) / g64.var <= 1e-6


def test_hoeffding_float32_predicts_a_simple_split():
    rng = np.random.default_rng(0)
    h = HoeffdingTreeClassifier(grace_period=20, dtype=np.float32)
    for _ in range(400):
        x = float(rng.normal(0.0 if rng.random() < 0.5 else 4.0, 0.5))
        h.learn_one({"x": x}, int(x > 2.0))
    p0 = h.predict_one({"x": 0.1})
    p4 = h.predict_one({"x": 4.0})
    assert p0 is not None and p4 is not None
    assert p0 == 0 and p4 == 1


def test_hoeffding_float32_max_nodes_one_is_a_leaf():
    rng = np.random.default_rng(0)
    X = (1e3 + 0.01 * rng.standard_normal((500, 2))).astype(np.float32)
    h = HoeffdingTreeClassifier(grace_period=50, max_nodes=1, dtype=np.float32)
    for i in range(500):
        h.learn_one({"f0": float(X[i, 0]), "f1": float(X[i, 1])}, int(X[i, 0] > 1e3))
    assert h.root_.is_leaf


def test_check_estimator_default_dtype():
    out = check_estimator(HoeffdingTreeClassifier())
    assert len(out["not_applicable"]) == 0
    assert HoeffdingTreeClassifier()._unit_test_skips() == set()


def test_check_estimator_float32_dtype():
    out = check_estimator(HoeffdingTreeClassifier(dtype=np.float32))
    assert len(out["not_applicable"]) == 0
    assert HoeffdingTreeClassifier(dtype=np.float32)._unit_test_skips() == set()
