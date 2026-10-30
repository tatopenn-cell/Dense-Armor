"""Float32 path of SelectKBest / _CorrStats: default bit for bit, float32 agrees.

Reference of the float32 comparison is the float64 variance of the same
float32 values, never the float64 values they came from.
"""

import numpy as np
import pytest

from dense_armor.checks import check_estimator
from dense_armor.utility.preprocessing.select import SelectKBest, _CorrStats


def test_default_matches_hand_welford_bivariate():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [2.0, 4.0, 6.0, 8.0, 10.0]
    n = 0
    mx = my = m2x = m2y = cxy = 0.0
    for x, y in zip(xs, ys):
        n += 1
        dx = x - mx
        dy = y - my
        mx += dx / n
        my += dy / n
        m2x += dx * (x - mx)
        m2y += dy * (y - my)
        cxy += dx * (y - my)
    st = _CorrStats()
    for x, y in zip(xs, ys):
        st.update(x, y)
    assert st.mean_x_ == mx
    assert st.mean_y_ == my
    assert st.m2_x_ == m2x
    assert st.m2_y_ == m2y
    assert st.c_xy_ == cxy
    assert st.n_ == n


def test_default_corr_perfect_line():
    st = _CorrStats()
    for i in range(20):
        st.update(float(i), 2.0 * i)
    assert abs(st.corr - 1.0) < 1e-12


def test_default_old_signature_positional():
    sk = SelectKBest(3, ["a", "b"])
    assert sk.k == 3
    assert sk.features == ["a", "b"]


def test_dtype_none_is_default_path():
    a = SelectKBest(dtype=None)
    b = SelectKBest()
    for i in range(20):
        a.learn_one({"x": float(i)}, y=float(i * 2))
        b.learn_one({"x": float(i)}, y=float(i * 2))
    assert a.stats_["x"].m2_x_ == b.stats_["x"].m2_x_
    assert a.stats_["x"].c_xy_ == b.stats_["x"].c_xy_


def test_dtype_float64_goes_through_default_path():
    a = SelectKBest(dtype=np.float64)
    b = SelectKBest()
    for i in range(20):
        a.learn_one({"x": float(i)}, y=float(i * 2))
        b.learn_one({"x": float(i)}, y=float(i * 2))
    assert a.stats_["x"].m2_x_ == b.stats_["x"].m2_x_


def test_float32_agrees_with_float64_on_the_decision_stream():
    rng = np.random.default_rng(0)
    xs = (1e3 + 0.01 * rng.standard_normal(100_000)).astype(np.float32)
    ys = (2.0 * (xs - 1e3)).astype(np.float32)
    sk = SelectKBest(k=1, dtype=np.float32)
    for x, y in zip(xs, ys):
        sk.learn_one({"a": float(x)}, y=float(y))
    st = sk.stats_["a"]
    ref_var = float(xs.astype(np.float64).var(ddof=0))
    assert abs(st.m2_x_ - ref_var) / ref_var <= 1e-6
    assert abs(st.corr - 1.0) < 1e-6
    assert st._s2x.dtype == np.float32


@pytest.mark.parametrize("big", [1e10, 1e38])
def test_float32_constant_stream_is_exactly_zero(big):
    sk = SelectKBest(k=1, dtype=np.float32)
    for _ in range(5):
        sk.learn_one({"a": float(big)}, y=float(big))
    st = sk.stats_["a"]
    assert st.m2_x_ == 0.0
    assert st.m2_y_ == 0.0
    assert st.corr == 0.0


def test_float32_transform_keeps_top_feature():
    sk = SelectKBest(k=1, dtype=np.float32)
    for i in range(50):
        sk.learn_one({"a": float(i), "b": 1.0}, y=float(i))
    out = sk.transform_one({"a": 1.0, "b": 1.0})
    assert "a" in out
    assert "b" not in out


def test_float32_no_learning_before_y_none():
    sk = SelectKBest(dtype=np.float32)
    sk.learn_one({"a": 1.0}, y=None)
    assert sk.stats_ is None


def test_check_estimator_default_dtype():
    out = check_estimator(SelectKBest())
    assert len(out["not_applicable"]) == 0
    assert SelectKBest()._unit_test_skips() == set()


def test_check_estimator_float32_dtype():
    out = check_estimator(SelectKBest(dtype=np.float32))
    assert len(out["not_applicable"]) == 0
    assert SelectKBest(dtype=np.float32)._unit_test_skips() == set()
