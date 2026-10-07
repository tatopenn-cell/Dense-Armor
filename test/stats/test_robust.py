"""Tests for dense_armor/utility/stats/robust.py."""
import doctest

import numpy as np
import pytest

import dense_armor.utility.stats.robust as robust_mod
from dense_armor.utility.stats.robust import (
    MAD_SCALE,
    RollingMedian,
    RollingMAD,
    RollingIQR,
    RollingQuantile,
    RollingMedianVector,
)
from dense_armor.checks import check_estimator


def _stream(n: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed)
    return list(rng.normal(0.0, 1.0, n))


def test_rolling_median_matches_numpy_at_end():
    xs = _stream(500)
    m = RollingMedian(window=20)
    for v in xs:
        m.learn_one({"x": v})
    assert m.value == pytest.approx(np.median(xs[-20:]))


def test_rolling_median_window_is_fifo():
    m = RollingMedian(window=3)
    for v in [10.0, 20.0, 30.0, 1.0]:
        m.learn_one({"x": v})
    assert m.value == pytest.approx(20.0)


def test_rolling_median_odd_even():
    m = RollingMedian(window=4)
    for v in [1.0, 2.0, 3.0, 4.0]:
        m.learn_one({"x": v})
    assert m.value == pytest.approx(2.5)


def test_rolling_mad_matches_hand_value():
    m = RollingMAD(window=5)
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        m.learn_one({"x": v})
    assert m.value == pytest.approx(MAD_SCALE * 1.0)


def test_rolling_mad_scale_configurable():
    m = RollingMAD(window=5, scale=1.0)
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        m.learn_one({"x": v})
    assert m.value == pytest.approx(1.0)


def test_rolling_iqr_matches_numpy():
    xs = _stream(300)
    m = RollingIQR(window=50)
    for v in xs:
        m.learn_one({"x": v})
    q1, q3 = np.percentile(xs[-50:], [25, 75])
    assert m.value == pytest.approx(q3 - q1)


def test_rolling_quantile_matches_numpy():
    xs = _stream(300)
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        m = RollingQuantile(q=q, window=40)
        for v in xs:
            m.learn_one({"x": v})
        assert m.value == pytest.approx(
            float(np.percentile(xs[-40:], q * 100))
        )


def test_rolling_quantile_q05_equals_median():
    xs = _stream(200)
    q = RollingQuantile(q=0.5, window=30)
    med = RollingMedian(window=30)
    for v in xs:
        q.learn_one({"x": v})
        med.learn_one({"x": v})
    assert q.value == pytest.approx(med.value)


def test_nan_skipped_and_counted():
    m = RollingMedian(window=5)
    for v in [1.0, float("nan"), 3.0, float("nan"), 5.0]:
        m.learn_one({"x": v})
    assert m.count == 3
    assert m.n_missing == 2
    assert m.value == pytest.approx(3.0)


def test_empty_window_returns_zero():
    m = RollingMedian(window=5)
    assert m.value == 0.0


def test_feature_selection():
    m = RollingMedian(window=5, feature="b")
    for i in range(20):
        m.learn_one({"a": 1000.0 * i, "b": float(i)})
    assert m.value == pytest.approx(17.0)


def test_window_s_uses_dt():
    m = RollingMedian(window=100, window_s=0.05)
    for i in range(100):
        m.learn_one({"x": float(i)}, t=i * 0.01)
    assert m.window_size == 5


def test_window_s_falls_back_before_dt_known():
    m = RollingMedian(window=7, window_s=0.1)
    for v in [1.0, 2.0, 3.0]:
        m.learn_one({"x": v})
    assert m.window_size == 7


def test_vector_matches_scalar():
    xs = _stream(200)
    ys = _stream(200, seed=1)
    vec = RollingMedianVector(window=10, features=["a", "b"])
    a = RollingMedian(window=10, feature="a")
    b = RollingMedian(window=10, feature="b")
    for x, y in zip(xs, ys):
        vec.learn_one({"a": x, "b": y})
        a.learn_one({"a": x})
        b.learn_one({"b": y})
    s = vec.transform_one({})
    assert s["a"]["value"] == pytest.approx(a.value)
    assert s["b"]["value"] == pytest.approx(b.value)


def test_dt_accepted():
    m = RollingMedian(window=10)
    for i in range(20):
        m.learn_one({"x": float(i)}, t=i * 0.01)
    assert m.dt == pytest.approx(0.01)


def test_check_estimator_rolling_median():
    check_estimator(RollingMedian())


def test_check_estimator_rolling_mad():
    check_estimator(RollingMAD())


def test_check_estimator_rolling_iqr():
    check_estimator(RollingIQR())


def test_check_estimator_rolling_quantile():
    check_estimator(RollingQuantile())


def test_check_estimator_rolling_median_vector():
    check_estimator(RollingMedianVector())


def test_doctests():
    assert doctest.testmod(robust_mod).failed == 0
