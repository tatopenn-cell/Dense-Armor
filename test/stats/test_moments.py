"""Tests for dense_armor/utility/stats/moments.py."""
import doctest

import numpy as np
import pytest
from scipy import stats as scipy_stats

import dense_armor.utility.stats.moments as moments_mod
from dense_armor.utility.stats.moments import (
    RunningMoments,
    RunningMomentsVector,
    EWStats,
)
from dense_armor.checks import check_estimator


def _stream(n: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    return list(rng.normal(0.0, 1.0, n))


def test_running_moments_match_numpy():
    xs = _stream(2000)
    m = RunningMoments()
    for v in xs:
        m.learn_one({"x": v})
    assert m.count == 2000
    assert m.mean == pytest.approx(np.mean(xs), abs=1e-12)
    assert m.var == pytest.approx(np.var(xs, ddof=1), abs=1e-12)
    assert m.std == pytest.approx(np.std(xs, ddof=1), abs=1e-12)
    assert m.min == pytest.approx(np.min(xs))
    assert m.max == pytest.approx(np.max(xs))


def test_skew_kurt_match_scipy():
    xs = list(np.random.default_rng(1).normal(0, 1, 5000))
    m = RunningMoments()
    for v in xs:
        m.learn_one({"x": v})
    assert m.skewness == pytest.approx(
        float(scipy_stats.skew(xs, bias=True)), abs=1e-10)
    assert m.kurtosis == pytest.approx(
        float(scipy_stats.kurtosis(xs, bias=True)), abs=1e-10)


def test_merge_matches_whole():
    xs = _stream(4000)
    whole = RunningMoments()
    for v in xs:
        whole.learn_one({"x": v})
    a, b = RunningMoments(), RunningMoments()
    for v in xs[:2000]:
        a.learn_one({"x": v})
    for v in xs[2000:]:
        b.learn_one({"x": v})
    merged = a.merge(b)
    assert merged.count == whole.count
    assert merged.mean == pytest.approx(whole.mean, abs=1e-12)
    assert merged.var == pytest.approx(whole.var, abs=1e-12)
    assert merged.skewness == pytest.approx(whole.skewness, abs=1e-10)
    assert merged.kurtosis == pytest.approx(whole.kurtosis, abs=1e-10)
    assert merged.min == whole.min
    assert merged.max == whole.max


def test_merge_empty_sides():
    a = RunningMoments()
    b = RunningMoments()
    for v in [1.0, 2.0, 3.0]:
        b.learn_one({"x": v})
    m = a.merge(b)
    assert m.count == 3
    assert m.mean == pytest.approx(2.0)
    m2 = b.merge(a)
    assert m2.count == 3
    assert m2.mean == pytest.approx(2.0)


def test_merge_type_error():
    with pytest.raises(TypeError):
        RunningMoments().merge("not a stat")


def test_nan_is_skipped_and_counted():
    m = RunningMoments()
    for v in [1.0, float("nan"), 3.0, float("nan"), 5.0]:
        m.learn_one({"x": v})
    assert m.count == 3
    assert m.n_missing == 2
    assert m.mean == pytest.approx(3.0)
    assert m.var == pytest.approx(4.0)


def test_feature_selection():
    m = RunningMoments(feature="b")
    for i in range(20):
        m.learn_one({"a": 1000.0 * i, "b": float(i)})
    assert m.mean == pytest.approx(9.5)


def test_vector_matches_scalar():
    xs = _stream(500)
    ys = _stream(500, seed=1)
    vec = RunningMomentsVector(features=["a", "b"])
    a, b = RunningMoments(), RunningMoments()
    for x, y in zip(xs, ys):
        vec.learn_one({"a": x, "b": y})
        a.learn_one({"a": x})
        b.learn_one({"b": y})
    summary = vec.transform_one({})
    assert summary["a"]["mean"] == pytest.approx(a.mean, abs=1e-12)
    assert summary["b"]["var"] == pytest.approx(b.var, abs=1e-12)


def test_vector_merge():
    xs = _stream(400)
    ys = _stream(400, seed=1)
    left = RunningMomentsVector()
    right = RunningMomentsVector()
    for i in range(200):
        left.learn_one({"a": xs[i], "b": ys[i]})
    for i in range(200, 400):
        right.learn_one({"a": xs[i], "b": ys[i]})
    merged = left.merge(right)
    whole = RunningMomentsVector()
    for i in range(400):
        whole.learn_one({"a": xs[i], "b": ys[i]})
    s_merged = merged.transform_one({})
    s_whole = whole.transform_one({})
    assert s_merged["a"]["mean"] == pytest.approx(s_whole["a"]["mean"], abs=1e-12)
    assert s_merged["b"]["var"] == pytest.approx(s_whole["b"]["var"], abs=1e-12)


def test_ew_mean_tracks_step():
    e = EWStats(alpha=0.05)
    for _ in range(500):
        e.learn_one({"x": 1.0})
    for _ in range(500):
        e.learn_one({"x": 5.0})
    assert e.mean == pytest.approx(5.0, abs=0.01)


def test_ew_var_shrinks_on_constant_stream():
    e = EWStats(alpha=0.1)
    for _ in range(100):
        e.learn_one({"x": 3.0})
    assert e.var == pytest.approx(0.0, abs=1e-12)


def test_ew_matches_hand_computation():
    e = EWStats(alpha=0.5)
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        e.learn_one({"x": v})
    assert e.mean == pytest.approx(4.0625)


def test_dt_accepted():
    m = RunningMoments()
    for i in range(10):
        m.learn_one({"x": float(i)}, t=i * 0.01)
    assert m.dt == pytest.approx(0.01)


def test_hand_case_running_moments():
    m = RunningMoments()
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        m.learn_one({"x": v})
    assert m.count == 5
    assert m.mean == pytest.approx(3.0)
    assert m.var == pytest.approx(2.5)
    assert m.ptp == pytest.approx(4.0)
    assert m.skewness == pytest.approx(0.0, abs=1e-12)
    assert m.kurtosis == pytest.approx(-1.3, abs=1e-12)


def test_check_estimator_running_moments():
    check_estimator(RunningMoments())


def test_check_estimator_running_moments_vector():
    check_estimator(RunningMomentsVector())


def test_check_estimator_ew_stats():
    check_estimator(EWStats())


def test_doctests():
    assert doctest.testmod(moments_mod).failed == 0
