"""Tests for dense_armor/utility/stats/dependence.py."""
import doctest
import math

import numpy as np
import pytest

import dense_armor.utility.stats.dependence as dep_mod
from dense_armor.utility.stats.dependence import (
    RunningCovariance,
    RunningCorrelation,
    RollingCovariance,
    RollingCorrelation,
    Autocorrelation,
)
from dense_armor.checks import check_estimator


def _pair(n: int = 2000, seed: int = 0, rho: float = 0.7):
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 1.0, n)
    b = rho * a + math.sqrt(1.0 - rho * rho) * rng.normal(0.0, 1.0, n)
    return list(a), list(b)


def test_running_covariance_matches_numpy():
    a, b = _pair(2000)
    c = RunningCovariance("a", "b")
    for x, y in zip(a, b):
        c.learn_one({"a": float(x), "b": float(y)})
    assert c.count == 2000
    assert c.cov == pytest.approx(float(np.cov(a, b, ddof=1)[0, 1]), abs=1e-12)


def test_running_correlation_matches_numpy():
    a, b = _pair(2000)
    c = RunningCovariance("a", "b")
    for x, y in zip(a, b):
        c.learn_one({"a": float(x), "b": float(y)})
    assert c.corr == pytest.approx(
        float(np.corrcoef(a, b)[0, 1]), abs=1e-12
    )


def test_running_covariance_perfect_linear():
    c = RunningCovariance("a", "b")
    for a, b in [(1.0, 3.0), (2.0, 5.0), (3.0, 7.0)]:
        c.learn_one({"a": a, "b": b})
    assert c.corr == pytest.approx(1.0, abs=1e-12)
    assert c.cov == pytest.approx(2.0, abs=1e-12)


def test_running_covariance_merge_matches_whole():
    a, b = _pair(4000)
    whole = RunningCovariance("a", "b")
    for x, y in zip(a, b):
        whole.learn_one({"a": float(x), "b": float(y)})
    left = RunningCovariance("a", "b")
    right = RunningCovariance("a", "b")
    for x, y in zip(a[:2000], b[:2000]):
        left.learn_one({"a": float(x), "b": float(y)})
    for x, y in zip(a[2000:], b[2000:]):
        right.learn_one({"a": float(x), "b": float(y)})
    merged = left.merge(right)
    assert merged.count == whole.count
    assert merged.cov == pytest.approx(whole.cov, abs=1e-12)
    assert merged.corr == pytest.approx(whole.corr, abs=1e-12)


def test_running_covariance_merge_type_error():
    with pytest.raises(TypeError):
        RunningCovariance().merge("x")


def test_running_covariance_merge_channel_mismatch():
    with pytest.raises(ValueError):
        RunningCovariance("a", "b").merge(RunningCovariance("a", "c"))


def test_running_covariance_nan_skipped():
    c = RunningCovariance("a", "b")
    for a, b in [(1.0, 1.0), (float("nan"), 2.0), (3.0, 3.0)]:
        c.learn_one({"a": a, "b": b})
    assert c.count == 2
    assert c.n_missing == 1


def test_running_covariance_empty():
    c = RunningCovariance("a", "b")
    assert c.cov == 0.0
    assert c.corr == 0.0


def test_running_correlation_class():
    a, b = _pair(1000, rho=-0.5)
    r = RunningCorrelation("a", "b")
    for x, y in zip(a, b):
        r.learn_one({"a": float(x), "b": float(y)})
    assert r.corr == pytest.approx(
        float(np.corrcoef(a, b)[0, 1]), abs=1e-12
    )


def test_rolling_covariance_matches_numpy():
    a, b = _pair(500)
    c = RollingCovariance("a", "b", window=50)
    for x, y in zip(a, b):
        c.learn_one({"a": float(x), "b": float(y)})
    assert c.cov == pytest.approx(
        float(np.cov(a[-50:], b[-50:], ddof=1)[0, 1]), abs=1e-12
    )
    assert c.corr == pytest.approx(
        float(np.corrcoef(a[-50:], b[-50:])[0, 1]), abs=1e-12
    )


def test_rolling_covariance_fifo():
    c = RollingCovariance("a", "b", window=3)
    for a, b in [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]:
        c.learn_one({"a": a, "b": b})
    assert c.count == 4
    assert len(c.buf_x_) == 3


def test_rolling_correlation_class():
    a, b = _pair(500, seed=1, rho=-0.9)
    r = RollingCorrelation("a", "b", window=40)
    for x, y in zip(a, b):
        r.learn_one({"a": float(x), "b": float(y)})
    assert r.corr == pytest.approx(
        float(np.corrcoef(a[-40:], b[-40:])[0, 1]), abs=1e-12
    )


def test_autocorrelation_white_noise():
    rng = np.random.default_rng(2)
    ac = Autocorrelation("x", lag=1)
    for v in rng.normal(0.0, 1.0, 5000):
        ac.learn_one({"x": float(v)})
    assert abs(ac.value) < 0.1


def test_autocorrelation_sine_at_period():
    n = 2000
    period = 20
    xs = [np.sin(2.0 * np.pi * i / period) for i in range(n)]
    ac = Autocorrelation("x", lag=period)
    for v in xs:
        ac.learn_one({"x": float(v)})
    assert ac.value > 0.9


def test_autocorrelation_anti_phase():
    n = 2000
    period = 20
    xs = [np.sin(2.0 * np.pi * i / period) for i in range(n)]
    ac = Autocorrelation("x", lag=period // 2)
    for v in xs:
        ac.learn_one({"x": float(v)})
    assert ac.value < -0.9


def test_autocorrelation_nan_skipped():
    ac = Autocorrelation("x", lag=1)
    for v in [1.0, float("nan"), 2.0, 3.0]:
        ac.learn_one({"x": v})
    assert ac.count == 3
    assert ac.n_missing == 1


def test_autocorrelation_invalid_lag():
    with pytest.raises(ValueError):
        Autocorrelation("x", lag=0)


def test_check_estimator_running_covariance():
    check_estimator(RunningCovariance("a", "b"))


def test_check_estimator_running_correlation():
    check_estimator(RunningCorrelation("a", "b"))


def test_check_estimator_rolling_covariance():
    check_estimator(RollingCovariance("a", "b"))


def test_check_estimator_rolling_correlation():
    check_estimator(RollingCorrelation("a", "b"))


def test_check_estimator_autocorrelation():
    check_estimator(Autocorrelation("a", lag=1))


def test_doctests():
    assert doctest.testmod(dep_mod).failed == 0
