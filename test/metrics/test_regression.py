"""Tests for dense_armor/utility/metrics/regression.py."""
import doctest
import math

import pytest
from sklearn import metrics as skm

import dense_armor.utility.metrics.regression as reg_mod
from dense_armor.roles import Classifier, Estimate, Regressor
from dense_armor.utility.metrics import (
    GaussianNLL,
    IntervalCoverage,
    MeanAbsoluteError,
    MeanIntervalWidth,
    MeanSquaredError,
    R2Score,
    RootMeanSquaredError,
)


def _stream(n: int = 500, seed: int = 0):
    rng = __import__("numpy").random.default_rng(seed)
    yt = rng.normal(0.0, 1.0, n).tolist()
    yp = [y + rng.normal(0.0, 0.2) for y in yt]
    return yt, yp


def test_mae_matches_sklearn():
    yt, yp = _stream()
    m = MeanAbsoluteError()
    for a, b in zip(yt, yp):
        m.update(a, b)
    assert m.get() == pytest.approx(skm.mean_absolute_error(yt, yp))


def test_mse_matches_sklearn():
    yt, yp = _stream()
    m = MeanSquaredError()
    for a, b in zip(yt, yp):
        m.update(a, b)
    assert m.get() == pytest.approx(skm.mean_squared_error(yt, yp))


def test_rmse_matches_sklearn():
    yt, yp = _stream()
    m = RootMeanSquaredError()
    for a, b in zip(yt, yp):
        m.update(a, b)
    assert m.get() == pytest.approx(math.sqrt(skm.mean_squared_error(yt, yp)))


def test_r2_matches_sklearn():
    yt, yp = _stream()
    m = R2Score()
    for a, b in zip(yt, yp):
        m.update(a, b)
    assert m.get() == pytest.approx(skm.r2_score(yt, yp))


def test_r2_perfect():
    m = R2Score()
    for a, b in [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]:
        m.update(a, b)
    assert m.get() == pytest.approx(1.0)


def test_revert_restores_mae():
    m = MeanAbsoluteError()
    for a, b in [(1.0, 1.5), (2.0, 2.5), (3.0, 2.5)]:
        m.update(a, b)
    before = m.get()
    m.update(0.0, 10.0)
    m.revert(0.0, 10.0)
    assert m.get() == pytest.approx(before)


def test_merge_matches_whole_mae():
    yt, yp = _stream(600)
    whole = MeanAbsoluteError()
    for a, b in zip(yt, yp):
        whole.update(a, b)
    left, right = MeanAbsoluteError(), MeanAbsoluteError()
    for a, b in zip(yt[:300], yp[:300]):
        left.update(a, b)
    for a, b in zip(yt[300:], yp[300:]):
        right.update(a, b)
    merged = left.merge(right)
    assert merged.get() == pytest.approx(whole.get())


def test_merge_matches_whole_r2():
    yt, yp = _stream(600)
    whole = R2Score()
    for a, b in zip(yt, yp):
        whole.update(a, b)
    a1, a2, a3 = R2Score(), R2Score(), R2Score()
    for i, (a, b) in enumerate(zip(yt, yp)):
        [a1, a2, a3][i % 3].update(a, b)
    merged = a1.merge(a2).merge(a3)
    assert merged.get() == pytest.approx(whole.get())


def test_interval_coverage_simple():
    m = IntervalCoverage()
    for y, lo, hi in [(0.5, 0.0, 1.0), (2.0, 0.0, 1.0), (0.9, 0.0, 1.0)]:
        m.update(y, (lo, hi))
    assert m.get() == pytest.approx(2.0 / 3.0)


def test_mean_interval_width_simple():
    m = MeanIntervalWidth()
    for lo, hi in [(0.0, 1.0), (0.0, 3.0)]:
        m.update(0.0, (lo, hi))
    assert m.get() == pytest.approx(2.0)


def test_gaussian_nll_matches_formula():
    m = GaussianNLL()
    y, mu, v = 1.0, 0.5, 4.0
    m.update(y, (mu, v))
    expected = 0.5 * math.log(2.0 * math.pi * v) + 0.5 * (y - mu) ** 2 / v
    assert m.get() == pytest.approx(expected)


def test_gaussian_nll_accepts_estimate():
    m = GaussianNLL()
    e = Estimate(mean=0.5, var=4.0)
    m.update(1.0, e)
    m2 = GaussianNLL()
    m2.update(1.0, (0.5, 4.0))
    assert m.get() == pytest.approx(m2.get())


def test_gaussian_nll_zero_variance_floored():
    m = GaussianNLL(eps=1e-6)
    m.update(0.0, (0.0, 0.0))
    assert math.isfinite(m.get())


def test_feature_selection_on_vector_pair():
    m = MeanAbsoluteError(feature="joint1")
    m.update({"joint0": 0.0, "joint1": 1.0},
             {"joint0": 100.0, "joint1": 1.5})
    assert m.get() == pytest.approx(0.5)


def test_missing_pairs_are_skipped():
    m = MeanSquaredError()
    m.update(1.0, 1.0)
    m.update(float("nan"), 1.0)
    m.update(1.0, float("nan"))
    m.update(None, 1.0)
    m.update(1.0, None)
    assert m.n == 1
    assert m.n_missing == 4


def test_works_with_role():
    assert MeanSquaredError().works_with(Regressor())
    assert not MeanSquaredError().works_with(Classifier())


def test_merge_type_error():
    with pytest.raises(TypeError):
        MeanAbsoluteError().merge("x")
    with pytest.raises(TypeError):
        MeanAbsoluteError().merge(MeanSquaredError())
    with pytest.raises(TypeError):
        R2Score().merge(MeanSquaredError())


def test_revert_after_update_returns_previous():
    for cls in (MeanAbsoluteError, MeanSquaredError,
                RootMeanSquaredError):
        m = cls()
        for a, b in [(1.0, 1.5), (2.0, 2.5), (3.0, 3.5)]:
            m.update(a, b)
        before = m.get()
        m.update(2.0, 5.0)
        m.revert(2.0, 5.0)
        assert m.get() == pytest.approx(before), cls.__name__


def test_r2_does_not_support_revert():
    m = R2Score()
    m.update(1.0, 1.0)
    with pytest.raises(NotImplementedError):
        m.revert(1.0, 1.0)


def test_doctests():
    assert doctest.testmod(reg_mod).failed == 0
