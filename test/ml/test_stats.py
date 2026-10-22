"""Hand cases, edge cases and the saved difference of the statistics."""

import doctest

import numpy as np

import dense_armor.ml.stats as S
from dense_armor.ml.stats import (
    Autocorrelation,
    ColumnMoments,
    Correlation,
    Covariance,
    EWStats,
    Moments,
    RollingCorrelation,
    RollingCovariance,
    RollingMedian,
    ShrinkageCovariance,
)


def test_doctests():
    r = doctest.testmod(S)
    assert r.failed == 0 and r.attempted > 10


def test_covariance_hand_case():
    m = Covariance().fit([1.0, 2.0, 3.0, float("nan")], [2.0, 4.0, 7.0, 1.0])
    assert m.n_missing_ == 1 and m.count_ == 3
    assert abs(m.cov_ - 2.5) < 1e-12
    assert abs(m.corr_ - 2.5 / np.sqrt(57 / 9)) < 1e-12
    one = Covariance().fit([1.0], [2.0])
    assert np.isnan(one.cov_) and np.isnan(one.corr_)
    flat = Covariance().fit([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
    assert np.isnan(flat.corr_) and flat.cov_ == 0.0
    assert Correlation().fit([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]).corr_ == -1.0


def test_rolling_hand_case():
    m = RollingCovariance(window=3).fit([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 5.0])
    assert m.cov_series_.shape == (2,)
    assert abs(m.cov_series_[0] - 1.0) < 1e-12
    assert abs(m.cov_ - 1.5) < 1e-12
    short = RollingCorrelation(window=5).fit([1.0, 2.0], [1.0, 2.0])
    assert np.isnan(short.cov_) and np.isnan(short.corr_)
    flat = RollingCorrelation(window=3).fit([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
    assert np.isnan(flat.corr_)
    ok = RollingCorrelation(window=3).fit([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
    assert abs(ok.corr_ - 1.0) < 1e-12


def test_autocorrelation_hand_case():
    a = Autocorrelation(lag=1).fit([1.0, 2.0, 1.0, 2.0, 1.0])
    assert abs(a.value_ - (-0.8)) < 1e-12 and a.n_pairs_ == 4
    assert np.isnan(Autocorrelation(lag=3).fit([1.0, 2.0]).value_)
    assert np.isnan(Autocorrelation(lag=1).fit([2.0, 2.0, 2.0]).value_)


def test_moments_hand_case():
    m = Moments().fit([1.0, 2.0, 3.0, 4.0, float("inf")])
    assert m.n_missing_ == 1 and m.count_ == 4
    assert m.mean_ == 2.5 and abs(m.var_ - 5.0 / 3.0) < 1e-12 and m.ptp_ == 3.0
    assert abs(m.skewness_) < 1e-12
    assert abs(m.kurtosis_ - (-1.36)) < 1e-12
    e = Moments().fit([])
    assert e.count_ == 0 and np.isnan(e.mean_) and np.isnan(e.ptp_)
    one = Moments().fit([5.0])
    assert np.isnan(one.var_) and np.isnan(one.skewness_)
    v = ColumnMoments().fit([[1.0, 10.0], [3.0, 20.0]])
    assert v.mean_.tolist() == [2.0, 15.0] and v.var_.tolist() == [2.0, 50.0]
    assert ColumnMoments().fit([1.0, 3.0]).mean_.tolist() == [2.0]


def test_ew_hand_case():
    e = EWStats(alpha=0.5).fit([1.0, 3.0, 3.0])
    assert abs(e.mean_ - 2.5) < 1e-12 and abs(e.var_ - 0.75) < 1e-12
    n = EWStats().fit([])
    assert n.count_ == 0 and np.isnan(n.mean_)


def test_rolling_median_hand_case():
    m = RollingMedian(window=3).fit([[1.0, 9.0], [5.0, 1.0], [2.0, 4.0], [8.0, 6.0]])
    assert m.series_.tolist() == [[2.0, 4.0], [5.0, 4.0]]
    assert m.median_.tolist() == [5.0, 4.0]
    short = RollingMedian(window=9).fit([1.0, 2.0])
    assert short.series_.shape == (0, 1) and np.isnan(short.median_[0])


def test_shrinkage_hand_case():
    X = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [2.0, 2.0]])
    c = ShrinkageCovariance(2).fit(X)
    s = np.cov(X.T)
    tr, tr2 = np.trace(s), np.sum(s * s)
    alpha = 1 - (tr * tr / 5) / (tr2 - tr * tr / 2)
    assert abs(c.alpha_ - alpha) < 1e-12
    assert np.allclose(c.shrunk_, alpha * s + (1 - alpha) * tr * np.eye(2) / 2)
    flat = ShrinkageCovariance(2).fit(np.zeros((3, 2)))
    assert flat.alpha_ == 1.0 and flat.beta_ == 0.0
    one = ShrinkageCovariance(2).fit(np.zeros((1, 2)))
    assert one.shrunk_.shape == (2, 2)


def test_sketches_are_declared_stopped():
    assert set(S.STOPPED) == {
        "DDSketch",
        "TDigest",
        "CountMinSketch",
        "HyperLogLog",
        "BloomFilter",
        "SpaceSaving",
    }


def test_base_is_not_online():
    import pytest

    from dense_armor.ml.base import MLBase

    with pytest.raises(NotImplementedError):
        MLBase().fit([1.0])
    assert not hasattr(Moments, "learn_one")
    assert Moments(feature="a").get_params() == {"feature": "a"}
