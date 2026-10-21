"""Hand cases, edge cases and the saved difference of the classic statistics."""

import doctest
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

import dense_armor.offline.stats as S
from dense_armor.offline.stats import (
    ClassicAutocorrelation,
    ClassicBloomFilter,
    ClassicCountMinSketch,
    ClassicDDSketch,
    ClassicEWStats,
    ClassicHyperLogLog,
    ClassicOnlineShrinkageCovariance,
    ClassicRollingCorrelation,
    ClassicRollingCovariance,
    ClassicRollingMedianVector,
    ClassicRunningCorrelation,
    ClassicRunningCovariance,
    ClassicRunningMoments,
    ClassicRunningMomentsVector,
    ClassicSpaceSaving,
    ClassicTDigest,
)

BENCH = Path(__file__).resolve().parents[2] / "benchmarks" / "offline"


def test_doctests():
    r = doctest.testmod(S)
    assert r.failed == 0 and r.attempted > 10


def test_covariance_hand_case():
    m = ClassicRunningCovariance().fit(
        [1.0, 2.0, 3.0, float("nan")], [2.0, 4.0, 7.0, 1.0]
    )
    assert m.n_missing_ == 1 and m.count_ == 3
    assert abs(m.cov_ - 2.5) < 1e-12
    assert abs(m.corr_ - 2.5 / np.sqrt(57 / 9)) < 1e-12
    one = ClassicRunningCovariance().fit([1.0], [2.0])
    assert np.isnan(one.cov_) and np.isnan(one.corr_)
    flat = ClassicRunningCovariance().fit([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
    assert np.isnan(flat.corr_) and flat.cov_ == 0.0
    assert (
        ClassicRunningCorrelation().fit([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]).corr_ == -1.0
    )


def test_rolling_hand_case():
    m = ClassicRollingCovariance(window=3).fit(
        [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 5.0]
    )
    assert m.cov_series_.shape == (2,)
    assert abs(m.cov_series_[0] - 1.0) < 1e-12
    assert abs(m.cov_ - 1.5) < 1e-12
    short = ClassicRollingCorrelation(window=5).fit([1.0, 2.0], [1.0, 2.0])
    assert np.isnan(short.cov_) and np.isnan(short.corr_)
    flat = ClassicRollingCorrelation(window=3).fit([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
    assert np.isnan(flat.corr_)
    ok = ClassicRollingCorrelation(window=3).fit([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
    assert abs(ok.corr_ - 1.0) < 1e-12


def test_autocorrelation_hand_case():
    a = ClassicAutocorrelation(lag=1).fit([1.0, 2.0, 1.0, 2.0, 1.0])
    assert abs(a.value_ - (-0.8)) < 1e-12 and a.n_pairs_ == 4
    assert np.isnan(ClassicAutocorrelation(lag=3).fit([1.0, 2.0]).value_)
    assert np.isnan(ClassicAutocorrelation(lag=1).fit([2.0, 2.0, 2.0]).value_)


def test_moments_hand_case():
    m = ClassicRunningMoments().fit([1.0, 2.0, 3.0, 4.0, float("inf")])
    assert m.n_missing_ == 1 and m.count_ == 4
    assert m.mean_ == 2.5 and abs(m.var_ - 5.0 / 3.0) < 1e-12 and m.ptp_ == 3.0
    assert abs(m.skewness_) < 1e-12
    assert abs(m.kurtosis_ - (-1.36)) < 1e-12
    e = ClassicRunningMoments().fit([])
    assert e.count_ == 0 and np.isnan(e.mean_) and np.isnan(e.ptp_)
    one = ClassicRunningMoments().fit([5.0])
    assert np.isnan(one.var_) and np.isnan(one.skewness_)
    v = ClassicRunningMomentsVector().fit([[1.0, 10.0], [3.0, 20.0]])
    assert v.mean_.tolist() == [2.0, 15.0] and v.var_.tolist() == [2.0, 50.0]
    assert ClassicRunningMomentsVector().fit([1.0, 3.0]).mean_.tolist() == [2.0]


def test_ew_hand_case():
    e = ClassicEWStats(alpha=0.5).fit([1.0, 3.0, 3.0])
    assert abs(e.mean_ - 2.5) < 1e-12 and abs(e.var_ - 0.75) < 1e-12
    n = ClassicEWStats().fit([])
    assert n.count_ == 0 and np.isnan(n.mean_)


def test_quantiles_exact():
    d = ClassicDDSketch().fit([4.0, 1.0, 3.0, 2.0, float("nan")])
    assert d.n_missing_ == 1 and d.count_ == 4
    assert d.quantile(0.0) == 1.0 and d.quantile(1.0) == 4.0 and d.quantile(0.5) == 2.0
    assert d.min_ == 1.0 and d.max_ == 4.0
    t = ClassicTDigest().fit([])
    assert np.isnan(t.quantile(0.5)) and np.isnan(t.min_)


def test_rolling_median_hand_case():
    m = ClassicRollingMedianVector(window=3).fit(
        [[1.0, 9.0], [5.0, 1.0], [2.0, 4.0], [8.0, 6.0]]
    )
    assert m.series_.tolist() == [[2.0, 4.0], [5.0, 4.0]]
    assert m.median_.tolist() == [5.0, 4.0]
    short = ClassicRollingMedianVector(window=9).fit([1.0, 2.0])
    assert short.series_.shape == (0, 1) and np.isnan(short.median_[0])


def test_shrinkage_hand_case():
    X = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [2.0, 2.0]])
    c = ClassicOnlineShrinkageCovariance(2).fit(X)
    s = np.cov(X.T)
    tr, tr2 = np.trace(s), np.sum(s * s)
    alpha = 1 - (tr * tr / 5) / (tr2 - tr * tr / 2)
    assert abs(c.alpha_ - alpha) < 1e-12
    assert np.allclose(c.shrunk_, alpha * s + (1 - alpha) * tr * np.eye(2) / 2)
    flat = ClassicOnlineShrinkageCovariance(2).fit(np.zeros((3, 2)))
    assert flat.alpha_ == 1.0 and flat.beta_ == 0.0
    one = ClassicOnlineShrinkageCovariance(2).fit(np.zeros((1, 2)))
    assert one.shrunk_.shape == (2, 2)


def test_exact_sketch_targets():
    items = ["a", "b", "a", "c", "a", "b"]
    assert ClassicCountMinSketch().fit(items).estimate("a") == 3
    assert ClassicCountMinSketch().fit(items).estimate("z") == 0
    h = ClassicHyperLogLog().fit(items)
    assert h.count_distinct_ == 3 and h.count_total_ == 6
    b = ClassicBloomFilter().fit(items)
    assert b.check("c") and not b.check("z") and b.count_ == 3
    s = ClassicSpaceSaving(k=2).fit(items)
    assert (
        s.top() == [("a", 3), ("b", 2)]
        and s.top(1) == [("a", 3)]
        and s.estimate("c") == 1
    )


def test_base_is_not_online():
    import pytest

    from dense_armor.offline.base import ClassicBase

    with pytest.raises(NotImplementedError):
        ClassicBase().fit([1.0])
    assert not hasattr(ClassicRunningMoments, "learn_one")
    assert ClassicRunningMoments(feature="a").get_params() == {"feature": "a"}


def test_bench_json_matches_the_classes(tmp_path):
    spec = importlib.util.spec_from_file_location("bench_stats", BENCH / "stats.py")
    sys.path.insert(0, str(BENCH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "s.json"
    res = mod.common.run("stats", mod.CASES, mod.WHY, ["--smoke", "--out", str(out)])
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["classes"].keys() == res["classes"].keys() == mod.CASES.keys()
    for name in (
        "RunningCovariance",
        "RunningMoments",
        "EWStats",
        "OnlineShrinkageCovariance",
        "RollingMedianVector",
    ):
        assert saved["classes"][name]["max"] < 1e-9
    assert saved["classes"]["DDSketch"]["max"] < 0.02
    assert "not computed" in saved["wilcoxon_a12"]
    committed = json.loads((BENCH / "stats.json").read_text(encoding="utf-8"))
    assert (
        committed["classes"].keys() == saved["classes"].keys()
        and committed["seeds"] == 10
    )
