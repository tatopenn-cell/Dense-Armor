"""Edge paths of the online statistics: update aliases, transform_one, empty states, merges."""
import math

import numpy as np
import pytest

import dense_armor.utility.stats as st

CLASSES = [getattr(st, n) for n in st.__all__ if isinstance(getattr(st, n), type)]


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_update_alias_transform_and_properties(cls):
    m = cls()
    rng = np.random.default_rng(0)
    for i in range(40):
        m.update({"x": float(rng.normal()), "y": float(rng.normal())}, t=i * 0.01)
    out = m.transform_one({"x": 1.0, "y": 2.0})
    assert isinstance(out, dict)
    for name in ("count", "n_missing", "window_size", "n_pairs", "min", "max", "n_tracked",
                 "width", "depth", "n_bits", "n_hashes", "fill_ratio", "count_total", "std"):
        if hasattr(type(m), name):
            getattr(m, name)


def test_running_covariance_merge_with_empty_sides():
    a, b = st.RunningCovariance(), st.RunningCovariance()
    for v in (1.0, 2.0, 4.0):
        a.learn_one({"x": v, "y": 2 * v})
    assert a.merge(st.RunningCovariance()).cov == pytest.approx(a.cov)
    assert st.RunningCovariance().merge(a).cov == pytest.approx(a.cov)
    assert b.merge(st.RunningCovariance()).count == 0


def test_rolling_covariance_seconds_window_missing_and_empty():
    r = st.RollingCovariance(window_s=0.05)
    assert r.cov == 0.0
    assert r.corr == 0.0
    r.learn_one({"x": float("nan"), "y": 1.0}, t=0.0)
    for i in range(1, 30):
        r.learn_one({"x": float(i), "y": float(i)}, t=i * 0.01)
    assert r.n_missing == 1
    assert r.window_size == 5
    r2 = st.RollingCovariance(window=10)
    for _ in range(10):
        r2.learn_one({"x": 1.0, "y": float(_)})
    assert r2.corr == 0.0


def test_autocorrelation_empty_and_flat():
    a = st.Autocorrelation()
    assert a.value == 0.0
    for _ in range(30):
        a.learn_one({"x": 1.0})
    assert a.value == 0.0
    assert a.n_pairs >= 0


def test_ddsketch_edges():
    with pytest.raises(ValueError):
        st.DDSketch(max_buckets=1)
    d = st.DDSketch(alpha=0.01, max_buckets=4)
    assert d.quantile(0.5) == 0.0
    for v in (0.0, -5.0, -1.0, 1e-3, 1.0, 10.0, 1e3, 1e6):
        d.learn_one({"x": v})
    assert d.quantile(0.0) == d.min
    assert d.quantile(1.0) == d.max
    e = st.DDSketch(alpha=0.01)
    for v in (-2.0, -3.0, 4.0):
        e.learn_one({"x": v})
    f = st.DDSketch(alpha=0.01)
    for v in (-2.0, -7.0):
        f.learn_one({"x": v})
    assert e.merge(f).count == 5


def test_tdigest_edges():
    t = st.TDigest(delta=50, buffer_size=8)
    assert t.quantile(0.5) == 0.0
    for v in np.linspace(-3, 3, 200):
        t.learn_one({"x": float(v)})
    assert t.quantile(0.0) == pytest.approx(-3.0)
    assert t.quantile(1.0) == pytest.approx(3.0)
    assert t.quantile(1e-6) <= t.quantile(0.5) <= t.quantile(1 - 1e-6)
    u = st.TDigest(delta=50, buffer_size=8)
    for v in (10.0, 11.0, 12.0):
        u.learn_one({"x": v})
    assert t.merge(u).count == 203
    assert t.n_centroids > 0


@pytest.mark.parametrize("p", [4, 5, 6, 8])
def test_hyperloglog_small_precisions(p):
    h = st.HyperLogLog(precision=p)
    for i in range(500):
        h.learn_one({"x": f"item{i}"})
    assert h.count() == pytest.approx(500, rel=0.6)
    assert h.count_total == 500


def test_sketch_merge_mismatch_and_contains():
    with pytest.raises(ValueError):
        st.CountMinSketch(epsilon=0.01).merge(st.CountMinSketch(epsilon=0.1))
    with pytest.raises(ValueError):
        st.BloomFilter(n_expected=100).merge(st.BloomFilter(n_expected=1000))
    with pytest.raises(ValueError):
        st.SpaceSaving(k=5).merge(st.SpaceSaving(k=6))
    b = st.BloomFilter(n_expected=100)
    b.learn_one({"x": "a"})
    assert "a" in b
    assert 0.0 < b.fill_ratio < 1.0


def test_moments_vector_merge_errors_and_empty():
    m = st.RunningMomentsVector()
    assert m.transform_one({"a": 1.0}) == {}
    with pytest.raises(TypeError):
        m.merge(object())
    with pytest.raises(ValueError):
        m.merge(st.RunningMomentsVector())
    a, b = st.RunningMomentsVector(), st.RunningMomentsVector()
    a.learn_one({"a": 1.0, "b": 2.0})
    b.learn_one({"c": 1.0})
    with pytest.raises(ValueError):
        a.merge(b)
    assert st.RunningMoments().ptp == 0.0


def test_ewstats_missing_and_std():
    e = st.EWStats(alpha=0.5)
    e.learn_one({"x": float("nan")})
    for v in (1.0, 3.0, 5.0):
        e.learn_one({"x": v})
    assert e.n_missing == 1
    assert e.count == 3
    assert e.std == pytest.approx(math.sqrt(e.var))
