"""Tests for dense_armor/utility/stats/quantiles.py."""
import doctest

import numpy as np
import pytest

import dense_armor.utility.stats.quantiles as quantiles_mod
from dense_armor.utility.stats.quantiles import DDSketch, TDigest
from dense_armor.checks import check_estimator


def _stream(n: int = 5000, seed: int = 0, loc: float = 0.0, scale: float = 1.0):
    rng = np.random.default_rng(seed)
    return list(rng.normal(loc, scale, n))


def test_ddsketch_relative_error_guarantee_gaussian():
    xs = _stream(5000, 0, 100.0, 20.0)
    s = DDSketch(alpha=0.01)
    for v in xs:
        s.learn_one({"x": v})
    for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
        exact = float(np.quantile(xs, q))
        est = s.quantile(q)
        rel = abs(est - exact) / max(abs(exact), 1e-12)
        assert rel <= 0.01 + 1e-9, f"q={q} exact={exact} est={est} rel={rel}"


def test_ddsketch_relative_error_guarantee_pareto():
    rng = np.random.default_rng(1)
    xs = list(1.0 + rng.pareto(1.5, 5000))
    s = DDSketch(alpha=0.02)
    for v in xs:
        s.learn_one({"x": v})
    for q in (0.5, 0.9, 0.99, 0.999):
        exact = float(np.quantile(xs, q))
        est = s.quantile(q)
        rel = abs(est - exact) / abs(exact)
        assert rel <= 0.02 + 1e-9, f"q={q} exact={exact} est={est} rel={rel}"


def test_ddsketch_size_is_bounded():
    rng = np.random.default_rng(2)
    xs = 1.0 + rng.pareto(1.5, 200_000)
    s = DDSketch(alpha=0.01, max_buckets=2048)
    for v in xs:
        s.learn_one({"x": float(v)})
    assert s.size <= 2 * 2048


def test_ddsketch_merge_matches_whole():
    xs = _stream(2000, 0, 50.0, 10.0)
    whole = DDSketch(alpha=0.01)
    for v in xs:
        whole.learn_one({"x": v})
    a = DDSketch(alpha=0.01)
    b = DDSketch(alpha=0.01)
    for v in xs[:1000]:
        a.learn_one({"x": v})
    for v in xs[1000:]:
        b.learn_one({"x": v})
    merged = a.merge(b)
    assert merged.count == whole.count
    for q in (0.25, 0.5, 0.75, 0.9):
        assert merged.quantile(q) == pytest.approx(
            whole.quantile(q), rel=1e-9
        )


def test_ddsketch_negative_values():
    xs = list(np.random.default_rng(3).normal(-5.0, 1.0, 500))
    s = DDSketch(alpha=0.01)
    for v in xs:
        s.learn_one({"x": float(v)})
    assert s.min is not None and s.min < 0
    assert s.quantile(0.5) < 0


def test_ddsketch_nan_skipped():
    s = DDSketch(alpha=0.01)
    for v in [1.0, float("nan"), 2.0, float("nan"), 3.0]:
        s.learn_one({"x": v})
    assert s.count == 3
    assert s.n_missing == 2


def test_ddsketch_empty_returns_zero():
    s = DDSketch(alpha=0.01)
    assert s.quantile(0.5) == 0.0


def test_ddsketch_invalid_alpha():
    with pytest.raises(ValueError):
        DDSketch(alpha=0.0)
    with pytest.raises(ValueError):
        DDSketch(alpha=1.0)


def test_ddsketch_merge_type_error():
    with pytest.raises(TypeError):
        DDSketch().merge("x")


def test_ddsketch_merge_alpha_mismatch():
    with pytest.raises(ValueError):
        DDSketch(alpha=0.01).merge(DDSketch(alpha=0.05))


def test_check_estimator_ddsketch():
    check_estimator(DDSketch())


def test_tdigest_accuracy_gaussian():
    xs = _stream(10_000, 0, 0.0, 1.0)
    t = TDigest(delta=200)
    for v in xs:
        t.learn_one({"x": v})
    for q in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
        exact = float(np.quantile(xs, q))
        est = t.quantile(q)
        assert abs(est - exact) < 0.1, f"q={q} exact={exact} est={est}"


def test_tdigest_tail_accuracy_pareto():
    rng = np.random.default_rng(4)
    xs = list(1.0 + rng.pareto(1.5, 20_000))
    t = TDigest(delta=200)
    for v in xs:
        t.learn_one({"x": v})
    for q in (0.9, 0.99, 0.999):
        exact = float(np.quantile(xs, q))
        est = t.quantile(q)
        rel = abs(est - exact) / abs(exact)
        assert rel < 0.15, f"q={q} exact={exact} est={est} rel={rel}"


def test_tdigest_centroids_bounded():
    xs = _stream(50_000, 5)
    t = TDigest(delta=100)
    for v in xs:
        t.learn_one({"x": v})
    assert t.n_centroids <= int(np.ceil(100)) + 5


def test_tdigest_merge_close_to_whole():
    xs = _stream(10_000, 6)
    whole = TDigest(delta=100)
    for v in xs:
        whole.learn_one({"x": v})
    a = TDigest(delta=100)
    b = TDigest(delta=100)
    for v in xs[:5000]:
        a.learn_one({"x": v})
    for v in xs[5000:]:
        b.learn_one({"x": v})
    merged = a.merge(b)
    assert merged.count == whole.count
    for q in (0.1, 0.5, 0.9):
        assert abs(merged.quantile(q) - whole.quantile(q)) < 0.2


def test_tdigest_nan_skipped():
    t = TDigest(delta=50)
    for v in [1.0, float("nan"), 2.0]:
        t.learn_one({"x": v})
    assert t.count == 2
    assert t.n_missing == 1


def test_tdigest_empty_returns_zero():
    t = TDigest(delta=100)
    assert t.quantile(0.5) == 0.0


def test_tdigest_invalid_delta():
    with pytest.raises(ValueError):
        TDigest(delta=0.0)


def test_tdigest_merge_type_error():
    with pytest.raises(TypeError):
        TDigest().merge("x")


def test_tdigest_merge_delta_mismatch():
    with pytest.raises(ValueError):
        TDigest(delta=100).merge(TDigest(delta=200))


def test_check_estimator_tdigest():
    check_estimator(TDigest(delta=50))


def test_doctests():
    assert doctest.testmod(quantiles_mod).failed == 0
