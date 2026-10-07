"""Tests for dense_armor/utility/stats/sketches.py."""
import doctest

import numpy as np
import pytest

import dense_armor.utility.stats.sketches as sketches_mod
from dense_armor.utility.stats.sketches import (
    CountMinSketch,
    HyperLogLog,
    BloomFilter,
    SpaceSaving,
)
from dense_armor.checks import check_estimator


def test_countmin_never_underestimates():
    cm = CountMinSketch(epsilon=0.001, delta=0.001)
    truth = {}
    rng = np.random.default_rng(0)
    for _ in range(5000):
        w = f"w{rng.integers(0, 100)}"
        truth[w] = truth.get(w, 0) + 1
        cm.learn_one({"w": w})
    for w, f in truth.items():
        assert cm.estimate(w) >= f


def test_countmin_error_bound():
    cm = CountMinSketch(epsilon=0.01, delta=0.01)
    truth = {}
    rng = np.random.default_rng(1)
    n = 5000
    for _ in range(n):
        w = f"w{rng.integers(0, 200)}"
        truth[w] = truth.get(w, 0) + 1
        cm.learn_one({"w": w})
    for w, f in truth.items():
        assert cm.estimate(w) - f <= 0.01 * n + 2


def test_countmin_merge_matches_whole():
    rng = np.random.default_rng(2)
    stream_a = [f"w{rng.integers(0, 50)}" for _ in range(500)]
    stream_b = [f"w{rng.integers(0, 50)}" for _ in range(500)]
    whole = CountMinSketch(epsilon=0.001, delta=0.001)
    for w in stream_a + stream_b:
        whole.learn_one({"w": w})
    a = CountMinSketch(epsilon=0.001, delta=0.001)
    b = CountMinSketch(epsilon=0.001, delta=0.001)
    for w in stream_a:
        a.learn_one({"w": w})
    for w in stream_b:
        b.learn_one({"w": w})
    merged = a.merge(b)
    for w in set(stream_a + stream_b):
        assert merged.estimate(w) == whole.estimate(w)


def test_countmin_merge_type_error():
    with pytest.raises(TypeError):
        CountMinSketch().merge("x")


def test_countmin_invalid_params():
    with pytest.raises(ValueError):
        CountMinSketch(epsilon=0.0)
    with pytest.raises(ValueError):
        CountMinSketch(delta=1.5)


def test_check_estimator_countmin():
    check_estimator(CountMinSketch())


def test_hll_distinct_count_small():
    h = HyperLogLog(precision=10)
    for i in range(1000):
        h.learn_one({"w": f"item{i}"})
    assert abs(h.count() - 1000) / 1000 < 0.1


def test_hll_distinct_count_large():
    h = HyperLogLog(precision=14)
    n = 100_000
    for i in range(n):
        h.learn_one({"w": f"item{i}"})
    assert abs(h.count() - n) / n < 0.05


def test_hll_low_cardinality_exact_after_correction():
    h = HyperLogLog(precision=14)
    for i in range(50):
        h.learn_one({"w": f"item{i}"})
    assert abs(h.count() - 50) / 50 < 0.2


def test_hll_merge():
    a = HyperLogLog(precision=12)
    b = HyperLogLog(precision=12)
    for i in range(2000):
        a.learn_one({"w": f"a{i}"})
    for i in range(2000):
        b.learn_one({"w": f"b{i}"})
    merged = a.merge(b)
    assert abs(merged.count() - 4000) / 4000 < 0.1


def test_hll_merge_type_error():
    with pytest.raises(TypeError):
        HyperLogLog().merge("x")


def test_hll_merge_precision_mismatch():
    with pytest.raises(ValueError):
        HyperLogLog(precision=10).merge(HyperLogLog(precision=12))


def test_hll_invalid_precision():
    with pytest.raises(ValueError):
        HyperLogLog(precision=3)
    with pytest.raises(ValueError):
        HyperLogLog(precision=20)


def test_check_estimator_hll():
    check_estimator(HyperLogLog(precision=10))


def test_bloom_no_false_negatives():
    b = BloomFilter(n_expected=1000, p_false=0.01)
    for i in range(1000):
        b.learn_one({"w": f"item{i}"})
    for i in range(1000):
        assert b.check(f"item{i}")


def test_bloom_false_positive_rate():
    b = BloomFilter(n_expected=1000, p_false=0.01)
    for i in range(1000):
        b.learn_one({"w": f"in{i}"})
    fp = sum(1 for i in range(10_000) if b.check(f"out{i}"))
    assert fp / 10_000 < 0.05


def test_bloom_merge():
    a = BloomFilter(n_expected=500, p_false=0.01)
    b = BloomFilter(n_expected=500, p_false=0.01)
    for i in range(500):
        a.learn_one({"w": f"a{i}"})
    for i in range(500):
        b.learn_one({"w": f"b{i}"})
    merged = a.merge(b)
    for i in range(500):
        assert merged.check(f"a{i}")
        assert merged.check(f"b{i}")


def test_bloom_merge_type_error():
    with pytest.raises(TypeError):
        BloomFilter().merge("x")


def test_bloom_invalid_params():
    with pytest.raises(ValueError):
        BloomFilter(n_expected=0)
    with pytest.raises(ValueError):
        BloomFilter(p_false=2.0)


def test_check_estimator_bloom():
    check_estimator(BloomFilter(n_expected=100))


def test_spacesaving_finds_heavy_hitter():
    s = SpaceSaving(k=5)
    for _ in range(1000):
        s.learn_one({"w": "hot"})
    for i in range(200):
        s.learn_one({"w": f"cold{i}"})
    assert s.top(1)[0][0] == "hot"
    assert s.estimate("hot") == 1000


def test_spacesaving_error_bound():
    s = SpaceSaving(k=10)
    truth = {}
    rng = np.random.default_rng(3)
    n = 10_000
    for _ in range(n):
        w = f"w{rng.integers(0, 500)}"
        truth[w] = truth.get(w, 0) + 1
        s.learn_one({"w": w})
    top_truth = sorted(truth.items(), key=lambda kv: -kv[1])[:10]
    for item, true_count in top_truth:
        est = s.estimate(item)
        assert est >= true_count - n // 10


def test_spacesaving_merge():
    a = SpaceSaving(k=5)
    b = SpaceSaving(k=5)
    for _ in range(100):
        a.learn_one({"w": "x"})
    for _ in range(50):
        b.learn_one({"w": "x"})
    for _ in range(200):
        b.learn_one({"w": "y"})
    merged = a.merge(b)
    assert merged.estimate("x") >= 100
    assert merged.estimate("y") >= 50


def test_spacesaving_merge_type_error():
    with pytest.raises(TypeError):
        SpaceSaving().merge("x")


def test_spacesaving_invalid_k():
    with pytest.raises(ValueError):
        SpaceSaving(k=0)


def test_check_estimator_spacesaving():
    check_estimator(SpaceSaving(k=3))


def test_doctests():
    assert doctest.testmod(sketches_mod).failed == 0
