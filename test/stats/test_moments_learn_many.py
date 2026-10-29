"""Tests for learn_many on RunningMoments and RunningMomentsVector."""

import numpy as np

from dense_armor.utility.stats.moments import (
    RunningMoments,
    RunningMomentsVector,
)


def _loop(arr):
    m = RunningMoments()
    for v in arr:
        m.learn_one({"x": float(v)})
    return m


def test_learn_many_matches_the_loop():
    rng = np.random.default_rng(0)
    arr = rng.normal(0.0, 1.0, 10000)
    a = _loop(arr)
    b = RunningMoments().learn_many(arr)
    assert a.n_ == b.n_
    assert abs(a.mean_ - b.mean_) < 1e-9
    assert abs(a.m2_ - b.m2_) < 1e-6 * max(1.0, abs(a.m2_))
    assert abs(a.m3_ - b.m3_) < 1e-6 * max(1.0, abs(a.m3_))
    assert abs(a.m4_ - b.m4_) < 1e-6 * max(1.0, abs(a.m4_))


def test_learn_many_in_chunks_matches_one_call():
    rng = np.random.default_rng(1)
    arr = rng.normal(0.0, 1.0, 10000)
    one = RunningMoments().learn_many(arr)
    for cut in (100, 1000, 5000):
        chunk = RunningMoments()
        for i in range(0, len(arr), cut):
            chunk.learn_many(arr[i : i + cut])
        assert abs(one.mean_ - chunk.mean_) < 1e-9
        assert abs(one.m2_ - chunk.m2_) < 1e-6 * max(1.0, abs(one.m2_))
        assert abs(one.m3_ - chunk.m3_) < 1e-6 * max(1.0, abs(one.m3_))
        assert abs(one.m4_ - chunk.m4_) < 1e-6 * max(1.0, abs(one.m4_))


def test_learn_many_min_max():
    m = RunningMoments().learn_many([1.0, -2.0, 5.0, 3.0])
    assert m.min_ == -2.0
    assert m.max_ == 5.0


def test_learn_many_skips_nan():
    m = RunningMoments()
    m.learn_many(np.array([1.0, 2.0, np.nan, 3.0]))
    assert m.n_ == 3


def test_learn_many_all_nan_counts_as_missing():
    m = RunningMoments()
    m.learn_many(np.array([np.nan, np.nan]))
    assert m.n_missing_ == 2


def test_learn_many_empty_returns_self():
    m = RunningMoments().learn_many([])
    assert m.n_ == 0


def test_learn_many_with_dict_and_feature():
    m = RunningMoments(feature="y").learn_many({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    assert m.n_ == 2
    assert m.mean_ == 3.5


def test_learn_many_vector():
    v = RunningMomentsVector(features=["a", "b"])
    v.learn_many({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    assert v.channels_["a"].n_ == 3
    assert v.channels_["b"].n_ == 3
    assert abs(v.channels_["a"].mean_ - 2.0) < 1e-12
    assert abs(v.channels_["b"].mean_ - 5.0) < 1e-12


def test_learn_many_vector_matches_loop():
    rng = np.random.default_rng(2)
    A = rng.normal(0.0, 1.0, 1000)
    B = rng.normal(1.0, 2.0, 1000)
    v = RunningMomentsVector(features=["a", "b"])
    for i in range(1000):
        v.learn_one({"a": float(A[i]), "b": float(B[i])})
    v2 = RunningMomentsVector(features=["a", "b"])
    v2.learn_many({"a": A, "b": B})
    for k in ("a", "b"):
        assert abs(v.channels_[k].mean_ - v2.channels_[k].mean_) < 1e-9
        assert abs(v.channels_[k].m2_ - v2.channels_[k].m2_) < 1e-6 * max(
            1.0, abs(v.channels_[k].m2_)
        )


def test_block_constant_large_values_no_overflow():
    """Non-regression: `_block` returns m2=0 for constant large values.

    With 500,000 samples at 1.79e301, `a.mean()` is 1.79e301 with a
    rounding error of ~1e285. The original `_block` computes
    `(a - mean)**2`, which is ~1e570 and overflows to `inf`. The
    shift-based `_block` subtracts `a[0]` first, so
    `(a - a[0]) - mean_shifted` is exactly zero and `m2 = 0`.
    """
    import numpy as np

    from dense_armor.utility.stats.moments import RunningMoments

    max_f = np.finfo(np.float64).max / 1e5
    min_f = max_f / 1e2
    batch = np.full(500000, min_f, dtype=np.float64)

    m = RunningMoments()
    m.learn_many(batch)

    assert np.isfinite(m.mean_), f"mean: {m.mean_}"
    assert np.isfinite(m.m2_), f"m2: {m.m2_}"
    assert m.m2_ == 0.0, f"m2 deve essere 0, invece è {m.m2_}"
    assert m.var == 0.0, f"var deve essere 0, invece è {m.var}"


def test_learn_one_first_sample_does_not_overflow():
    """Non-regression: the first sample of `learn_one` is stored as is.

    For `v = 1e307`, the first update computed `d * dn = inf` and then
    `inf * n_old` with `n_old = 0`, so `m2_` became `nan`. Identical
    values at 1e307 (float64) and 1e38 give variance 0.
    """
    for v in (1e307, 1e38, -1e307):
        m = RunningMoments()
        for _ in range(3):
            m.learn_one({"x": v})
        assert m.n_ == 3
        assert m.mean_ == v
        assert (m.m2_, m.m3_, m.m4_) == (0.0, 0.0, 0.0)
        assert m.var == 0.0
        assert m.min_ == m.max_ == v


def test_learn_one_first_sample_matches_the_old_update():
    """The shortcut gives the same state as the general update for n_old = 0."""
    m = RunningMoments()
    m.learn_one({"x": 3.5})
    assert (m.n_, m.mean_, m.m2_, m.m3_, m.m4_) == (1, 3.5, 0.0, 0.0, 0.0)
    assert m.min_ == m.max_ == 3.5
    m.learn_one({"x": 4.5})
    assert (m.n_, m.mean_) == (2, 4.0)
    assert m.m2_ == 0.5
