"""Tests for learn_many on RunningMoments and RunningMomentsVector."""

import numpy as np
import pytest

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
            chunk.learn_many(arr[i:i + cut])
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
    assert m.n_missing_ == 1


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
        assert abs(v.channels_[k].m2_ - v2.channels_[k].m2_) < 1e-6 * max(1.0, abs(v.channels_[k].m2_))
