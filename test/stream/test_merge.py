"""Tests for merge_by_time."""

import numpy as np

from dense_armor.utility.stream import iter_array, merge_by_time


def test_merge_sorted():
    a = iter_array(np.array([1.0, 3.0]), t=[0.0, 2.0], names=["a"], units=[""])
    b = iter_array(np.array([2.0, 4.0]), t=[1.0, 3.0], names=["b"], units=[""])
    out = list(merge_by_time(a, b))
    assert [s.t for s, _ in out] == [0.0, 1.0, 2.0, 3.0]
    assert [next(iter(s.to_dict())) for s, _ in out] == ["a", "b", "a", "b"]


def test_merge_equal_timestamps_keep_order():
    a = iter_array(np.array([1.0]), t=[0.0], names=["a"], units=[""])
    b = iter_array(np.array([2.0]), t=[0.0], names=["b"], units=[""])
    c = iter_array(np.array([3.0]), t=[0.0], names=["c"], units=[""])
    out = list(merge_by_time(a, b, c))
    assert [next(iter(s.to_dict())) for s, _ in out] == ["a", "b", "c"]


def test_merge_every_sample_once():
    a = iter_array(np.arange(100.0), t=np.arange(100.0), names=["a"], units=[""])
    b = iter_array(np.arange(100.0) * 2, t=np.arange(100.0), names=["b"], units=[""])
    out = list(merge_by_time(a, b))
    assert len(out) == 200
    assert [s.t for s, _ in out] == sorted(s.t for s, _ in out)
