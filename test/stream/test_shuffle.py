"""Tests for shuffle."""

import numpy as np
import pytest

from dense_armor.utility.stream import iter_array, shuffle


def test_shuffle_same_multiset():
    X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    stream = iter_array(X, names=["x"], units=[""])
    out = [float(sig["x"]) for sig, _ in shuffle(stream, buffer_size=3, seed=0)]
    assert sorted(out) == list(X)


def test_shuffle_deterministic():
    X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    a = [
        float(sig["x"])
        for sig, _ in shuffle(iter_array(X, names=["x"], units=[""]), 3, seed=0)
    ]
    b = [
        float(sig["x"])
        for sig, _ in shuffle(iter_array(X, names=["x"], units=[""]), 3, seed=0)
    ]
    assert a == b


def test_shuffle_rejects_bad_buffer():
    X = np.array([1.0])
    with pytest.raises(ValueError):
        list(shuffle(iter_array(X, names=["x"], units=[""]), buffer_size=0))
