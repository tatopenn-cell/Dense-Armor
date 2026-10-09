"""Tests for iter_array."""

import numpy as np
import pytest

from dense_armor.utility.stream import iter_array


def test_iter_array_default_names():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = list(iter_array(X))
    assert len(out) == 2
    sig0, y0 = out[0]
    assert sig0.names == ["x0", "x1"]
    assert sig0.t == 0.0
    assert y0 is None
    assert sig0["x0"] == 1.0


def test_iter_array_with_y_and_t():
    X = np.array([[1.0], [3.0], [5.0]])
    y = np.array([0, 1, 0])
    t = np.array([10.0, 11.0, 12.0])
    out = list(iter_array(X, y=y, t=t, names=["v"], units=["m/s"]))
    assert [o[0].t for o in out] == [10.0, 11.0, 12.0]
    assert [o[1] for o in out] == [0, 1, 0]
    assert out[0][0].units == ["m/s"]


def test_iter_array_1d():
    X = np.array([1.0, 2.0, 3.0])
    out = list(iter_array(X))
    assert len(out) == 3
    assert out[0][0].names == ["x0"]


def test_iter_array_rejects_bad_names():
    X = np.zeros((3, 2))
    with pytest.raises(ValueError):
        list(iter_array(X, names=["a"]))


def test_iter_array_rejects_bad_lengths():
    X = np.zeros((3, 2))
    with pytest.raises(ValueError):
        list(iter_array(X, y=np.zeros(2)))
    with pytest.raises(ValueError):
        list(iter_array(X, t=np.zeros(4)))


def test_iter_array_rejects_3d():
    X = np.zeros((2, 2, 2))
    with pytest.raises(ValueError):
        list(iter_array(X))
