"""Tests for the sliding-window skyline (Börzsönyi 2001, Tao 2006)."""

import doctest

import numpy as np
import pytest

from dense_armor.utility.misc import Skyline
from dense_armor.utility.misc import skyline as skyline_module


def _brute_skyline(points):
    pts = [np.asarray(p, dtype=float) for p in points]
    keep = []
    for i, q in enumerate(pts):
        dominated = False
        for j, p in enumerate(pts):
            if i == j:
                continue
            if np.all(p <= q) and np.any(p < q):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def test_dominance_definition():
    from dense_armor.utility.misc.skyline import _dominates

    assert _dominates(np.array([1.0, 1.0]), np.array([2.0, 2.0]))
    assert _dominates(np.array([1.0, 1.0]), np.array([1.0, 2.0]))
    assert not _dominates(np.array([1.0, 1.0]), np.array([1.0, 1.0]))
    assert not _dominates(np.array([1.0, 2.0]), np.array([2.0, 1.0]))


def test_skyline_matches_bruteforce():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = rng.integers(3, 15)
        pts = rng.normal(size=(n, 2))
        s = Skyline(window=n)
        for p in pts:
            _ = s.update({"x": float(p[0]), "y": float(p[1])})
        got = s.skyline()
        expected = _brute_skyline(pts)
        assert sorted(got) == sorted(expected)


def test_sliding_window_drops_oldest():
    s = Skyline(window=3)
    for p in [
        {"x": 1.0, "y": 5.0},
        {"x": 3.0, "y": 3.0},
        {"x": 5.0, "y": 1.0},
        {"x": 0.5, "y": 0.5},
    ]:
        _ = s.update(p)
    pts = [(1.0, 5.0), (3.0, 3.0), (5.0, 1.0), (0.5, 0.5)][-3:]
    got = s.skyline()
    expected = _brute_skyline(pts)
    assert sorted(got) == sorted(expected)


def test_single_point_is_its_own_skyline():
    s = Skyline(window=5)
    _ = s.update({"x": 1.0, "y": 1.0})
    assert s.skyline() == [0]


def test_nan_skipped():
    s = Skyline(window=3)
    _ = s.update({"x": 1.0, "y": 1.0})
    _ = s.update({"x": float("nan"), "y": 0.0})
    _ = s.update({"x": 0.5, "y": 0.5})
    assert s.n_missing == 1
    assert s.n_seen_ == 2


def test_missing_key_skipped():
    s = Skyline(window=3)
    _ = s.update({"x": 1.0, "y": 1.0})
    _ = s.update({"x": 0.5})
    assert s.n_missing == 1


def test_rejects_zero_window():
    with pytest.raises(ValueError):
        Skyline(window=0)


def test_doctest():
    res = doctest.testmod(skyline_module, verbose=False)
    assert res.failed == 0
