"""Tests for HalfSpaceTrees."""

import numpy as np
import pytest

from dense_armor.checks import check_estimator
from dense_armor.utility.anomaly.hst import HalfSpaceTrees


def test_hst_shapes():
    hst = HalfSpaceTrees(n_trees=5, depth=6, window=30, range_init=5, seed=0)
    for _ in range(10):
        hst.learn_one({"a": 0.5, "b": 0.5})
    s = hst.score_one({"a": 0.5, "b": 0.5})
    assert isinstance(s, float)


def test_hst_outlier_scores_higher():
    """An out-of-range point scores higher than the centre of the range.

    The working range is given explicitly: when the range is learned
    online from the first samples of a narrow distribution, a point
    far outside the working range is mapped to the same outermost leaf
    as a point at the centre (with the range learned from
    ``N(0, 0.05)`` both ``(0, 0)`` and ``(10, 10)`` reach the rightmost
    leaf of every tree), so the test would not measure the geometry.
    The range ``[-1, 1]`` on both features puts the centre and the far
    point in different leaves.
    """
    hst = HalfSpaceTrees(
        n_trees=15,
        depth=8,
        window=100,
        feature_ranges=[(-1.0, 1.0), (-1.0, 1.0)],
        seed=0,
    )
    rng = np.random.default_rng(0)
    for _ in range(200):
        hst.learn_one(
            {
                "a": float(rng.normal(0.0, 0.05)),
                "b": float(rng.normal(0.0, 0.05)),
            }
        )
    s_in = hst.score_one({"a": 0.0, "b": 0.0})
    s_out = hst.score_one({"a": 10.0, "b": 10.0})
    assert s_out > s_in


def _walk_node_ranges(hst, tr):
    """Restituisce (split_att, split_val, mid_of_range) per ogni nodo interno.

    Cammina l'albero ricostruendo il range del nodo feature per feature,
    come il codice fa in _path_one. Il range dipende solo dall'albero,
    non dai dati.
    """
    assert hst._lo is not None and hst._hi is not None
    assert hst._feature is not None
    lo_root = hst._lo.copy()
    hi_root = hst._hi.copy()
    n_internal = 2**hst.depth - 1
    out = []
    # BFS: per ogni nodo, tengo traccia del range su ogni feature
    queue = [(0, lo_root, hi_root)]
    while queue:
        node, lo, hi = queue.pop(0)
        if node >= n_internal:
            continue
        feat = int(hst._feature[tr, node])
        mid = 0.5 * (lo[feat] + hi[feat])
        out.append((feat, mid, lo.copy(), hi.copy()))
        lo_l = lo.copy()
        hi_l = hi.copy()
        hi_l[feat] = mid
        lo_r = lo.copy()
        hi_r = hi.copy()
        lo_r[feat] = mid
        queue.append((2 * node + 1, lo_l, hi_l))
        queue.append((2 * node + 2, lo_r, hi_r))
    return out


@pytest.mark.parametrize("d", [1, 2, 4])
def test_hst_split_is_midpoint_of_node_range(d):
    """Ogni split e' il punto medio del range del suo nodo sulla sua feature.

    Regola di Tan et al. 2011, Section 3.2 ("the split point is the
    mid-point of the current range of q") e Ulrich 2025, Section 5.2
    ("s = (x + y) / 2"). Nel codice precedente il punto di taglio era
    ``(2k+1) / 2^(level+1)`` dalla posizione BFS del nodo, corretto solo
    per ``d = 1``.
    """
    ranges = [(-1.0, 1.0)] * d
    for seed in (0, 1, 7):
        hst = HalfSpaceTrees(
            n_trees=1,
            depth=6,
            window=20,
            feature_ranges=ranges,
            seed=seed,
        )
        hst.learn_one({chr(ord("a") + i): 0.0 for i in range(d)})
        walks = _walk_node_ranges(hst, 0)
        for feat, mid_used, lo, hi in walks:
            expected = 0.5 * (lo[feat] + hi[feat])
            assert abs(mid_used - expected) < 1e-12, (
                f"d={d} seed={seed} feat={feat}: mid={mid_used} != {expected}"
            )


def test_hst_nan_counted_missing():
    hst = HalfSpaceTrees(n_trees=5, depth=6, window=20, range_init=5, seed=0)
    hst.learn_one({"a": float("nan"), "b": 0.5})
    assert hst.n_missing == 1


def test_hst_rejects_bad_params():
    with pytest.raises(ValueError):
        HalfSpaceTrees(n_trees=0)
    with pytest.raises(ValueError):
        HalfSpaceTrees(depth=0)
    with pytest.raises(ValueError):
        HalfSpaceTrees(window=0)
    with pytest.raises(ValueError):
        HalfSpaceTrees(range_init=0)
    with pytest.raises(ValueError):
        HalfSpaceTrees(threshold_quantile=0.0)
    with pytest.raises(ValueError):
        HalfSpaceTrees(threshold_quantile=1.0)


def test_hst_explicit_ranges():
    hst = HalfSpaceTrees(
        n_trees=5,
        depth=6,
        window=20,
        feature_ranges=[(-10.0, 10.0), (-10.0, 10.0)],
        seed=0,
    )
    hst.learn_one({"a": 5.0, "b": -5.0})
    s = hst.score_one({"a": 5.0, "b": -5.0})
    assert isinstance(s, float)


def test_hst_online_ranges_learned():
    hst = HalfSpaceTrees(n_trees=5, depth=6, window=30, range_init=10, seed=0)
    for i in range(30):
        hst.learn_one({"a": float(i), "b": float(-i)})
    assert hst._ready
    assert hst._lo is not None and hst._hi is not None
    assert hst._lo[0] < 0.0
    assert hst._hi[0] > 5.0


def test_hst_adaptive_threshold_flags_nothing_at_warmup():
    hst = HalfSpaceTrees(
        n_trees=5,
        depth=6,
        window=30,
        range_init=5,
        threshold_warmup=50,
        seed=0,
    )
    for _ in range(10):
        hst.learn_one({"a": 0.5, "b": 0.5})
    assert hst.threshold == 1e9
    assert not hst.is_outlier({"a": 0.5, "b": 0.5})


def test_hst_window_bounds_memory():
    hst = HalfSpaceTrees(n_trees=3, depth=5, window=10, range_init=5, seed=0)
    for i in range(100):
        hst.learn_one({"a": float(i % 10) / 10.0})
    assert hst._buf_size == 10


def test_hst_check_estimator():
    check_estimator(HalfSpaceTrees(n_trees=3, depth=5, window=20, range_init=5, seed=0))
