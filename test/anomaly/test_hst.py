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
    """An out-of-range point scores higher than the centre of dense data.

    The working range is given explicitly. When it is learned from the
    first ``range_init`` samples of ``N(0, 0.05)``, the samples that
    later fall outside that range are mapped to the outermost leaves,
    so those leaves are not empty: in this configuration the far point
    ``(10, 10)`` lands in leaves with a mean count of 1.27 per tree and
    the centre ``(0, 0)`` in leaves with a mean count of 0.80, and the
    far point does not score higher.
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
    """Return (feature, split value used, expected mid-point) per internal node.

    The walk rebuilds the range of every node in the normalised space of
    the detector, where the root ranges over ``[0, 1]`` on every feature,
    and descends with the split values the detector stores in
    ``_midpoints``. The expected value is the middle of the node's range
    on its split feature.
    """
    assert hst._feature is not None and hst._midpoints is not None
    n_internal = 2**hst.depth - 1
    lo_root = np.zeros(len(hst._keys))
    hi_root = np.ones(len(hst._keys))
    out = []
    queue = [(0, lo_root, hi_root)]
    while queue:
        node, lo, hi = queue.pop(0)
        if node >= n_internal:
            continue
        feat = int(hst._feature[tr, node])
        used = float(hst._midpoints[tr, node])
        out.append((feat, used, 0.5 * (lo[feat] + hi[feat])))
        hi_l = hi.copy()
        hi_l[feat] = used
        lo_r = lo.copy()
        lo_r[feat] = used
        queue.append((2 * node + 1, lo.copy(), hi_l))
        queue.append((2 * node + 2, lo_r, hi.copy()))
    return out


@pytest.mark.parametrize("d", [1, 2, 4])
def test_hst_split_is_midpoint_of_node_range(d):
    """Every split value is the mid-point of its node's range on its feature.

    Rule of Tan et al. 2011, Section 3.2 ("the split point is the
    mid-point of the current range of q") and Ulrich 2025, Section 5.2
    ("s = (x + y) / 2"). The earlier code used ``(2k+1) / 2^(level+1)``
    from the BFS position of the node, which is the mid-point only for
    ``d = 1``.
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
        for feat, used, expected in _walk_node_ranges(hst, 0):
            assert abs(used - expected) < 1e-12, (
                f"d={d} seed={seed} feat={feat}: split {used} != {expected}"
            )


@pytest.mark.parametrize("lo,hi", [(0.0, 1.0), (-1.0, 1.0), (-0.1, 0.1), (10.0, 20.0)])
def test_hst_root_split_is_the_middle_of_the_working_range(lo, hi):
    """The root splits the working range in two halves, whatever its units.

    The split values are compared with the normalised sample, so the
    root split of a single feature is 0.5 for every working range.
    """
    hst = HalfSpaceTrees(
        n_trees=3, depth=4, window=20, feature_ranges=[(lo, hi)], seed=0
    )
    hst.learn_one({"a": 0.5 * (lo + hi)})
    assert np.allclose(hst._midpoints[:, 0], 0.5)
    left = [
        x
        for x in np.linspace(lo, hi, 20)
        if hst._to_norm({"a": float(x)})[0] <= hst._midpoints[0, 0]
    ]
    assert len(left) == 10


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
