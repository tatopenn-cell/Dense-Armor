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
    """The far point scores above 0.5 (uniform mass), the centre below it."""
    hst = HalfSpaceTrees(n_trees=15, depth=8, window=100, range_init=10, seed=0)
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
    assert s_in < 0.5 < s_out


def _walk_node_ranges(hst, tr):
    """Return (feature, split value used, expected mid-point) per internal node.

    The walk starts from the work space of the tree (``_work_lo``,
    ``_work_hi``, in the normalised space of the detector), rebuilds the
    range of every node, and descends with the split values the detector
    stores in ``_midpoints``. The expected value is the middle of the
    node's range on its split feature.
    """
    assert hst._feature is not None and hst._midpoints is not None
    assert hst._work_lo is not None and hst._work_hi is not None
    n_internal = 2**hst.depth - 1
    lo_root = hst._work_lo[tr].copy()
    hi_root = hst._work_hi[tr].copy()
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


@pytest.mark.parametrize("d", [1, 3])
def test_hst_work_space_follows_the_paper(d):
    """Work space of Tan et al. 2011, page 3: ``s_q +- 2 * max(s_q, 1 - s_q)``.

    ``s_q`` is uniform in ``[0, 1]``, one value per tree and dimension.
    """
    hst = HalfSpaceTrees(
        n_trees=20,
        depth=3,
        window=10,
        feature_ranges=[(0.0, 1.0)] * d,
        seed=3,
    )
    hst.learn_one({chr(ord("a") + i): 0.5 for i in range(d)})
    centre = 0.5 * (hst._work_lo + hst._work_hi)
    half = 0.5 * (hst._work_hi - hst._work_lo)
    assert np.all((centre >= 0.0) & (centre <= 1.0))
    assert np.allclose(half, 2.0 * np.maximum(centre, 1.0 - centre), atol=1e-12)
    assert len(np.unique(centre)) > 1


@pytest.mark.parametrize("lo,hi", [(-1.0, 1.0), (-0.1, 0.1), (10.0, 20.0)])
def test_hst_split_values_do_not_depend_on_the_units(lo, hi):
    """The split values live in the normalised space: units do not change them."""
    kw = {"n_trees": 3, "depth": 5, "window": 20, "seed": 0}
    ref = HalfSpaceTrees(feature_ranges=[(0.0, 1.0)] * 2, **kw)
    other = HalfSpaceTrees(feature_ranges=[(lo, hi)] * 2, **kw)
    ref.learn_one({"a": 0.5, "b": 0.5})
    other.learn_one({"a": 0.5 * (lo + hi), "b": 0.5 * (lo + hi)})
    assert np.array_equal(ref._midpoints, other._midpoints)


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
        HalfSpaceTrees(size_limit_frac=0.0)
    with pytest.raises(ValueError):
        HalfSpaceTrees(size_limit_frac=1.5)
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


def test_hst_memory_does_not_grow():
    hst = HalfSpaceTrees(n_trees=3, depth=5, window=10, range_init=5, seed=0)
    for i in range(30):
        hst.learn_one({"a": float(i % 10) / 10.0})
    shape = (hst._r.shape, hst._l.shape)
    for i in range(500):
        hst.learn_one({"a": float(i % 10) / 10.0})
    assert (hst._r.shape, hst._l.shape) == shape
    assert hst._count < hst.window


def test_hst_scores_zero_until_the_reference_window_is_full():
    """The ranges take ``range_init`` samples, the reference ``window`` more."""
    hst = HalfSpaceTrees(n_trees=3, depth=4, window=10, range_init=5, seed=0)
    x = {"a": 0.5, "b": 0.5}
    scores = []
    for i in range(16):
        scores.append(hst.score_one(x))
        hst.learn_one({"a": 0.1 * (i % 5), "b": 0.2})
    assert scores[:15] == [0.0] * 15
    assert scores[15] > 0.0


def test_hst_profiles_follow_algorithm_3():
    """Tan et al. 2011, Algorithm 3: ``r`` is the reference, ``l`` the latest.

    The first ``window`` samples fill ``r``; afterwards each sample goes to
    ``l`` and at the end of every window ``r`` takes the value of ``l`` and
    ``l`` is reset.
    """
    hst = HalfSpaceTrees(
        n_trees=2, depth=3, window=8, feature_ranges=[(0.0, 1.0)], seed=0
    )

    def learn(n):
        for _ in range(n):
            hst.learn_one({"a": 0.3})

    learn(8)
    assert np.all(hst._r[:, 0] == 8) and np.all(hst._l[:, 0] == 0)
    learn(3)
    assert np.all(hst._r[:, 0] == 8) and np.all(hst._l[:, 0] == 3)
    learn(5)
    assert np.all(hst._r[:, 0] == 8) and np.all(hst._l[:, 0] == 0)
    learn(2)
    assert np.all(hst._r[:, 0] == 8) and np.all(hst._l[:, 0] == 2)


def test_hst_score_matches_a_hand_computation():
    """``r * 2^k`` of the terminal node, ``M / (M + S)`` with ``M = n_trees * window``.

    One tree of depth 2 on one feature with split values 0.5 (root), 0.25
    and 0.75, window 4 and ``sizeLimit = 1``; the four reference samples are
    0.1, 0.1, 0.2 and 0.9.
    """
    hst = HalfSpaceTrees(
        n_trees=1,
        depth=2,
        window=4,
        feature_ranges=[(0.0, 1.0)],
        size_limit_frac=0.25,
        seed=0,
    )
    hst._lazy_init({"a": 0.0})
    hst._midpoints[0, :] = [0.5, 0.25, 0.75]
    for v in (0.1, 0.1, 0.2, 0.9):
        hst.learn_one({"a": v})
    assert list(hst._r[0, :3]) == [4, 3, 1]
    assert hst.score_one({"a": 0.15}) == pytest.approx(4.0 / (4.0 + 3 * 2**2))
    assert hst.score_one({"a": 0.9}) == pytest.approx(4.0 / (4.0 + 1 * 2**1))
    assert hst.score_one({"a": 0.4}) == pytest.approx(1.0)


def test_hst_check_estimator():
    check_estimator(HalfSpaceTrees(n_trees=3, depth=5, window=20, range_init=5, seed=0))
