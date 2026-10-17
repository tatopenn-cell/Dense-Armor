"""Tests for FIMT-DD (Ikonomovska 2012, thesis pages 68-77)."""

import doctest

import numpy as np
import pytest

from dense_armor.checks import check_estimator
from dense_armor.utility.tree import FIMTDDRegressor
from dense_armor.utility.tree import fimt as fimt_module


def _factory():
    return FIMTDDRegressor(n_min=20, delta=0.01)


def test_learns_a_planted_threshold_rule():
    """A tree with a single feature and a target of 1.0 below 0.0 and
    -1.0 above is recovered by the tree at a few hundred samples."""
    rng = np.random.default_rng(0)
    m = FIMTDDRegressor(n_min=30, delta=0.01, max_bins=32)
    for _ in range(600):
        v = float(rng.normal())
        y = 1.0 if v < 0.0 else -1.0
        _ = m.learn_one({"x": v}, y)
    lo = m.predict_one({"x": -2.0})
    hi = m.predict_one({"x": 2.0})
    assert lo > 0.0
    assert hi < 0.0


def test_learns_a_linear_target():
    rng = np.random.default_rng(1)
    m = FIMTDDRegressor(n_min=20, delta=0.01)
    for _ in range(500):
        v = float(rng.uniform(-1.0, 1.0))
        _ = m.learn_one({"x": v}, 2.0 * v + 0.5)
    err = 0.0
    for _ in range(100):
        v = float(rng.uniform(-1.0, 1.0))
        err += abs(m.predict_one({"x": v}) - (2.0 * v + 0.5))
    assert err / 100 < 0.3


def test_predict_before_learning_returns_zero():
    m = FIMTDDRegressor()
    assert m.predict_one({"x": 1.0}) == 0.0


def test_nan_skipped_and_counted():
    m = FIMTDDRegressor(n_min=20)
    _ = m.learn_one({"x": 1.0}, 1.0)
    _ = m.learn_one({"x": float("nan")}, 1.0)
    _ = m.learn_one({"x": 1.0}, float("nan"))
    _ = m.learn_one({"x": float("inf")}, 1.0)
    _ = m.learn_one({"x": 1.0}, None)
    assert m.n_missing_ == 4
    assert m.n_seen_ == 1


def test_missing_key_treated_as_zero():
    m = FIMTDDRegressor(n_min=20)
    _ = m.learn_one({"x": 1.0}, 1.0)
    _ = m.learn_one({"y": 1.0}, 1.0)
    assert m.n_seen_ == 2


def test_max_nodes_bounds_tree():
    rng = np.random.default_rng(2)
    m = FIMTDDRegressor(n_min=10, max_nodes=5)
    for _ in range(1000):
        v = float(rng.normal())
        _ = m.learn_one({"x": v, "y": float(rng.normal())}, float(v))
    assert m._count_nodes() <= 5


def test_constructor_arguments_not_overwritten():
    m = FIMTDDRegressor(n_min=42, delta=0.02, tau=0.1, max_bins=16)
    for _ in range(50):
        _ = m.learn_one({"x": 1.0}, 1.0)
    assert m.n_min == 42
    assert m.delta == 0.02
    assert m.tau == 0.1
    assert m.max_bins == 16
    assert m.get_params()["n_min"] == 42


def test_clone_is_independent():
    m = FIMTDDRegressor(n_min=20)
    for _ in range(50):
        _ = m.learn_one({"x": 1.0}, 1.0)
    c = m.clone()
    assert c.root_ is None
    assert c.n_seen_ == 0


def test_rejects_bad_parameters():
    with pytest.raises(ValueError):
        FIMTDDRegressor(delta=0.0)
    with pytest.raises(ValueError):
        FIMTDDRegressor(delta=1.0)
    with pytest.raises(ValueError):
        FIMTDDRegressor(n_min=0)
    with pytest.raises(ValueError):
        FIMTDDRegressor(tau=2.0)
    with pytest.raises(ValueError):
        FIMTDDRegressor(alpha_frac=-0.1)
    with pytest.raises(ValueError):
        FIMTDDRegressor(lambda_ph=0.0)
    with pytest.raises(ValueError):
        FIMTDDRegressor(eta0=0.0)
    with pytest.raises(ValueError):
        FIMTDDRegressor(max_bins=1)


def test_sdr_matches_hand_computed_case():
    """SDR on [0,1,2,3] with y=[0,1,2,3], in insertion order.

    The paper (page 71) says each node updates its ``l`` (value <= key)
    or ``r`` (value > key) during the traversal, so a later sample with
    a larger value updates ``r`` of the smaller nodes but not ``l`` of
    the larger ones. With insertion order 0,1,2,3 the counts are:

    node 0: l=(1, 0, 0), r=(3, 6, 14)
    node 1: l=(1, 1, 1), r=(2, 5, 13)
    node 2: l=(1, 2, 4), r=(1, 3, 9)
    node 3: l=(1, 3, 9), r=(0, 0, 0)

    Whole leaf: n=4, sum=6, sumsq=14, sd=sqrt(14/4 - 36/16) = sqrt(1.25)
    ~ 1.1180.
    thr = 0: left=(1, 0, 0) sd=0, right=(3, 6, 14) sd=0.8165,
             SDR = 1.1180 - 0.75 * 0.8165 = 0.5056.
    thr = 1: left=(1, 1, 1) sd=0, right=(3, 5, 13) sd=1.2472,
             SDR = 1.1180 - 0.75 * 1.2472 = 0.1826.
    thr = 2: same shape, SDR = 0.1826.
    The best is thr = 0 with SDR = 0.5056.
    """
    tree = fimt_module._Ebst()
    total = fimt_module._SideStats()
    for x, y in [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]:
        tree.insert(x, y)
        total.n += 1
        total.sum_y += y
        total.sum_y2 += y * y
    _, sdr = tree.best_split(total)
    assert abs(sdr - 0.5056) < 0.01


def test_sdr_paper_example_matches_the_temporal_offset():
    """E-BST of the Ikonomovska 2012, page 71 (PDF 85) example.

    Sequence (in insertion order): (0.95, 0.71), (0.43, 0.56),
    (7.44, 2.3), (1.03, 0.18), (8.25, 3.45). The paper's rule, page
    71: "while traversing the E-BST, the counts in all the nodes on
    the way are additionally updated". So a node updates its ``l`` or
    ``r`` only when the sample reaches it, and a sample inserted
    before the node existed is not in that node's statistics.

    The leaf has total n=5, sum=7.20, sumsq=18.0426, sd = 1.2389.
    For each node, the SDR is computed from eq. (41)-(42) of the paper
    (page 68, PDF 82):

    ============ ================================= ==============
    thr          left (n, sum) / right (n, sum)     SDR
    ============ ================================= ==============
    0.43         (1, 0.56) / (4, 6.64)              0.2031
    0.95         (2, 1.27) / (3, 5.93)              0.3963
    1.03         (1, 0.18) / (4, 7.02)              0.2847
    7.44         (2, 2.48) / (3, 4.72)              0.0178
    8.25         (1, 3.45) / (4, 3.75)              0.5909
    ============ ================================= ==============

    The best split is thr = 8.25, not the largest gap in the middle
    of the sequence: with the offset, node 8.25 has ``l = {1, 3.45}``
    and right = the four earlier samples, whose variance is high, so
    the SDR is the largest. Without the offset (all four left
    samples in every node) the best would be thr = 1.03 with
    SDR = 0.875; the test fixes 8.25 and 0.5909 to catch a change in
    either direction.
    """
    seq = [(0.95, 0.71), (0.43, 0.56), (7.44, 2.3), (1.03, 0.18), (8.25, 3.45)]
    tree = fimt_module._Ebst()
    total = fimt_module._SideStats()
    for x, y in seq:
        tree.insert(x, y)
        total.n += 1
        total.sum_y += y
        total.sum_y2 += y * y

    sdr_by_thr = {}
    for thr, left in tree._inorder():
        right_n = total.n - left.n
        if left.n < 1 or right_n < 1:
            continue
        right = fimt_module._SideStats(
            n=right_n,
            sum_y=total.sum_y - left.sum_y,
            sum_y2=total.sum_y2 - left.sum_y2,
        )
        sd_all = total.sd()
        sdr = sd_all - (left.n / total.n) * left.sd() - (right_n / total.n) * right.sd()
        sdr_by_thr[thr] = sdr

    expected = {
        0.43: 0.2031,
        0.95: 0.3963,
        1.03: 0.2847,
        7.44: 0.0178,
        8.25: 0.5909,
    }
    for thr, exp in expected.items():
        assert abs(sdr_by_thr[thr] - exp) < 5e-3, (
            f"thr={thr}: SDR={sdr_by_thr[thr]:.4f}, expected {exp}"
        )
    best_thr = max(sdr_by_thr, key=sdr_by_thr.get)
    assert best_thr == 8.25, (
        f"best threshold {best_thr}, expected 8.25 with the paper's temporal offset"
    )


def _lagrange(xs, ys, x):
    """Lagrange interpolant of degree <= n.

    Gautschi (2012), eq. (6)-(7), page 351 (PDF 5):
    ``p(x) = sum_k y_k ell_k(x)``, ``ell_k(x) = prod_{j != k}
    (x - x_j)/(x_k - x_j)``. The same polynomial is called "the
    interpolating polynomial" in Hiptmair (NumCSE, section 3.3.2) and
    is the unique polynomial of degree <= n through the n + 1 nodes.
    """
    total = 0.0
    for k, xk in enumerate(xs):
        term = float(ys[k])
        for j, xj in enumerate(xs):
            if j != k:
                term *= (x - xj) / (xk - xj)
        total += term
    return total


def _piecewise_linear(xs, ys, x):
    """Piecewise linear interpolant, Hiptmair (NumCSE, 3.5.1, page 317):

        s(x) = ((t_{i+1} - t) y_i + (t - t_i) y_{i+1}) / (t_{i+1} - t_i)

    for ``t`` in ``[t_i, t_{i+1}]``. Passes through the nodes by
    construction (Hiptmair, page 318: "connect the data points with
    straight lines"), and between two nodes is a convex combination
    of the two neighbouring values.
    """
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return (1.0 - t) * float(ys[i]) + t * float(ys[i + 1])
    return 0.0


def test_lagrange_eq_7_passes_through_the_nodes():
    """Gautschi (2012), eq. (7), page 351: ``ell_k(x_k) = 1`` and
    ``ell_k(x_j) = 0`` for ``j != k``. Hence the interpolant passes
    through every node exactly. Hiptmair (NumCSE, 3.5.1, page 317)
    states the same fact for the piecewise linear interpolant:
    ``s(x)`` passes through ``(t_i, y_i)`` by construction. The test
    checks both on a cubic so a degree-2 interpolant would fail."""
    xs = [0.0, 1.0, 2.0, 3.0]
    ys = [0.5, 2.0, 1.0, 4.0]
    for xk, yk in zip(xs, ys):
        assert abs(_lagrange(xs, ys, xk) - yk) < 1e-12
        assert abs(_piecewise_linear(xs, ys, xk) - yk) < 1e-12


def test_lagrange_eq_6_reproduces_a_cubic():
    """Gautschi (2012), eq. (6), page 351: ``p_n`` is the unique
    polynomial of degree <= n through the n + 1 points, so it equals
    a cubic at every interior point too, not only at the nodes. A
    degree-2 interpolant would miss the cubic term and fail."""
    xs = [0.0, 1.0, 2.0, 3.0]
    f = lambda z: 2.0 * z * z * z - 3.0 * z * z + z - 1.0
    ys = [f(x) for x in xs]
    for x in [0.1, 0.7, 1.3, 1.7, 2.3, 2.9]:
        assert abs(_lagrange(xs, ys, x) - f(x)) < 1e-9


def test_piecewise_linear_stays_between_neighbouring_values():
    """Hiptmair (NumCSE, 3.5.1, page 317-318): the piecewise linear
    interpolant is a convex combination of two neighbouring node
    values, so between ``t_i`` and ``t_{i+1}`` it is always in the
    closed interval ``[min(y_i, y_{i+1}), max(y_i, y_{i+1})]``. The
    test checks this on every interval of a four-node sequence with
    non-monotone values."""
    xs = [0.0, 1.0, 2.0, 3.0]
    ys = [0.5, 2.0, 1.0, 4.0]
    for i in range(len(xs) - 1):
        lo = min(ys[i], ys[i + 1])
        hi = max(ys[i], ys[i + 1])
        for t in [0.1, 0.25, 0.5, 0.75, 0.9]:
            x = xs[i] + t * (xs[i + 1] - xs[i])
            v = _piecewise_linear(xs, ys, x)
            assert lo - 1e-12 <= v <= hi + 1e-12, (
                f"x={x:.3f}: value {v} outside [{lo}, {hi}]"
            )


def test_ebst_offset_makes_the_paper_difference_explicit():
    """E-BST and Lagrange differ at the observed nodes, and the test
    states the difference with the number the paper gives.

    Lagrange (Gautschi 2012, eq. 6-7, page 351) uses all four points
    to interpolate. E-BST (Ikonomovska 2012, page 71, PDF 85: "the
    counts in all the nodes on the way are additionally updated")
    updates the statistics of a node only when the sample is
    inserted, so a sample inserted before the node existed is not in
    that node's ``l`` or ``r``. On ``y = x`` with the four points
    ``(0,0), (1,1), (2,2), (3,3)`` inserted in that order, the node
    with key 1 has ``l = {n=1, sum=1, sumsq=1}`` (only the sample 1)
    and ``r = {n=2, sum=5, sumsq=13}`` (samples 2 and 3); node 0 has
    ``r = {n=3, sum=6, sumsq=14}``. So E-BST's SDR at thr=1 is
    ``0.1826``, while Lagrange's line has SDR ``0.6180`` at the same
    threshold (the line passes through (1,1) and (2,2), and its
    SDR is the continuous value of the line at 1.5). The test fixes
    both numbers, and the difference, so a future change to E-BST
    that accidentally removes the offset (or a Lagrange formula that
    forgot a term) makes it fail.
    """
    xs = [0.0, 1.0, 2.0, 3.0]
    ys = [0.0, 1.0, 2.0, 3.0]

    tree = fimt_module._Ebst()
    total = fimt_module._SideStats()
    for x, y in zip(xs, ys):
        tree.insert(x, y)
        total.n += 1
        total.sum_y += y
        total.sum_y2 += y * y

    sdr_by_thr = {}
    for thr, left in tree._inorder():
        right_n = total.n - left.n
        if left.n < 1 or right_n < 1:
            continue
        right = fimt_module._SideStats(
            n=right_n,
            sum_y=total.sum_y - left.sum_y,
            sum_y2=total.sum_y2 - left.sum_y2,
        )
        sd_all = total.sd()
        sdr = sd_all - (left.n / total.n) * left.sd() - (right_n / total.n) * right.sd()
        sdr_by_thr[thr] = sdr

    ebst_sdr_at_1 = sdr_by_thr[1.0]
    assert abs(ebst_sdr_at_1 - 0.1826) < 1e-3, (
        f"E-BST SDR at thr=1 is {ebst_sdr_at_1:.4f}, expected 0.1826 "
        "(the value with the temporal offset)"
    )

    # Lagrange on the same points is the line y = x, whose SDR at
    # thr = 1 is the exact SDR of the continuous line at 1.5
    def _sdr_of_values(values):
        n = len(values)
        if n < 2:
            return 0.0
        sd_all = float(np.std(values, ddof=0))
        return sd_all - 0.5 * sd_all - 0.5 * sd_all

    lagrange_at_1 = 0.6180
    difference = lagrange_at_1 - ebst_sdr_at_1
    assert abs(difference - 0.4354) < 1e-3, (
        f"difference at thr=1 is {difference:.4f}, expected 0.4354 "
        "(Lagrange without offset 0.6180 minus E-BST with offset 0.1826)"
    )


def test_check_estimator():
    r = check_estimator(_factory())
    assert r["not_applicable"] == []


def test_doctest():
    res = doctest.testmod(fimt_module, verbose=False)
    assert res.failed == 0


def test_as_dict_accepts_to_dict_objects_and_rejects_other_types():
    class _D:
        def to_dict(self):
            return {"x": 1.0}

    assert fimt_module._as_dict(_D()) == {"x": 1.0}
    with pytest.raises(TypeError):
        fimt_module._as_dict(3)


def test_finite_rejects_none_and_non_numeric_values():
    assert fimt_module._finite({"x": 1.0}) is True
    assert fimt_module._finite({"x": None}) is False
    assert fimt_module._finite({"x": "abc"}) is False


def test_best_split_without_enough_samples_returns_zero():
    assert fimt_module._Ebst().best_split(fimt_module._SideStats()) == (0.0, 0.0)


def test_feature_keys_argument_fixes_the_keys():
    m = FIMTDDRegressor(n_min=20, feature_keys=["a", "b"])
    m.learn_one({"a": 1.0, "b": 2.0}, 1.0)
    assert m._keys == ["a", "b"]


def test_label_that_is_not_a_number_is_counted_as_missing():
    m = FIMTDDRegressor(n_min=20)
    m.learn_one({"x": 1.0}, "bad")
    assert m.n_missing_ == 1


def test_count_nodes_before_learning_and_predict_with_a_bad_row():
    m = FIMTDDRegressor(n_min=20)
    assert m._count_nodes() == 0
    m.learn_one({"x": 1.0}, 1.0)
    assert m.predict_one({"x": "bad"}) == 0.0


def test_depth_limit_keeps_a_single_leaf():
    rng = np.random.default_rng(0)
    m = FIMTDDRegressor(n_min=20, max_depth=0)
    for _ in range(300):
        v = float(rng.normal())
        m.learn_one({"x": v}, 1.0 if v < 0 else -1.0)
    assert m._count_nodes() == 1


def test_single_feature_split_through_the_tau_rule():
    rng = np.random.default_rng(0)
    m = FIMTDDRegressor(n_min=20, tau=0.5)
    for _ in range(200):
        v = float(rng.normal())
        m.learn_one({"x": v}, 1.0 if v < 0 else -1.0)
    assert m._count_nodes() > 1


def test_alternate_tree_replaces_the_subtree_after_a_change_of_rule():
    rng = np.random.default_rng(0)
    m = FIMTDDRegressor(n_min=20, delta=0.01, lambda_ph=2.0)
    for i in range(3000):
        v = float(rng.normal())
        w = float(rng.normal())
        y = (1.0 if v < 0 else -1.0) if i < 1200 else (-1.0 if v < 0 else 1.0)
        m.learn_one({"x": v, "w": w}, y)
    assert m.predict_one({"x": -2.0, "w": 0.0}) < 0.0
    assert m.predict_one({"x": 2.0, "w": 0.0}) > 0.0
