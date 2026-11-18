"""Hoeffding trees: criteria, splitters, categorical features, memory management."""

import pickle

import numpy as np
import pytest

from dense_armor.checks import check_estimator
from dense_armor.core.online.drift.adwin import ADWIN
from dense_armor.core.online.tree.adaptive import HoeffdingAdaptiveTreeClassifier
from dense_armor.core.online.tree.efdt import HoeffdingAnytimeTreeClassifier
from dense_armor.core.online.tree.hoeffding import HoeffdingTreeClassifier


def _threshold_stream(n, thr=1.0, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        x = float(rng.normal())
        yield {"x": x, "noise": float(rng.random())}, int(x > thr)


def _accuracy(tree, thr=1.0, n=1000):
    rng = np.random.default_rng(99)
    return np.mean(
        [
            tree.predict_one({"x": float(v), "noise": 0.5}) == int(v > thr)
            for v in rng.normal(size=n)
        ]
    )


@pytest.mark.parametrize("criterion", ["entropy", "gini"])
@pytest.mark.parametrize("splitter", ["class_means", "gaussian", "ebst", "quantile"])
def test_every_criterion_and_splitter_learns_a_threshold(criterion, splitter):
    h = HoeffdingTreeClassifier(
        grace_period=100, criterion=criterion, splitter=splitter
    )
    for x, y in _threshold_stream(4000):
        h.learn_one(x, y)
    assert not h.root_.is_leaf
    assert h.root_.split_feature_ == "x"
    assert _accuracy(h) > 0.97


def test_ebst_finds_the_exact_threshold():
    h = HoeffdingTreeClassifier(grace_period=100, splitter="ebst", criterion="gini")
    for x, y in _threshold_stream(4000):
        h.learn_one(x, y)
    assert abs(h.root_.split_threshold_ - 1.0) < 0.1


@pytest.mark.parametrize("splitter", ["gaussian", "quantile"])
def test_gaussian_and_quantile_thresholds_are_near_the_true_one(splitter):
    h = HoeffdingTreeClassifier(grace_period=100, splitter=splitter, criterion="gini")
    for x, y in _threshold_stream(4000):
        h.learn_one(x, y)
    assert abs(h.root_.split_threshold_ - 1.0) < 0.6


def test_quantile_radius_is_used():
    h = HoeffdingTreeClassifier(
        grace_period=100, splitter="quantile", quantile_radius=0.2
    )
    for x, y in _threshold_stream(2000):
        h.learn_one(x, y)
    obs = next(iter(h.root_.feature_stats_.values())) if h.root_.is_leaf else None
    if obs is not None:
        assert obs._radius == 0.2


def test_bad_arguments():
    with pytest.raises(ValueError):
        HoeffdingTreeClassifier(criterion="variance")
    with pytest.raises(ValueError):
        HoeffdingTreeClassifier(splitter="histogram")
    with pytest.raises(ValueError):
        HoeffdingTreeClassifier(n_splits=0)
    with pytest.raises(ValueError):
        HoeffdingTreeClassifier(max_active_leaves=0)
    with pytest.raises(ValueError):
        HoeffdingTreeClassifier(memory_period=0)
    with pytest.raises(ValueError):
        HoeffdingTreeClassifier(quantile_radius=0.0)
    with pytest.raises(ValueError):
        HoeffdingTreeClassifier(nominal_attributes="colour")


CATS = ["a", "b", "c", "d"]


def _cat_stream(n, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        c = CATS[int(rng.integers(4))]
        yield {"colour": c, "x": float(rng.random())}, int(c == "c")


@pytest.mark.parametrize("criterion", ["entropy", "gini"])
def test_categorical_feature_is_split_on_a_category(criterion):
    h = HoeffdingTreeClassifier(
        grace_period=60, criterion=criterion, nominal_attributes=["colour"]
    )
    for x, y in _cat_stream(1500):
        h.learn_one(x, y)
    assert h.root_.split_feature_ == "colour"
    assert h.root_.split_category_ == "c"
    assert [h.predict_one({"colour": c, "x": 0.3}) for c in CATS] == [0, 0, 1, 0]
    assert h.explain_one({"colour": "c", "x": 0.3}) == [
        {"feature": "colour", "category": "c", "side": "left"}
    ]


def test_unseen_category_goes_to_the_other_side_and_does_not_fail():
    h = HoeffdingTreeClassifier(grace_period=60, nominal_attributes=["colour"])
    for x, y in _cat_stream(1500):
        h.learn_one(x, y)
    assert h.predict_one({"colour": "never seen", "x": 0.3}) == 0
    assert sum(
        h.predict_proba_one({"colour": "never seen", "x": 0.3}).values()
    ) == pytest.approx(1.0)


def test_strings_are_missing_unless_the_feature_is_nominal():
    plain = HoeffdingTreeClassifier()
    plain.learn_one({"colour": "red"}, 1)
    assert plain.n_missing == 1 and plain.n_seen_ == 0
    declared = HoeffdingTreeClassifier(nominal_attributes=["colour"])
    declared.learn_one({"colour": "red"}, 1)
    assert declared.n_missing == 0 and declared.n_seen_ == 1


def test_auto_nominal_takes_non_numeric_strings_as_categories():
    h = HoeffdingTreeClassifier(grace_period=60, nominal_attributes="auto")
    for x, y in _cat_stream(1500):
        h.learn_one(x, y)
    assert h.root_.split_feature_ == "colour"
    h.learn_one({"colour": "a", "x": "0.4"}, 0)
    assert h.n_missing == 0


def test_numeric_value_for_a_categorical_feature_is_a_category():
    h = HoeffdingTreeClassifier(grace_period=40, nominal_attributes=["k"])
    rng = np.random.default_rng(0)
    for _ in range(1200):
        k = int(rng.integers(3))
        h.learn_one({"k": k}, int(k == 2))
    assert h.root_.split_category_ == 2
    assert h.predict_one({"k": 2}) == 1 and h.predict_one({"k": 0}) == 0


def test_mixed_numeric_and_categorical_features():
    rng = np.random.default_rng(3)
    h = HoeffdingTreeClassifier(
        grace_period=60, nominal_attributes=["colour"], splitter="ebst"
    )
    for _ in range(3000):
        c = CATS[int(rng.integers(4))]
        x = float(rng.normal())
        h.learn_one({"colour": c, "x": x}, int(c == "c" and x > 0.0))
    assert h.predict_one({"colour": "c", "x": 1.5}) == 1
    assert h.predict_one({"colour": "c", "x": -1.5}) == 0
    assert h.predict_one({"colour": "a", "x": 1.5}) == 0


def _staircase(cap, n=12000):
    rng = np.random.default_rng(0)
    h = HoeffdingTreeClassifier(
        grace_period=40, max_active_leaves=cap, memory_period=200
    )
    for _ in range(n):
        x = float(rng.random())
        h.learn_one({"x": x, "n": float(rng.random())}, int(x * 8))
    xs = np.random.default_rng(1).random(1500)
    acc = np.mean([h.predict_one({"x": float(v), "n": 0.5}) == int(v * 8) for v in xs])
    return h, acc


def test_deactivation_bounds_the_leaves_with_statistics_not_the_depth():
    free, acc_free = _staircase(None)
    capped, acc_capped = _staircase(3)
    assert free.n_active_leaves_ == len(list(free._leaves()))
    assert capped.n_active_leaves_ <= 3
    assert capped.depth_ >= free.depth_
    assert len(list(capped._leaves())) >= 8
    assert acc_capped > 0.95
    assert acc_free > 0.95


def test_a_deactivated_leaf_keeps_counting_and_predicts_the_majority():
    capped, _ = _staircase(3)
    dead = [lf for lf in capped._leaves() if not lf.active_]
    assert dead
    assert all(lf.feature_stats_ == {} for lf in dead)
    assert all(lf.n_ > 0 for lf in dead)


def test_inactive_leaf_does_not_split():
    h = HoeffdingTreeClassifier(grace_period=10)
    for x, y in _threshold_stream(50):
        h.learn_one(x, y)
    leaf = h.root_
    leaf.active_ = False
    assert h._try_split(leaf) is False


def test_max_nodes_and_max_depth_still_limit_the_tree():
    h = HoeffdingTreeClassifier(grace_period=40, max_nodes=5)
    rng = np.random.default_rng(0)
    for _ in range(6000):
        x = float(rng.random())
        h.learn_one({"x": x}, int(x * 8))
    assert h.n_nodes_ <= 5
    d = HoeffdingTreeClassifier(grace_period=40, max_depth=1)
    for _ in range(6000):
        x = float(rng.random())
        d.learn_one({"x": x}, int(x * 8))
    assert d.depth_ <= 1


@pytest.mark.parametrize(
    "make",
    [
        lambda: HoeffdingTreeClassifier(
            grace_period=60, criterion="gini", nominal_attributes=["colour"]
        ),
        lambda: HoeffdingAnytimeTreeClassifier(
            grace_period=60, criterion="gini", nominal_attributes=["colour"]
        ),
        lambda: HoeffdingAdaptiveTreeClassifier(
            grace_period=60,
            drift_detector=ADWIN(),
            criterion="gini",
            nominal_attributes=["colour"],
        ),
    ],
)
def test_the_derived_trees_take_the_new_options(make):
    h = make()
    for x, y in _cat_stream(1500):
        h.learn_one(x, y)
    assert h.predict_one({"colour": "c", "x": 0.3}) == 1
    assert h.predict_one({"colour": "a", "x": 0.3}) == 0


def test_anytime_tree_re_evaluates_a_categorical_split():
    h = HoeffdingAnytimeTreeClassifier(
        grace_period=40, nominal_attributes=["colour"], splitter="ebst"
    )
    for x, y in _cat_stream(1500):
        h.learn_one(x, y)
    assert h.n_reevals_ > 0
    assert h.predict_one({"colour": "c", "x": 0.3}) == 1


def test_pickle_and_clone_keep_the_new_options():
    h = HoeffdingTreeClassifier(
        grace_period=60,
        criterion="gini",
        splitter="ebst",
        nominal_attributes=["colour"],
    )
    for x, y in _cat_stream(800):
        h.learn_one(x, y)
    h2 = pickle.loads(pickle.dumps(h))
    assert h2.predict_one({"colour": "c", "x": 0.3}) == h.predict_one(
        {"colour": "c", "x": 0.3}
    )
    c = h.clone()
    assert (c.criterion, c.splitter, c.nominal_attributes) == (
        "gini",
        "ebst",
        ["colour"],
    )


@pytest.mark.parametrize("splitter", ["gaussian", "ebst", "quantile"])
def test_check_estimator_with_each_splitter(splitter):
    check_estimator(
        HoeffdingTreeClassifier(grace_period=20, splitter=splitter, criterion="gini")
    )
