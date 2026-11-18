"""Edge branches of the Hoeffding trees, the regressor and the splitters."""

import numpy as np
import pytest

from dense_armor.core.online.drift.adwin import ADWIN
from dense_armor.core.online.tree import splitters as sp
from dense_armor.core.online.tree.adaptive import HoeffdingAdaptiveTreeClassifier
from dense_armor.core.online.tree.efdt import HoeffdingAnytimeTreeClassifier
from dense_armor.core.online.tree.hoeffding import (
    HoeffdingTreeClassifier,
    _entropy,
    _Node,
)
from dense_armor.core.online.tree.hoeffding_regressor import HoeffdingTreeRegressor


def _split_tree(cls=HoeffdingTreeClassifier, **kw):
    rng = np.random.default_rng(0)
    h = cls(grace_period=50, **kw)
    for _ in range(1500):
        x = float(rng.random())
        h.learn_one({"x": x}, int(x > 0.5))
    assert not h.root_.is_leaf
    return h


def test_entropy_helpers():
    assert _entropy({"a": 2, "b": 2}) == pytest.approx(1.0)
    assert _entropy({}) == 0.0
    h = HoeffdingTreeClassifier()
    node = _Node(0)
    node.class_counts_ = {"a": 1, "b": 1}
    assert h._entropy(node) == pytest.approx(1.0)


def test_nominal_values_that_cannot_be_used_are_missing():
    h = HoeffdingTreeClassifier(nominal_attributes=["c"])
    h.learn_one({"c": [1, 2]}, 1)
    h.learn_one({"c": float("nan")}, 1)
    assert h.n_missing == 2 and h.n_seen_ == 0
    r = HoeffdingTreeRegressor(nominal_attributes=["c"])
    r.learn_one({"c": [1, 2]}, 1.0)
    r.learn_one({"c": float("nan")}, 1.0)
    assert r.n_missing == 2 and r.n_seen_ == 0


def test_a_bad_value_at_prediction_stops_at_the_node():
    h = _split_tree()
    assert h.predict_one({"x": "abc"}) in (0, 1)
    assert h.explain_one({"x": "abc"}) == []
    r = HoeffdingTreeRegressor(grace_period=50)
    rng = np.random.default_rng(0)
    for _ in range(1500):
        x = float(rng.random())
        r.learn_one({"x": x}, 5.0 if x > 0.5 else 1.0)
    assert not r.root_.is_leaf
    assert isinstance(r.predict_one({"x": "abc"}), float)
    assert r.explain_one({"x": "abc"}) == []
    assert r.explain_one({}) == []
    assert isinstance(r.predict_one({}), float)


def test_adaptive_tree_stops_on_a_missing_or_bad_value():
    h = _split_tree(HoeffdingAdaptiveTreeClassifier, drift_detector=ADWIN())
    assert h._path_of(h.root_, {"x": "abc"})[-1] is h.root_
    assert h._path_of(h.root_, {})[-1] is h.root_


def test_adaptive_grow_alt_without_an_alternate_does_nothing():
    h = _split_tree(HoeffdingAdaptiveTreeClassifier, drift_detector=ADWIN())
    h.root_.alt_ = None
    n = h.n_seen_
    h._grow_alt_on(h.root_, {"x": 0.3}, 0)
    assert h.n_seen_ == n


def test_anytime_helpers_on_a_leaf_and_without_a_split():
    h = HoeffdingAnytimeTreeClassifier(grace_period=10)
    for i in range(60):
        h.learn_one({"x": 1.0}, i % 2)
    assert h.root_.is_leaf
    assert h._current_gain(h.root_) == 0.0
    assert h._try_split(h.root_) is False


def test_no_memory_limit_makes_the_check_a_no_op():
    h = HoeffdingTreeClassifier(max_active_leaves=None)
    h.learn_one({"x": 1.0}, 0)
    h._enforce_memory()
    r = HoeffdingTreeRegressor(max_active_leaves=None)
    r.learn_one({"x": 1.0}, 1.0)
    r._enforce_memory()
    assert h.n_active_leaves_ == 1 and r.n_active_leaves_ == 1


def test_a_tree_without_classes_predicts_nothing():
    h = HoeffdingTreeClassifier()
    assert h._predict_proba_leaf(_Node(0), {"x": 1.0}) == {}
    assert h._predict_class_leaf(_Node(0), {"x": 1.0}) is None


def test_naive_bayes_skips_classes_with_few_samples_and_bad_values():
    h = HoeffdingTreeClassifier(grace_period=10_000)
    h.learn_one({"x": 0.0}, "rare")
    for v in np.linspace(1.0, 2.0, 20):
        h.learn_one({"x": float(v)}, "common")
    assert h.predict_one({"x": 1.5}) == "common"
    assert h.predict_one({"x": "abc"}) == "common"
    assert sum(h.predict_proba_one({"x": "abc"}).values()) == pytest.approx(1.0)
    assert sum(h.predict_proba_one({"x": 1.5}).values()) == pytest.approx(1.0)


def test_regressor_bad_arguments():
    for kw in (
        {"tau": -1.0},
        {"max_depth": 0},
        {"max_nodes": 0},
        {"max_active_leaves": 0},
        {"memory_period": 0},
        {"quantile_radius": 0.0},
        {"grace_period": 0},
    ):
        with pytest.raises(ValueError):
            HoeffdingTreeRegressor(**kw)


def test_regressor_depth_auto_nominal_and_std_of_one_sample():
    r = HoeffdingTreeRegressor(grace_period=50, nominal_attributes="auto")
    assert r.depth_ == 0
    r.learn_one({"k": "a", "x": "1.5"}, 2.0)
    assert r.predict_one({"k": "a", "x": 1.5}, return_std=True) == (2.0, 0.0)
    rng = np.random.default_rng(0)
    for _ in range(2000):
        k = "ab"[int(rng.integers(2))]
        r.learn_one({"k": k}, 1.0 if k == "a" else 9.0)
    assert r.depth_ >= 1
    assert r.root_.split_category_ is not None


def test_scan_skips_empty_sides():
    pts = [(1.0, {}), (2.0, {"a": 1}), (3.0, {"b": 1})]
    gain, thr = sp._scan("gini", pts)
    assert thr == pytest.approx(2.5)
    assert sp._exact_gain("gini", [], 1.0) is None


def test_target_helpers_on_degenerate_input():
    assert sp._variance(0, 0.0, 0.0) == 0.0
    assert sp._target_scan([]) == (-1.0, None)
    assert sp._target_scan([(1.0, [1, 1.0, 1.0])]) == (-1.0, None)
    assert sp._target_scan([(1.0, [0, 0.0, 0.0]), (2.0, [1, 1.0, 1.0])]) == (-1.0, None)
    empty = sp.TargetNumericObserver("ebst")
    assert empty.test_gain(1.0) is None
    assert empty.best_candidate() == (-1.0, None)


def test_ebst_repeated_value_updates_the_same_slot():
    obs = sp.TargetNumericObserver("ebst")
    for y in (1.0, 2.0, 3.0):
        obs.update(1.0, y)
    obs.update(2.0, 9.0)
    assert len(obs._vals) == 2 and obs._stats[0][0] == 3


def test_target_nominal_with_one_category_or_an_empty_one():
    obs = sp.TargetNominalObserver()
    obs.update("a", 1.0)
    obs.update("a", 2.0)
    assert obs.best_candidate() == (-1.0, None)
    assert obs.test_gain("a") is None
    obs["z"] = [0, 0.0, 0.0]
    assert obs.best_candidate() == (-1.0, None)
    assert obs.test_gain("z") is None


def test_leaf_posterior_ignores_a_value_that_is_not_a_number():
    h = HoeffdingTreeClassifier(grace_period=10_000)
    for v in np.linspace(0.0, 1.0, 10):
        h.learn_one({"x": float(v)}, "a")
        h.learn_one({"x": float(v) + 2.0}, "b")
    leaf = h.root_
    assert sum(h._predict_proba_leaf(leaf, {"x": "abc"}).values()) == pytest.approx(1.0)
    assert h._predict_class_leaf(leaf, {"x": "abc"}) in ("a", "b")


def test_leaf_posterior_with_a_categorical_feature():
    h = HoeffdingTreeClassifier(grace_period=10_000, nominal_attributes=["c"])
    for i in range(40):
        h.learn_one({"c": "r" if i % 2 else "g"}, i % 2)
    assert h.predict_one({"c": "r"}) == 1
    assert sum(h.predict_proba_one({"c": "g"}).values()) == pytest.approx(1.0)
