"""Edge paths of the tree package: bad parameters, bad inputs, small trees."""

import math

import numpy as np
import pytest

from dense_armor.utility.drift.adwin import ADWIN
from dense_armor.utility.tree import (
    HoeffdingAdaptiveTreeClassifier,
    HoeffdingAnytimeTreeClassifier,
    HoeffdingTreeClassifier,
    MondrianForestClassifier,
    MondrianForestRegressor,
    SGTClassifier,
    SGTRegressor,
)
from dense_armor.utility.tree.sgt import _betainc, _student_t_sf

BAD_X = [{}, {"a": None}, {"a": "x"}, {"a": float("nan")}, None]


def _hat(**kw):
    return HoeffdingAdaptiveTreeClassifier(drift_detector=ADWIN(), **kw)


@pytest.mark.parametrize(
    "make, kw",
    [
        (HoeffdingTreeClassifier, {"grace_period": 0}),
        (HoeffdingTreeClassifier, {"delta": 1.0}),
        (HoeffdingTreeClassifier, {"tau": -1.0}),
        (HoeffdingTreeClassifier, {"max_depth": 0}),
        (HoeffdingTreeClassifier, {"max_nodes": 0}),
        (HoeffdingTreeClassifier, {"leaf": "x"}),
        (HoeffdingAdaptiveTreeClassifier, {}),
        (_hat, {"delta_alt": 1.0}),
        (_hat, {"kappa_alt": 0}),
        (_hat, {"error_alpha": 0.0}),
        (SGTRegressor, {"delta": 0.0}),
        (SGTRegressor, {"min_samples_split": 1}),
        (SGTRegressor, {"grace_period": 0}),
        (SGTRegressor, {"n_bins": 1}),
        (SGTRegressor, {"bin_samples": 1}),
        (SGTRegressor, {"lambda_": -1.0}),
        (SGTRegressor, {"gamma": -1.0}),
        (SGTRegressor, {"max_depth": 0}),
        (SGTRegressor, {"max_nodes": 0}),
        (MondrianForestRegressor, {"n_trees": 0}),
        (MondrianForestRegressor, {"lifetime": 0.0}),
        (MondrianForestRegressor, {"max_nodes": 0}),
    ],
)
def test_bad_parameters(make, kw):
    with pytest.raises(ValueError):
        make(**kw)


@pytest.mark.parametrize(
    "est",
    [
        HoeffdingTreeClassifier(grace_period=20, leaf="majority"),
        HoeffdingAnytimeTreeClassifier(grace_period=20),
        _hat(grace_period=20),
        SGTClassifier(min_samples_split=10, bin_samples=10, n_bins=8),
        MondrianForestClassifier(n_trees=2, seed=0),
    ],
)
def test_classifiers_bad_inputs_and_explain(est):
    assert est.predict_one({"a": 1.0}) in (None, 0, False)
    for x in BAD_X:
        est.learn_one(x, 1)
        est.predict_proba_one(x)
    est.learn_one({"a": 1.0}, None)
    assert est.n_missing >= len(BAD_X)
    rng = np.random.default_rng(0)
    for _ in range(400):
        y = int(rng.random() < 0.5)
        est.learn_one(
            {"a": float(rng.normal(3.0 * y, 0.3)), "b": float(rng.random())}, y
        )
    assert est.predict_one({"a": 3.0, "b": 0.5}) == 1
    assert est.predict_one({"a": 0.0, "b": 0.5}) == 0
    if hasattr(est, "explain_one"):
        path = est.explain_one({"a": 3.0, "b": 0.5})
        assert isinstance(path, list)
        assert est.explain_one({"a": "x"}) == [] or isinstance(
            est.explain_one({"a": "x"}), list
        )
        assert est.explain_one({}) == [] or isinstance(est.explain_one({}), list)


@pytest.mark.parametrize(
    "est",
    [
        SGTRegressor(min_samples_split=10, bin_samples=10, n_bins=8),
        MondrianForestRegressor(n_trees=2, seed=0),
    ],
)
def test_regressors_bad_inputs_and_std(est):
    assert est.predict_one({"a": 1.0}) == 0.0
    for x in BAD_X:
        est.learn_one(x, 1.0)
        assert est.predict_one(x) == 0.0
    for y in (None, "y", float("nan")):
        est.learn_one({"a": 1.0}, y)
    assert est.n_missing >= len(BAD_X) + 3
    rng = np.random.default_rng(1)
    for _ in range(400):
        a = float(rng.uniform(-1.0, 1.0))
        est.learn_one({"a": a}, 2.0 * a)
    m, s = est.predict_one({"a": 0.5}, return_std=True)
    assert abs(m - 1.0) < 0.6 and s >= 0.0
    if isinstance(est, MondrianForestRegressor):
        e = est.predict_one({"a": float("nan")}, return_estimate=True)
        assert e.var >= 0.0
    m0, s0 = est.predict_one({"a": float("nan")}, return_std=True)
    assert math.isfinite(m0) and s0 >= 0.0
    if hasattr(est, "explain_one"):
        assert isinstance(est.explain_one({"a": 0.5}), list)


def test_sgt_explain_before_learning_and_t_edges():
    assert SGTRegressor().explain_one({"a": 1.0}) == []
    assert _student_t_sf(1.0, 0) == 0.5
    assert _betainc(2.0, 3.0, 0.0) == 0.0
    assert _betainc(2.0, 3.0, 1.0) == 1.0
    assert math.isclose(_betainc(1.0, 1.0, 0.3), 0.3, rel_tol=1e-9)


def test_trees_respect_caps():
    rng = np.random.default_rng(2)
    for est in (
        HoeffdingTreeClassifier(grace_period=20, max_depth=1),
        HoeffdingAnytimeTreeClassifier(grace_period=20, max_nodes=3),
        _hat(grace_period=20, max_depth=1),
    ):
        for _ in range(2000):
            y = int(rng.random() < 0.5)
            x = {
                "a": float(rng.normal(3.0 * y, 0.3)),
                "b": float(rng.normal(3.0 * y, 0.3)),
            }
            est.learn_one(x, y)
        assert est.n_nodes_ <= 3
    f = MondrianForestRegressor(n_trees=1, seed=0, max_nodes=5)
    for i in range(200):
        f.learn_one({"a": float(i)}, float(i))
    assert f.trees_[0].n_nodes <= 5


def test_mondrian_finite_lifetime_and_missing_feature():
    rng = np.random.default_rng(3)
    f = MondrianForestRegressor(n_trees=2, seed=0, lifetime=0.5)
    for _ in range(300):
        a = float(rng.uniform(0.0, 1.0))
        f.learn_one({"a": a, "b": a}, a)
    assert all(t.n_nodes < 600 for t in f.trees_)
    m, s = f.predict_one({"c": 1.0}, return_std=True)
    assert math.isfinite(m) and s >= 0.0
    c = MondrianForestClassifier(n_trees=2, seed=0)
    for i in range(50):
        c.learn_one({"a": float(i)}, i % 2)
    assert set(c.predict_proba_one({"z": 1.0})) <= {0, 1}


def test_sgt_classifier_bad_labels_and_missing_feature():
    est = SGTClassifier(min_samples_split=10, bin_samples=10, n_bins=8)
    for y in ("y", float("nan")):
        est.learn_one({"a": 1.0}, y)
    assert est.n_missing == 2
    rng = np.random.default_rng(4)
    for _ in range(300):
        y = int(rng.random() < 0.5)
        est.learn_one({"a": float(rng.normal(3.0 * y, 0.3))}, y)
    assert set(est.predict_proba_one({"b": 1.0})) <= {0, 1}


def test_adaptive_detector_without_clone_is_copied():
    class Fixed:
        def __init__(self, level):
            self.level = level
            self.drift_detected = False

        def update(self, x, t=None):
            self.drift_detected = x > self.level
            return self

    h = HoeffdingAdaptiveTreeClassifier(grace_period=20, drift_detector=Fixed(2.0))
    rng = np.random.default_rng(5)
    for _ in range(200):
        y = int(rng.random() < 0.5)
        h.learn_one({"a": float(rng.normal(3.0 * y, 0.3))}, y)
    assert h.n_nodes_ >= 1


def test_efdt_collapses_a_split_that_no_longer_helps():
    ef = HoeffdingAnytimeTreeClassifier(grace_period=10)
    ef.learn_one({"a": 0.2}, 0)
    root = ef.root_
    assert ef._bound(root) == math.inf
    root.split_feature_, root.split_threshold_ = "a", 0.5
    root.left_, root.right_ = ef._new_leaf(1), ef._new_leaf(1)
    ef._re_evaluate(root)
    assert not root.is_leaf
    rng = np.random.default_rng(6)
    for _ in range(2000):
        ef._update_leaf(root, {"a": float(rng.random())}, int(rng.random() < 0.5))
    ef._re_evaluate(root)
    assert root.is_leaf and ef.n_collapsed_ == 1
