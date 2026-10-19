"""Tests for dense_armor/utility/tree/batch.py (CART)."""

import numpy as np
import pytest

from dense_armor.checks import check_estimator
from dense_armor.utility.tree.batch import BatchDecisionTreeClassifier


def test_simple_split():
    X = [{"x": 0.0}, {"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
    y = [0, 0, 1, 1]
    clf = BatchDecisionTreeClassifier(max_depth=2).fit(X, y)
    assert clf.predict_one({"x": 0.5}) == 0
    assert clf.predict_one({"x": 2.5}) == 1


def test_proba_sums_to_one():
    X = [{"x": 0.0}, {"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
    y = [0, 0, 1, 1]
    clf = BatchDecisionTreeClassifier(max_depth=2).fit(X, y)
    p = clf.predict_proba_one({"x": 2.5})
    assert abs(sum(p.values()) - 1.0) < 1e-9


def test_multiclass():
    rng = np.random.default_rng(0)
    X, y = [], []
    for _ in range(300):
        c = int(rng.integers(0, 3))
        X.append({"x": float(c + rng.normal(0, 0.1))})
        y.append(c)
    clf = BatchDecisionTreeClassifier(max_depth=5).fit(X, y)
    for c in range(3):
        assert clf.predict_one({"x": float(c)}) == c


def test_pure_node_is_leaf():
    X = [{"x": 0.0}, {"x": 1.0}]
    y = [1, 1]
    clf = BatchDecisionTreeClassifier().fit(X, y)
    assert clf.predict_one({"x": 0.5}) == 1


def test_min_samples_split():
    X = [{"x": 0.0}, {"x": 0.1}, {"x": 10.0}, {"x": 10.1}]
    y = [0, 0, 1, 1]
    clf = BatchDecisionTreeClassifier(min_samples_split=10).fit(X, y)
    p = clf.predict_proba_one({"x": 0.5})
    assert abs(p.get(0, 0) - 0.5) < 1e-9


def test_max_depth_zero_is_leaf():
    X = [{"x": 0.0}, {"x": 1.0}]
    y = [0, 1]
    clf = BatchDecisionTreeClassifier(max_depth=0).fit(X, y)
    assert clf.predict_one({"x": 0.5}) in (0, 1)


def test_predict_before_fit_returns_none():
    clf = BatchDecisionTreeClassifier()
    assert clf.predict_one({"x": 0.5}) is None
    assert clf.predict_proba_one({"x": 0.5}) == {}


def test_learn_one_is_noop():
    X = [{"x": 0.0}, {"x": 1.0}]
    y = [0, 1]
    clf = BatchDecisionTreeClassifier(max_depth=1).fit(X, y)
    before = clf.predict_one({"x": 0.5})
    clf.learn_one({"x": 0.5}, 0)
    clf.learn_one({"x": 0.5}, 1)
    assert clf.predict_one({"x": 0.5}) == before


def test_empty_fit_raises():
    with pytest.raises(ValueError):
        BatchDecisionTreeClassifier().fit([], [])


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        BatchDecisionTreeClassifier().fit([{"x": 0.0}], [0, 1])


def test_bad_depth_raises():
    with pytest.raises(ValueError):
        BatchDecisionTreeClassifier(max_depth=-1)


def test_bad_min_samples_raises():
    with pytest.raises(ValueError):
        BatchDecisionTreeClassifier(min_samples_split=1)


def test_classification_on_iris_like_data():
    rng = np.random.default_rng(0)
    centres = np.array([[0.0, 0.0], [3.0, 3.0]])
    X, y = [], []
    for c in range(2):
        for _ in range(100):
            p = centres[c] + rng.normal(0, 0.3, 2)
            X.append({"a": float(p[0]), "b": float(p[1])})
            y.append(c)
    clf = BatchDecisionTreeClassifier(max_depth=4).fit(X, y)
    correct = sum(clf.predict_one(x) == t for x, t in zip(X, y))
    assert correct / len(y) > 0.9


def test_passes_check_estimator():
    out = check_estimator(BatchDecisionTreeClassifier(max_depth=3))
    assert out["not_applicable"] == []
    assert len(out["skipped"]) == 2


def test_clone_independent():
    X = [{"x": 0.0}, {"x": 1.0}]
    y = [0, 1]
    a = BatchDecisionTreeClassifier(max_depth=2).fit(X, y)
    b = a.clone()
    assert b.predict_one({"x": 0.5}) is None  # not fitted


def test_fit_without_labels_raises():
    with pytest.raises(ValueError):
        BatchDecisionTreeClassifier().fit([{"a": 1.0}])


def test_fit_accepts_the_root_arguments():
    X = [{"a": 0.0}, {"a": 1.0}, {"a": 0.1}, {"a": 0.9}]
    y = [0, 1, 0, 1]
    clf = BatchDecisionTreeClassifier(max_depth=2).fit(
        X, y, t=None, epochs=1, shuffle=False, seed=3
    )
    assert clf.predict_one({"a": 0.05}) == 0


def test_gini_empty_returns_zero():
    from dense_armor.utility.tree.batch import _gini
    assert _gini({}) == 0.0


def test_majority_empty_returns_none():
    from dense_armor.utility.tree.batch import _majority
    assert _majority({}) is None


def test_constant_feature_is_skipped():
    """A constant feature column is skipped (no split on it)."""
    from dense_armor.utility.tree.batch import BatchDecisionTreeClassifier
    X = [{"a": 1.0, "b": 0.0}, {"a": 1.0, "b": 1.0},
         {"a": 1.0, "b": 2.0}, {"a": 1.0, "b": 3.0}]
    y = [0, 0, 1, 1]
    clf = BatchDecisionTreeClassifier(max_depth=3).fit(X, y)
    # the tree splits on b, not on the constant a
    assert clf._root is not None
    assert clf._root.feature == "b"


def test_no_split_returns_leaf():
    """Two classes, no feature separates them: the node stays a leaf."""
    from dense_armor.utility.tree.batch import BatchDecisionTreeClassifier
    X = [{"a": 1.0}, {"a": 1.0}]
    y = [0, 1]
    clf = BatchDecisionTreeClassifier(max_depth=3).fit(X, y)
    p = clf.predict_proba_one({"a": 1.0})
    assert abs(p.get(0, 0) - 0.5) < 1e-9


def test_predict_missing_feature_returns_leaf():
    """A missing feature key returns the current node (no crash)."""
    from dense_armor.utility.tree.batch import BatchDecisionTreeClassifier
    X = [{"a": 0.0}, {"a": 1.0}, {"a": 2.0}, {"a": 3.0}]
    y = [0, 0, 1, 1]
    clf = BatchDecisionTreeClassifier(max_depth=2).fit(X, y)
    out = clf.predict_one({})
    assert out in (0, 1)


def test_predict_non_numeric_returns_leaf():
    """A non-numeric value returns the current node (no crash)."""
    from dense_armor.utility.tree.batch import BatchDecisionTreeClassifier
    X = [{"a": 0.0}, {"a": 1.0}, {"a": 2.0}, {"a": 3.0}]
    y = [0, 0, 1, 1]
    clf = BatchDecisionTreeClassifier(max_depth=2).fit(X, y)
    out = clf.predict_one({"a": "not a number"})
    assert out in (0, 1)


def test_proba_on_empty_leaf_returns_empty_dict():
    """A leaf with no class counts gives an empty probability dict."""
    from dense_armor.utility.tree.batch import (
        BatchDecisionTreeClassifier,
        _Node,
    )
    clf = BatchDecisionTreeClassifier(max_depth=2)
    clf._root = _Node(counts={})
    assert clf.predict_proba_one({"a": 1.0}) == {}
