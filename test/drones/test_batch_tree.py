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
