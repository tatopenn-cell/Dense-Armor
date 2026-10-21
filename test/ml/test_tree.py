"""Hand cases of the trees of the machine learning package."""

import doctest

import numpy as np

import dense_armor.ml.tree as T
from dense_armor.ml.tree import (
    BatchDecisionTreeClassifier,
    MondrianForestClassifier,
    MondrianForestRegressor,
)


def test_doctests():
    r = doctest.testmod(T)
    assert r.failed == 0 and r.attempted >= 6


def test_tree_hand_case():
    X = [[0.0, 5.0], [1.0, 4.0], [2.0, 5.0], [3.0, 4.0]]
    m = BatchDecisionTreeClassifier().fit(X, [0, 0, 1, 1])
    assert m.tree_["feature"] == 0 and m.tree_["threshold"] == 1.5 and m.depth_ == 1
    assert m.predict([[1.4, 0.0], [1.6, 0.0]]).tolist() == [0, 1]
    assert m.predict_proba([[0.0, 0.0]]).tolist() == [[1.0, 0.0]]


def test_tree_limits_and_ties():
    X = [[0.0], [1.0], [2.0], [3.0]]
    m = BatchDecisionTreeClassifier(max_depth=0).fit(X, [0, 1, 1, 0])
    assert m.depth_ == 0 and m.predict([[9.0]]).tolist() == [0]
    m = BatchDecisionTreeClassifier(min_samples_split=5).fit(X, [0, 0, 1, 1])
    assert m.depth_ == 0
    m = BatchDecisionTreeClassifier().fit([[1.0], [1.0]], [0, 1])
    assert m.depth_ == 0 and m.predict([[1.0]]).tolist() == [0]
    m = BatchDecisionTreeClassifier().fit([[0.0], [1.0], [2.0]], [0, 1, 0])
    assert m.depth_ == 2


def test_mondrian_classifier():
    X = np.array([[0.0], [0.1], [1.0], [1.1]])
    m = MondrianForestClassifier(n_trees=6, lifetime=0.0, seed=2).fit(X, [0, 0, 1, 1])
    assert all("feature" not in t for t in m.trees_)
    assert np.allclose(m.predict_proba([[0.5]]), [[0.5, 0.5]])
    m = MondrianForestClassifier(n_trees=6, seed=2).fit(X, [0, 0, 1, 1])
    p = m.predict_proba([[0.05], [1.05], [50.0]])
    assert np.allclose(p.sum(axis=1), 1.0) and m.predict([[0.0], [1.1]]).tolist() == [
        0,
        1,
    ]


def test_mondrian_regressor():
    X = np.array([[0.0], [0.1], [1.0], [1.1]])
    m = MondrianForestRegressor(n_trees=6, lifetime=0.0, seed=1).fit(
        X, [0.0, 0.0, 4.0, 4.0]
    )
    mu, sd = m.predict([[0.5]], return_std=True)
    assert (
        abs(mu[0] - 2.0) < 1e-12
        and abs(sd[0] - np.sqrt(np.var([0, 0, 4, 4], ddof=1))) < 1e-12
    )
    g = MondrianForestRegressor(n_trees=3, seed=1).fit([[1.0], [1.0]], [2.0, 2.0])
    assert abs(g.predict([[1.0]])[0] - 2.0) < 1e-12


def test_mondrian_split_at_the_upper_edge_is_a_leaf():
    class Edge:
        def exponential(self, scale):
            return 0.1

        def choice(self, n, p):
            return 0

        def uniform(self, lo, hi):
            return hi

    m = MondrianForestClassifier(lifetime=10.0)
    m.classes_ = np.array([0, 1])
    X, y = np.array([[0.0], [1.0]]), np.array([0, 1])
    node = m._block(X, y, 0.0, Edge())
    assert "feature" not in node and node["tau"] == 10.0 and node["n"] == 2


def test_stopped():
    assert len(T.STOPPED) == 6 and all(T.STOPPED.values())
