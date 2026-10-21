"""Hand cases of the forests of the machine learning package."""

import doctest

import numpy as np

import dense_armor.ml.forest as F
from dense_armor.ml.forest import ExtraTreesRegressor, RandomForestClassifier

X = np.array(
    [[i / 10, i % 2] for i in range(6)] + [[1.0 + i / 10, i % 2] for i in range(6)]
)
Y = np.array([0] * 6 + [1] * 6)


def test_doctests():
    r = doctest.testmod(F)
    assert r.failed == 0 and r.attempted >= 6


def test_random_forest_hand_case():
    m = RandomForestClassifier(n_trees=12, seed=4).fit(X, Y)
    p = m.predict_proba(X)
    assert np.allclose(p.sum(axis=1), 1.0) and len(m.trees_) == 12
    assert m.predict([[0.2, 0.5], [1.3, 0.5]]).tolist() == [0, 1]
    again = RandomForestClassifier(n_trees=12, seed=4).fit(X, Y)
    assert np.array_equal(again.predict_proba(X), p)
    one = RandomForestClassifier(n_trees=5, max_features=1, max_depth=2, seed=1).fit(
        X, Y
    )
    assert all(t.depth_ <= 2 for t in one.trees_)
    skew = RandomForestClassifier(n_trees=6, seed=0).fit(X, np.array([0] * 11 + [1]))
    assert skew.predict_proba(X).shape == (12, 2)


def test_extra_trees_hand_case():
    t = np.where(X[:, 0] < 0.8, 0.0, 4.0)
    m = ExtraTreesRegressor(n_trees=15, seed=2).fit(X, t)
    pred = m.predict(X)
    assert pred.min() >= 0.0 and pred.max() <= 4.0
    assert m.predict([[0.1, 0.0]])[0] < 2.0 < m.predict([[1.3, 0.0]])[0]
    assert np.array_equal(
        ExtraTreesRegressor(n_trees=15, seed=2).fit(X, t).predict(X), pred
    )


def test_stopped_classes_are_declared():
    assert set(F.STOPPED) == {
        "AdaptiveRandomForestClassifier",
        "AggregatedMondrianForestClassifier",
        "AggregatedMondrianForestRegressor",
    }
