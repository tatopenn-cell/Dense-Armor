"""Hand cases of the nearest-neighbour models of the machine learning package."""

import doctest

import numpy as np
import pytest

import dense_armor.ml.neighbors as N
from dense_armor.ml.neighbors import KNNClassifier, KNNRegressor


def test_doctests():
    r = doctest.testmod(N)
    assert r.failed == 0 and r.attempted >= 3


def test_classifier_hand_case():
    X, y = [[0.0], [1.0], [5.0], [6.0]], [0, 0, 1, 1]
    m = KNNClassifier(n_neighbors=3).fit(X, y)
    assert np.allclose(m.predict_proba([[2.0]]), [[2 / 3, 1 / 3]])
    assert m.predict([[2.0], [5.5]]).tolist() == [0, 1]
    d = KNNClassifier(n_neighbors=3, weights="distance", eps=0.0).fit(X, y)
    w0, w1 = 1 / 2.0 + 1 / 1.0, 1 / 3.0
    assert np.allclose(d.predict_proba([[2.0]]), [[w0 / (w0 + w1), w1 / (w0 + w1)]])
    assert KNNClassifier(n_neighbors=9).fit(X, y).predict_proba([[0.0]]).tolist() == [
        [0.5, 0.5]
    ]


def test_minkowski_order():
    X = [[0.0, 0.0], [3.0, 4.0], [4.0, 3.5]]
    near2 = KNNClassifier(n_neighbors=1, p=2.0).fit(X, [0, 1, 2]).predict([[3.5, 3.9]])
    near1 = KNNClassifier(n_neighbors=1, p=1.0).fit(X, [0, 1, 2]).predict([[3.5, 3.9]])
    assert near2.tolist() == [1] and near1.tolist() == [1]


def test_regressor_hand_case():
    X, y = [[0.0], [1.0], [9.0]], [0.0, 2.0, 10.0]
    assert KNNRegressor(n_neighbors=2).fit(X, y).predict([[0.4]]).tolist() == [1.0]
    d = (
        KNNRegressor(n_neighbors=2, weights="distance", eps=0.0)
        .fit(X, y)
        .predict([[0.25]])
    )
    assert np.allclose(d, [(4.0 * 0.0 + (1 / 0.75) * 2.0) / (4.0 + 1 / 0.75)])


def test_bad_weights():
    with pytest.raises(ValueError):
        KNNClassifier(weights="votes")
