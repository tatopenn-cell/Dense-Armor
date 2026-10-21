"""Hand cases of the multiclass strategies of the machine learning package."""

import doctest

import numpy as np

import dense_armor.ml.multiclass as M
from dense_armor.ml.multiclass import (
    OneVsOneClassifier,
    OneVsRestClassifier,
    OutputCodeClassifier,
)

X = np.array([[0.0, 0.0], [0.0, 1.0], [5.0, 0.0], [5.0, 1.0], [0.0, 5.0], [1.0, 5.0]])
Y = np.array([0, 0, 1, 1, 2, 2])


class Stub:
    """A binary classifier that always answers the probability ``p``."""

    def __init__(self, p):
        self.p = p

    def fit(self, X, y=None):
        self.y_ = np.asarray(y)
        return self

    def predict_proba(self, X):
        return np.full(len(X), self.p)


def stubs(*ps):
    it = iter(ps)
    return lambda: Stub(next(it))


def test_doctests():
    r = doctest.testmod(M)
    assert r.failed == 0 and r.attempted >= 10


def test_one_vs_rest_hand_case():
    m = OneVsRestClassifier(stubs(0.2, 0.9, 0.4)).fit(X, Y)
    assert np.allclose(m.predict_proba(X[:1]), [[0.2 / 1.5, 0.9 / 1.5, 0.4 / 1.5]])
    assert m.predict(X[:2]).tolist() == [1, 1]
    assert [e.y_.tolist() for e in m.estimators_][0] == [1, 1, 0, 0, 0, 0]


def test_one_vs_one_hand_case():
    m = OneVsOneClassifier(stubs(0.2, 0.9, 0.9)).fit(X, Y)
    assert m.pairs_ == [(0, 1), (0, 2), (1, 2)]
    assert m.votes(X[:1]).tolist() == [[1.0, 0.0, 2.0]]
    assert m.predict(X[:1]).tolist() == [2]
    assert m.estimators_[1].y_.tolist() == [0, 0, 1, 1]


def test_output_code_hand_case():
    m = OutputCodeClassifier(stubs(*([1.0] * 6)), code_size=6, seed=3).fit(X, Y)
    assert m.codes_.shape == (3, 6) and set(np.unique(m.codes_)) == {-1, 1}
    assert all(len(set(m.codes_[:, b])) == 2 for b in range(6))
    dist = (m.codes_ != 1).sum(axis=1)
    assert m.predict(X[:1]).tolist() == [int(np.argmin(dist))]
    again = OutputCodeClassifier(stubs(*([1.0] * 6)), code_size=6, seed=3).fit(X, Y)
    assert np.array_equal(again.codes_, m.codes_)
