"""Hand cases of the ensembles of the machine learning package."""

import doctest

import numpy as np
import pytest

import dense_armor.ml.ensemble as E
from dense_armor.ml.base import MLBase
from dense_armor.ml.ensemble import (
    BaggingClassifier,
    BoostingClassifier,
    ExponentiallyWeightedAverage,
    StackingClassifier,
    VotingClassifier,
)
from dense_armor.ml.neighbors import KNNClassifier
from dense_armor.ml.tree import DecisionTreeClassifier

X = np.array([[0.0], [1.0], [2.0], [3.0]])
Y = np.array([0, 0, 1, 1])


class Fixed(MLBase):
    """A model that ignores the data and always answers ``labels`` (and ``probs``)."""

    def __init__(self, labels=(0, 0, 1, 0), probs=None):
        self.labels = labels
        self.probs = probs

    def fit(self, X, y=None, sample_weight=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        return np.array(self.labels)[: len(X)]

    def predict_proba(self, X):
        p = np.array(self.probs if self.probs is not None else [[1.0, 0.0]] * len(X))
        return p[: len(X)]


def test_doctests():
    r = doctest.testmod(E)
    assert r.failed == 0 and r.attempted >= 20


def test_bagging_hand_case():
    m = BaggingClassifier(Fixed(probs=[[0.2, 0.8]] * 4), n_models=3, seed=0).fit(X, Y)
    assert np.allclose(m.predict_proba(X), [[0.2, 0.8]] * 4) and m.predict(
        X
    ).tolist() == [1, 1, 1, 1]
    real = BaggingClassifier(DecisionTreeClassifier(), n_models=5, seed=1).fit(X, Y)
    assert abs(real.predict_proba(X).sum() - 4.0) < 1e-12


def test_boosting_samme_hand_case():
    m = BoostingClassifier(Fixed(labels=(0, 0, 1, 0)), n_models=4).fit(X, Y)
    assert len(m.models_) == 1 and abs(m.alphas_[0] - np.log(3.0)) < 1e-12
    bad = BoostingClassifier(Fixed(labels=(0, 0, 0, 0)), n_models=3).fit(X, Y)
    assert bad.alphas_ == [1.0] and len(bad.models_) == 1
    good = BoostingClassifier(Fixed(labels=(0, 0, 1, 1)), n_models=3).fit(X, Y)
    assert len(good.models_) == 1 and good.predict(X).tolist() == [0, 0, 1, 1]
    assert np.allclose(good.predict_proba(X).sum(axis=1), 1.0)
    stumps = BoostingClassifier(DecisionTreeClassifier(max_depth=1), n_models=6).fit(
        np.arange(8.0)[:, None], [0, 0, 1, 1, 1, 1, 0, 0]
    )
    assert len(stumps.models_) >= 2 and stumps.predict(
        np.arange(8.0)[:, None]
    ).tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    one = BoostingClassifier(Fixed(labels=(1, 1)), n_models=2).fit(
        [[0.0], [1.0]], [1, 1]
    )
    assert one.alphas_[0] == 1.0


def test_exponentially_weighted_average_hand_case():
    P = [[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]]
    m = ExponentiallyWeightedAverage(eta=1.0).fit(P, [1.0, 2.0, 3.0])
    assert m.weights_.round(6).tolist() == [1.0, 0.0]
    two = ExponentiallyWeightedAverage(eta=1.0).fit([[0.0, 0.0]] * 1, [1.0])
    assert np.allclose(two.weights_, [0.5, 0.5])
    third = ExponentiallyWeightedAverage(eta=1.0).fit(
        [[0.0, 1.0 - np.sqrt(2.0)]], [1.0]
    )
    assert abs(third.weights_[0] - 1.0 / (1.0 + np.exp(-1.0))) < 1e-9
    assert np.allclose(m.predict([[2.0, 9.0]]), [2.0])


def test_voting_hand_case():
    a = Fixed(labels=(0, 1), probs=[[0.9, 0.1], [0.4, 0.6]])
    b = Fixed(labels=(0, 0), probs=[[0.6, 0.4], [0.9, 0.1]])
    hard = VotingClassifier([a, b], mode="hard", weights=[1.0, 2.0]).fit(X[:2], [0, 1])
    assert hard.predict(X[:2]).tolist() == [0, 0]
    soft = VotingClassifier([a, b], mode="soft").fit(X[:2], [0, 1])
    assert np.allclose(soft.predict_proba(X[:2]), [[0.75, 0.25], [0.65, 0.35]])
    assert soft.predict(X[:2]).tolist() == [0, 0]
    with pytest.raises(ValueError):
        VotingClassifier([a], mode="loud")


def test_stacking_hand_case():
    Xs = np.array([[0.0], [0.1], [0.2], [0.3], [1.0], [1.1], [1.2], [1.3]])
    Ys = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    m = StackingClassifier(
        [KNNClassifier(n_neighbors=1)], DecisionTreeClassifier(), cv=2
    ).fit(Xs, Ys)
    assert m.meta_.tree_ is not None and m.predict([[0.15], [1.15]]).tolist() == [0, 1]
    assert np.allclose(m.predict_proba([[0.15]]).sum(), 1.0)
