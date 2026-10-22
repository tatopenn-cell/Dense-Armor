"""Hand cases of the naive Bayes classifiers of the machine learning package."""

import doctest

import numpy as np
import pytest

import dense_armor.ml.bayes as B
from dense_armor.ml.bayes import BernoulliNB, CategoricalNB, ComplementNB, MultinomialNB

COUNTS = [[3, 0], [2, 1], [0, 3], [1, 2]]
Y = [0, 0, 1, 1]


def test_doctests():
    r = doctest.testmod(B)
    assert r.failed == 0 and r.attempted >= 8


def test_multinomial_hand_case():
    m = MultinomialNB().fit(COUNTS, Y)
    assert np.allclose(np.exp(m.feature_log_prob_), [[0.75, 0.25], [0.25, 0.75]])
    assert abs(m.predict_proba([[4, 0]])[0, 0] - 81.0 / 82.0) < 1e-12
    skew = MultinomialNB(class_prior={0: 3.0, 1: 1.0}).fit(COUNTS, Y)
    assert abs(skew.predict_proba([[1, 1]])[0, 0] - 0.75) < 1e-12


def test_bernoulli_hand_case():
    m = BernoulliNB().fit([[1, 0], [1, 1], [0, 1], [0, 1]], Y)
    assert abs(m.predict_proba([[1, 0]])[0, 0] - 6.0 / 7.0) < 1e-12
    assert m.predict([[0, 1]]).tolist() == [1]


def test_categorical_hand_case():
    m = CategoricalNB().fit([[0, 1], [0, 0], [1, 1], [1, 0]], Y)
    assert abs(m.predict_proba([[0, 1]])[0, 0] - 0.75) < 1e-12
    assert abs(m.predict_proba([[5, 1]])[0, 0] - 0.5) < 1e-12
    assert m.predict([[1, 0]]).tolist() == [1]
    assert m.classes_.tolist() == [0, 1]


def test_complement_hand_case():
    m = ComplementNB().fit(COUNTS, Y)
    s = abs(np.log(0.25)) + abs(np.log(0.75))
    assert np.allclose(
        m.weights_,
        [[np.log(0.25) / s, np.log(0.75) / s], [np.log(0.75) / s, np.log(0.25) / s]],
    )
    assert m.predict([[4, 0], [0, 4]]).tolist() == [0, 1]
    raw = ComplementNB(norm=False).fit(COUNTS, Y)
    assert np.allclose(raw.weights_[0], [np.log(0.25), np.log(0.75)])
    assert abs(m.predict_proba([[4, 0]]).sum() - 1.0) < 1e-12


def test_the_base_needs_scores():
    with pytest.raises(NotImplementedError):
        B._NB()._scores(np.zeros((1, 1)))
