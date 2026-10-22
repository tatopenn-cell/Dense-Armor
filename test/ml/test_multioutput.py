"""Hand cases of the multi-target models of the machine learning package."""

import doctest

import numpy as np

import dense_armor.ml.multioutput as MO
from dense_armor.ml.linear import LinearRegression
from dense_armor.ml.multioutput import (
    ClassifierChain,
    PerOutputClassifier,
    PerOutputRegressor,
    ProbabilisticClassifierChain,
    RegressorChain,
)

X = np.array([[0.0], [1.0], [2.0], [3.0]])
Y = np.array([[0, 1], [0, 1], [1, 0], [1, 0]])


class Rec:
    """A classifier that records the width of its input and answers by a rule."""

    made: list = []

    def __init__(self, rule):
        self.rule = rule
        Rec.made.append(self)

    def fit(self, X, y=None):
        self.fit_width = X.shape[1]
        return self

    def predict_proba(self, X):
        self.pred_width = X.shape[1]
        return np.array([self.rule(r) for r in X])


def rules(*rs):
    Rec.made = []
    it = iter(rs)
    return lambda: Rec(next(it))


def test_doctests():
    r = doctest.testmod(MO)
    assert r.failed == 0 and r.attempted >= 20


def test_per_output_hand_case():
    m = PerOutputClassifier(rules(lambda r: 0.9, lambda r: 0.2)).fit(X, Y)
    assert m.predict(X[:2]).tolist() == [[1, 0], [1, 0]]
    assert [e.fit_width for e in Rec.made] == [1, 1]
    r = PerOutputRegressor(LinearRegression).fit(
        X, np.c_[X[:, 0] * 2, X[:, 0] * -1 + 1]
    )
    assert np.allclose(r.predict([[4.0]]), [[8.0, -3.0]])


def test_chain_sees_the_earlier_targets():
    m = ClassifierChain(
        rules(lambda r: 0.8, lambda r: 0.9 if r[-1] >= 0.5 else 0.1), order=[0, 1]
    ).fit(X, Y)
    assert [e.fit_width for e in Rec.made] == [1, 2]
    assert m.predict(X[:1]).tolist() == [[1, 1]]
    assert [e.pred_width for e in Rec.made] == [1, 2]
    rev = ClassifierChain(rules(lambda r: 0.8, lambda r: 0.1), order=[1, 0]).fit(X, Y)
    assert rev.order_ == [1, 0] and rev.predict_proba(X[:1]).tolist() == [[0.1, 0.8]]
    rnd = ClassifierChain(rules(lambda r: 0.5, lambda r: 0.5), seed=1).fit(X, Y)
    assert sorted(rnd.order_) == [0, 1]


def test_regressor_chain_hand_case():
    Yr = np.c_[2.0 * X[:, 0] + 1.0, 4.0 * X[:, 0] + 2.0 + 0.5 * X[:, 0] ** 2]
    m = RegressorChain(lambda: LinearRegression(l2=1e-9), order=[0, 1]).fit(X, Yr)
    assert np.allclose(m.predict([[1.0]])[0, 0], 3.0, atol=1e-6)
    assert m.predict([[1.0]]).shape == (1, 2)


def test_probabilistic_chain_hand_case():
    m = ProbabilisticClassifierChain(
        rules(lambda r: 0.8, lambda r: 0.9 if r[-1] >= 0.5 else 0.3), order=[0, 1]
    ).fit(X, Y)
    combos, joint = m.joint_proba(X[:1])
    got = {tuple(c): j for c, j in zip(combos.tolist(), joint[0])}
    assert np.allclose(
        [got[(1, 1)], got[(1, 0)], got[(0, 1)], got[(0, 0)]], [0.72, 0.08, 0.06, 0.14]
    )
    assert abs(joint.sum() - 1.0) < 1e-12 and m.predict(X[:1]).tolist() == [[1, 1]]
