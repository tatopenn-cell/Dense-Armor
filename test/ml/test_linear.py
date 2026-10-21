"""Hand cases and edge cases of the linear models of the machine learning package."""

import doctest

import numpy as np

import dense_armor.ml.linear as L
from dense_armor.ml.linear import (
    BayesianLinearRegression,
    LinearRegression,
    LogisticRegression,
)


def test_doctests():
    r = doctest.testmod(L)
    assert r.failed == 0 and r.attempted >= 5


def test_ridge_hand_case():
    m = LinearRegression(l2=2.0, intercept=False).fit([[1.0], [-1.0]], [1.0, -1.0])
    assert abs(m.coef_[0] - 0.5) < 1e-12 and m.intercept_ == 0.0
    assert m.predict([[2.0]]).tolist() == [1.0]
    line = LinearRegression().fit([0.0, 1.0, 2.0], [1.0, 3.0, 5.0])
    assert np.allclose(line.predict([3.0]), [7.0])


def test_logistic_stationarity():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [1.5]])
    y = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    m = LogisticRegression(l2=0.1).fit(X, y)
    A = np.hstack([np.ones((5, 1)), X])
    w = np.r_[m.intercept_, m.coef_]
    p = 1 / (1 + np.exp(-A @ w))
    grad = A.T @ (p - y) / 5 + 0.1 * w
    assert np.abs(grad).max() < 1e-8 and m.n_iter_ < 100
    assert m.predict_proba([[3.0]])[0] > 0.5 and m.predict([[0.0]]).tolist() == [0]
    free = LogisticRegression(intercept=False, max_iter=3).fit(
        [[1.0], [-1.0], [2.0]], [1, 0, 1]
    )
    assert free.intercept_ == 0.0


def test_bayesian_posterior_hand_case():
    m = BayesianLinearRegression(alpha=2.0, beta=3.0, intercept=False).fit(
        [[1.0], [2.0]], [1.0, 2.0]
    )
    s = 1.0 / (2.0 + 3.0 * 5.0)
    assert abs(m.cov_[0, 0] - s) < 1e-12 and abs(m.mean_[0] - 3.0 * s * 5.0) < 1e-12
    mu, sd = m.predict([[1.0]], return_std=True)
    assert abs(mu[0] - m.mean_[0]) < 1e-12 and abs(sd[0] - np.sqrt(1 / 3.0 + s)) < 1e-12
    assert m.predict([[1.0]]).shape == (1,)
    ridge = LinearRegression(l2=2.0 * 2.0 / (3.0 * 4), intercept=True).fit(
        [[0.0], [1.0], [2.0], [3.0]], [1.0, 2.0, 2.0, 5.0]
    )
    bayes = BayesianLinearRegression(alpha=2.0, beta=3.0).fit(
        [[0.0], [1.0], [2.0], [3.0]], [1.0, 2.0, 2.0, 5.0]
    )
    assert np.allclose(bayes.mean_, np.r_[ridge.intercept_, ridge.coef_])


def test_payload_is_declared_stopped():
    assert "PayloadEstimator" in L.STOPPED
