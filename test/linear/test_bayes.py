"""Tests for the conjugate Gaussian linear model."""

import doctest

import jax.numpy as jnp
import numpy as np

from dense_armor.roles.signal import Signal
from dense_armor.utility.linear import bayes as bayes_mod
from dense_armor.utility.linear.bayes import BayesianLinearRegression


def test_doctests():
    r = doctest.testmod(bayes_mod)
    assert r.failed == 0


def test_streaming_matches_batch():
    rng = np.random.default_rng(0)
    n, d = 8, 3
    X = rng.normal(size=(n, d))
    w_true = np.array([1.0, -2.0, 0.5])
    y = X @ w_true + 0.1 * rng.normal(size=n)
    alpha, beta = 2.0, 4.0

    model = BayesianLinearRegression(
        alpha=alpha, beta=beta, intercept=False, feature_keys=["a", "b", "c"]
    )
    for i in range(n):
        model.learn_one({"a": X[i, 0], "b": X[i, 1], "c": X[i, 2]}, y[i])

    S_batch = np.linalg.inv(alpha * np.eye(d) + beta * X.T @ X)
    m_batch = beta * S_batch @ X.T @ y

    assert float(np.max(np.abs(model._S - S_batch))) < 1e-12
    assert float(np.max(np.abs(model._m - m_batch))) < 1e-12


def test_predictive_variance_grows_away_from_data():
    model = BayesianLinearRegression(alpha=1.0, beta=1.0)
    for i in range(20):
        model.learn_one({"x": float(i) / 10.0}, 2.0 * float(i) / 10.0)
    near = model.predict_one({"x": 0.5}, return_estimate=True).var
    far = model.predict_one({"x": 10.0}, return_estimate=True).var
    assert far > 5.0 * near
    assert near >= 1.0 / model.beta - 1e-12


def test_variance_non_negative():
    model = BayesianLinearRegression(alpha=1.0, beta=1.0)
    for i in range(10):
        model.learn_one({"x": float(i)}, float(i))
    est = model.predict_one({"x": 0.0}, return_estimate=True)
    assert est.var >= 0.0


def test_signal_matches_dict():
    model = BayesianLinearRegression(alpha=1.0, beta=1.0, intercept=False)
    for i in range(10):
        model.learn_one({"x": float(i) / 10.0}, float(i) / 5.0)
    pd = model.predict_one({"x": 0.5})
    sig = Signal(values=jnp.array([0.5]), names=["x"], units=[""])
    ps = model.predict_one(sig)
    assert abs(float(pd) - float(ps)) < 1e-12
