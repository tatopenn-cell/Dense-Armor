"""Tests for the online GLM in dense_armor.utility.linear.glm."""

import doctest

import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.roles.signal import Signal
from dense_armor.utility.linear import glm as glm_mod
from dense_armor.utility.linear.glm import LinearRegression, LogisticRegression
from dense_armor.utility.optim import (
    SGD,
    AdaGrad,
    Adam,
    FTRLProximal,
    OnlineNewtonStep,
)


def test_doctests():
    r = doctest.testmod(glm_mod)
    assert r.failed == 0


def test_linear_converges():
    rng = np.random.default_rng(0)
    reg = LinearRegression(optimizer=SGD(lr=0.05), intercept=True)
    for _ in range(400):
        a, b = rng.normal(size=2)
        y = 1.5 * a - 2.5 * b + 0.7
        reg.learn_one({"a": float(a), "b": float(b)}, float(y))
    for _ in range(20):
        a, b = rng.normal(size=2)
        y = 1.5 * a - 2.5 * b + 0.7
        pred = reg.predict_one({"a": float(a), "b": float(b)})
        assert abs(pred - y) < 0.2


def test_logistic_separates():
    rng = np.random.default_rng(0)
    clf = LogisticRegression(optimizer=SGD(lr=0.5))
    for _ in range(200):
        a = rng.normal()
        b = rng.normal()
        y = 1 if a + b > 0 else 0
        clf.learn_one({"a": float(a), "b": float(b)}, y)
    correct = 0
    for _ in range(100):
        a = rng.normal()
        b = rng.normal()
        y = 1 if a + b > 0 else 0
        p = clf.predict_one({"a": float(a), "b": float(b)})
        correct += int(p == y)
    assert correct > 90


@pytest.mark.parametrize(
    "cls,opt",
    [
        (LinearRegression, SGD(lr=0.1)),
        (LinearRegression, AdaGrad(lr=0.5)),
        (LinearRegression, Adam(lr=0.05)),
        (LinearRegression, FTRLProximal(l2=0.1)),
        (LinearRegression, OnlineNewtonStep()),
        (LogisticRegression, SGD(lr=0.5)),
        (LogisticRegression, AdaGrad(lr=0.5)),
    ],
    ids=lambda o: getattr(o, "__name__", type(o).__name__),
)
def test_object_matches_pure(cls, opt):
    est = cls(optimizer=opt, intercept=False, feature_keys=["a", "b"])
    state = est.init_state(2)
    rng = np.random.default_rng(0)
    for _ in range(30):
        a, b = rng.normal(size=2)
        y = 0.7 * a - 0.3 * b
        est.learn_one({"a": float(a), "b": float(b)}, float(y))
        state, _ = est.step(
            state, {"a": jnp.asarray(a), "b": jnp.asarray(b), "y": jnp.asarray(y)}
        )
    err = float(np.max(np.abs(est._w - np.asarray(state[0]))))
    assert err < 1e-6


def test_signal_matches_dict():
    reg = LinearRegression(optimizer=SGD(lr=0.1), intercept=False)
    for i in range(10):
        reg.learn_one({"x": float(i) / 10.0}, float(i) / 5.0)
    pd = reg.predict_one({"x": 0.5})
    sig = Signal(values=jnp.array([0.5]), names=["x"], units=[""])
    ps = reg.predict_one(sig)
    assert abs(float(pd) - float(ps)) < 1e-12


def test_classifier_empty_before_learning():
    clf = LogisticRegression(optimizer=SGD(lr=0.1))
    assert clf.predict_proba_one({"x": 1.0}) == {}
    assert clf.predict_one({"x": 1.0}) is None


def test_regressor_zero_before_learning():
    reg = LinearRegression(optimizer=SGD(lr=0.1))
    assert reg.predict_one({"x": 1.0}) == 0.0
