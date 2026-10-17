"""Tests for BayesUCB, quantile of the posterior as the index."""

import numpy as np
import pytest

from dense_armor.utility.bandit import BayesUCB


def test_bayes_ucb_bernoulli_finds_best_arm():
    rng = np.random.default_rng(0)
    p = BayesUCB(arms=[0, 1, 2], family="bernoulli", horizon=600, seed=0)
    for _ in range(600):
        a = p.pull()
        r = float(rng.binomial(1, [0.2, 0.8, 0.5][a]))
        _ = p.update(a, r)
    assert p.counts_[1] > p.counts_[0]
    assert p.counts_[1] > p.counts_[2]


def test_bayes_ucb_gaussian_finds_best_arm():
    rng = np.random.default_rng(1)
    p = BayesUCB(arms=[0, 1], family="gaussian", horizon=400, sigma=0.5, seed=0)
    for _ in range(400):
        a = p.pull()
        r = float(rng.normal([0.0, 2.0][a], 0.5))
        _ = p.update(a, r)
    assert p.counts_[1] > p.counts_[0]


def test_bayes_ucb_quantile_rises_with_t():
    """The level is 1 - 1/(t log(n)^c), so the quantile rises with t."""
    p = BayesUCB(arms=[0, 1], family="bernoulli", horizon=100, seed=0)
    for _ in range(20):
        _ = p.update(0, 1.0)
    q10 = p._quantile(0, 10)
    q100 = p._quantile(0, 100)
    assert q10 < q100
    assert q100 < 1.0


def test_bayes_ucb_rejects_bad_family():
    with pytest.raises(ValueError):
        BayesUCB(arms=[0, 1], family="poisson")


def test_bayes_ucb_rejects_bad_sigma():
    with pytest.raises(ValueError):
        BayesUCB(arms=[0, 1], family="gaussian", sigma=-1.0)


def test_bayes_ucb_nan_counted():
    p = BayesUCB(arms=[0, 1], family="bernoulli", seed=0)
    _ = p.update(0, float("nan"))
    assert p.n_missing == 1


def test_bayes_ucb_clone():
    p = BayesUCB(arms=[0, 1], family="bernoulli", horizon=100, c=5.0, seed=3)
    c = p.clone()
    assert type(c) is BayesUCB
    assert c.horizon == 100
    assert c.c == 5.0
    assert c.seed == 3


def test_bayes_ucb_student_t_posterior():
    rng = np.random.default_rng(0)
    p = BayesUCB(
        arms=[0, 1],
        family="gaussian",
        estimate_var=True,
        sigma=1.0,
        horizon=200,
        seed=0,
    )
    for _ in range(200):
        a = p.pull()
        r = float(rng.normal([0.0, 2.0][a], 1.0))
        _ = p.update(a, r)
    assert p.counts_[1] > p.counts_[0]
