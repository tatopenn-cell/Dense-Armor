"""Tests for ThompsonSampling, Beta-Bernoulli and Gaussian."""

import numpy as np
import pytest

from dense_armor.utility.bandit import ThompsonSampling


def test_ts_bernoulli_finds_best_arm():
    rng = np.random.default_rng(0)
    p = ThompsonSampling(arms=[0, 1, 2], family="bernoulli", seed=0)
    for _ in range(600):
        a = p.pull()
        r = float(rng.binomial(1, [0.2, 0.8, 0.5][a]))
        _ = p.update(a, r)
    assert p.counts_[1] > p.counts_[0]
    assert p.counts_[1] > p.counts_[2]


def test_ts_bernoulli_prior_is_beta_1_1():
    p = ThompsonSampling(arms=[0, 1], family="bernoulli", seed=0)
    samples = [p._draw(0) for _ in range(2000)]
    assert 0.0 <= min(samples) <= 1.0
    assert 0.0 <= max(samples) <= 1.0
    assert abs(np.mean(samples) - 0.5) < 0.05


def test_ts_gaussian_finds_best_arm():
    rng = np.random.default_rng(1)
    p = ThompsonSampling(arms=[0, 1], family="gaussian", sigma=0.5, seed=0)
    for _ in range(400):
        a = p.pull()
        r = float(rng.normal([0.0, 2.0][a], 0.5))
        _ = p.update(a, r)
    assert p.counts_[1] > p.counts_[0]


def test_ts_rejects_bad_family():
    with pytest.raises(ValueError):
        ThompsonSampling(arms=[0, 1], family="poisson")


def test_ts_rejects_bad_sigma():
    with pytest.raises(ValueError):
        ThompsonSampling(arms=[0, 1], family="gaussian", sigma=0.0)


def test_ts_nan_counted():
    p = ThompsonSampling(arms=[0, 1], family="bernoulli", seed=0)
    _ = p.update(0, float("nan"))
    assert p.n_missing == 1


def test_ts_unknown_arm_counted():
    p = ThompsonSampling(arms=[0, 1], family="bernoulli", seed=0)
    _ = p.update(99, 1.0)
    assert p.n_missing == 1


def test_ts_clone():
    p = ThompsonSampling(arms=[0, 1], family="bernoulli", seed=5)
    c = p.clone()
    assert type(c) is ThompsonSampling
    assert c.family == "bernoulli"
    assert c.seed == 5
