"""Tests for Exp3."""

import numpy as np
import pytest

from dense_armor.utility.bandit import Exp3


def test_exp3_finds_best_arm_stochastic():
    rng = np.random.default_rng(0)
    p = Exp3(arms=[0, 1, 2], gamma=0.1, seed=0)
    for _ in range(2000):
        a = p.pull()
        r = float(rng.binomial(1, [0.2, 0.8, 0.5][a]))
        _ = p.update(a, r)
    assert p.weights_[1] > p.weights_[0]
    assert p.weights_[1] > p.weights_[2]


def test_exp3_handles_adversarial():
    p = Exp3(arms=[0, 1], gamma=0.1, seed=0)
    for _ in range(100):
        a = p.pull()
        _ = p.update(a, 1.0 if a == 0 else 0.0)
    for _ in range(100):
        a = p.pull()
        _ = p.update(a, 1.0 if a == 1 else 0.0)
    assert p.n_seen_ == 200


def test_exp3_rejects_bad_gamma():
    for g in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            Exp3(arms=[0, 1], gamma=g)


def test_exp3_default_gamma_needs_horizon():
    with pytest.raises(ValueError):
        Exp3(arms=[0, 1])


def test_exp3_default_gamma_from_horizon():
    p = Exp3(arms=[0, 1], horizon=1000)
    assert 0.0 < p.gamma <= 1.0


def test_exp3_nan_counted():
    p = Exp3(arms=[0, 1], gamma=0.1, seed=0)
    _ = p.update(0, float("nan"))
    assert p.n_missing == 1


def test_exp3_unknown_arm_counted():
    p = Exp3(arms=[0, 1], gamma=0.1, seed=0)
    _ = p.update(99, 1.0)
    assert p.n_missing == 1


def test_exp3_weights_stay_finite():
    p = Exp3(arms=[0, 1], gamma=0.5, seed=0)
    for _ in range(2000):
        a = p.pull()
        _ = p.update(a, 1.0)
    assert all(np.isfinite(w) for w in p.weights_.values())
    assert all(w > 0.0 for w in p.weights_.values())


def test_exp3_clone():
    p = Exp3(arms=[0, 1, 2], gamma=0.05, seed=7)
    c = p.clone()
    assert type(c) is Exp3
    assert c.gamma == 0.05
    assert c.seed == 7


def test_exp3_is_uniform_when_every_weight_vanishes():
    p = Exp3(arms=[0, 1, 2], gamma=0.1, seed=0)
    p.weights_ = {0: 0.0, 1: 0.0, 2: 0.0}
    assert p._probabilities() == {0: 1.0 / 3, 1: 1.0 / 3, 2: 1.0 / 3}


def test_exp3_returns_the_last_arm_when_the_draw_exceeds_the_cumulative_sum():
    class _Draw:
        def random(self):
            return 2.0

    p = Exp3(arms=[0, 1], gamma=0.1, seed=0)
    p._rng = _Draw()
    assert p.pull() == 1
