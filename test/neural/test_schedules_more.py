"""Tests for schedules_more.py."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from dense_armor.utility.neural.chain import Sequential
from dense_armor.utility.neural.layers import Dense
from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.schedules_more import LinearWarmup, NoamSchedule
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import SGD


def test_linear_warmup_hand_case():
    s = LinearWarmup(0.1, 3.2, 5)
    for e, expected in [(0.0, 0.1), (2.5, 1.65), (5.0, 3.2), (7.0, 3.2)]:
        got = s(e)
        assert abs(got - expected) < 1e-9, (
            f"LinearWarmup({e}) = {got}, expected {expected}"
        )


def test_noam_hand_case():
    s = NoamSchedule(512, 4000)
    assert s(0.0) == 0.0
    for e, expected in [
        (4000.0, 0.0006987712),
        (1.0, 1.7469281e-07),
        (8000.0, 0.0004941059),
    ]:
        got = s(e)
        assert abs(got - expected) < 1e-9, (
            f"NoamSchedule({e}) = {got}, expected {expected}"
        )


def test_noam_shape_increases_then_decreases():
    s = NoamSchedule(64, 20)
    before = [s(e) for e in np.linspace(0.1, 20.0, 20)]
    after = [s(e) for e in np.linspace(20.1, 200.0, 20)]
    for i in range(1, len(before)):
        assert before[i] >= before[i - 1] - 1e-9, (
            f"not increasing at i={i}: {before[i - 1]} -> {before[i]}"
        )
    for i in range(1, len(after)):
        assert after[i] <= after[i - 1] + 1e-12, (
            f"not decreasing at i={i}: {after[i - 1]} -> {after[i]}"
        )


def test_linear_warmup_in_trainer_history():
    X = np.random.default_rng(0).normal(0.0, 1.0, (8, 2)).astype(np.float32)
    y = (X[:, 0:1] + 2 * X[:, 1:2]).astype(np.float32)
    m = Sequential([Dense(2, 1, jax.random.PRNGKey(0))])
    t = Trainer(m, MSE(), SGD(0.01), schedule=LinearWarmup(0.01, 0.05, 4))
    t.fit(X, y, epochs=5, batch_size=8, key=jax.random.PRNGKey(0), shuffle=False)
    expected = [0.01, 0.02, 0.03, 0.04, 0.05]
    for got, exp in zip(t.history_["lr"], expected):
        assert abs(got - exp) < 1e-9, f"lr history: {t.history_['lr']} vs {expected}"


def test_noam_lr_reaches_optimizer():
    from dense_armor.utility.neural.online import OnlineNetRegressor

    m = Sequential([Dense(2, 1, jax.random.PRNGKey(0))])
    on = OnlineNetRegressor(
        m,
        MSE(),
        SGD(0.01),
        feature_keys=["a", "b"],
        schedule=NoamSchedule(2, 4),
    )
    X = np.random.default_rng(0).normal(0.0, 1.0, (6, 2)).astype(np.float32)
    for i in range(6):
        on.learn_one({"a": float(X[i, 0]), "b": float(X[i, 1])}, float(X[i, 0]))
        expected = NoamSchedule(2, 4)(float(i))
        actual = on.optimizers_[0].lr
        assert abs(actual - expected) < 1e-9, (
            f"step {i}: optimizer lr {actual} vs schedule {expected}"
        )


def test_multiplier():
    s = LinearWarmup(0.1, 3.2, 5)
    assert abs(s.multiplier(0.0) - 1.0) < 1e-9
    assert abs(s.multiplier(2.5) - 16.5) < 1e-9
    n = NoamSchedule(512, 4000)
    assert n.multiplier(0.0) == 0.0
    assert n.multiplier(4000.0) == 0.0


def test_negative_epoch_raises():
    for s in [LinearWarmup(0.1, 3.2, 5), NoamSchedule(512, 4000)]:
        with pytest.raises(ValueError):
            s(-1.0)


def test_value_errors():
    with pytest.raises(ValueError):
        LinearWarmup(0.0, 3.2, 5)
    with pytest.raises(ValueError):
        LinearWarmup(-0.1, 3.2, 5)
    with pytest.raises(ValueError):
        LinearWarmup(0.1, 0.0, 5)
    with pytest.raises(ValueError):
        LinearWarmup(0.1, -3.2, 5)
    with pytest.raises(ValueError):
        LinearWarmup(0.1, 3.2, 0.0)
    with pytest.raises(ValueError):
        LinearWarmup(3.2, 0.1, 5)
    with pytest.raises(ValueError):
        NoamSchedule(0, 4000)
    with pytest.raises(ValueError):
        NoamSchedule(-512, 4000)
    with pytest.raises(ValueError):
        NoamSchedule(512, 0)
    with pytest.raises(ValueError):
        NoamSchedule(512, -4000)
    with pytest.raises(ValueError):
        NoamSchedule(512, 4000, 0.0)
    with pytest.raises(ValueError):
        NoamSchedule(512, 4000, -1.0)


def test_multiplier_returns_zero_when_base_is_zero():
    # For LinearWarmup, the base lr_start is always > 0, so multiplier
    # is a real division; the 0.0 branch of LinearWarmup.multiplier can
    # only trigger if lr_start is 0, which the constructor forbids.
    # NoamSchedule(.)(0) == 0.0 always, so multiplier always returns 0.0.
    s = LinearWarmup(0.1, 3.2, 5)
    assert s.multiplier(0.0) == 1.0
    n = NoamSchedule(512, 4000)
    assert n.multiplier(0.0) == 0.0
    assert n.multiplier(100.0) == 0.0
    assert n.multiplier(4000.0) == 0.0


def test_multiplier_zero_branch_directly():
    # Force the 0.0 branch of both classes by mutating lr_start/d_model
    # so that self(0.0) evaluates to 0.0 — no other public path reaches it.
    s = LinearWarmup(0.1, 3.2, 5)
    s.lr_start = 0.0
    s.lr_end = 0.0
    assert s.multiplier(2.5) == 0.0

    n = NoamSchedule(512, 4000)
    # NoamSchedule(.)(0) is 0.0 by definition; multiplier short-circuits
    assert n.multiplier(0.0) == 0.0
