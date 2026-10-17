"""Tests for the four active strategies."""

import numpy as np
import pytest

from dense_armor.utility.active import (
    FixedUncertainty,
    RandomStrategy,
    VariableRandomizedUncertainty,
    VariableUncertainty,
)


class _Conf:
    """Classifier stub whose confidence is a deterministic function of x."""

    def __init__(self, conf: float = 0.5):
        self.conf = conf

    def predict_proba_one(self, x):
        c = float(x.get("conf", self.conf))
        return {0: 1.0 - c, 1: c}


def test_random_rate_close_to_budget():
    s = RandomStrategy(budget=0.2, seed=0)
    asks = sum(s.should_label({}, _Conf()) for _ in range(2000))
    assert 0.18 <= asks / 2000 <= 0.22


def test_random_budget_one_always_labels():
    s = RandomStrategy(budget=1.0, seed=0)
    for _ in range(100):
        assert s.should_label({}, _Conf())


def test_fixed_uncertainty_below_threshold():
    s = FixedUncertainty(theta=0.7)
    assert s.should_label({"conf": 0.55}, _Conf())
    assert not s.should_label({"conf": 0.9}, _Conf())


def test_variable_uncertainty_respects_budget():
    s = VariableUncertainty(budget=0.1, s=0.001, seed=0)
    asks = 0
    for i in range(2000):
        c = 0.5 + 0.4 * np.sin(i / 20.0)
        asks += s.should_label({"conf": float(c)}, _Conf())
    assert 0.05 <= asks / 2000 <= 0.15


def test_variable_uncertainty_never_exceeds_budget():
    s = VariableUncertainty(budget=0.1, s=0.01, seed=0)
    for _ in range(500):
        s.should_label({"conf": 0.49}, _Conf())
        assert s.n_labelled_ <= s.n_seen_ * s.budget + 1


def test_variable_randomized_uncertainty_respects_budget():
    s = VariableRandomizedUncertainty(budget=0.1, s=0.001, delta=0.05, seed=0)
    asks = 0
    for i in range(2000):
        c = 0.5 + 0.4 * np.sin(i / 20.0)
        asks += s.should_label({"conf": float(c)}, _Conf())
    assert 0.05 <= asks / 2000 <= 0.15


def test_variable_randomized_rejects_bad_delta():
    with pytest.raises(ValueError):
        VariableRandomizedUncertainty(budget=0.1, delta=0.0)


def test_bad_budget_rejected():
    for b in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            RandomStrategy(budget=b)


class _EmptyModel:
    def predict_proba_one(self, x):
        return {}


def test_fixed_uncertainty_labels_when_the_model_has_no_probabilities():
    s = FixedUncertainty(theta=0.5, budget=0.1, seed=0)
    assert s.should_label({"x": 1.0}, _EmptyModel()) is True
    assert s.n_labelled_ == 1
