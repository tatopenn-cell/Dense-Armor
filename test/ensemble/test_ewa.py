"""Exponentially weighted average: Hedge weight and Freund-Schapire Eq. 9."""

import numpy as np

from dense_armor.roles import Regressor
from dense_armor.utility.ensemble import ExponentiallyWeightedAverageRegressor


class _Const(Regressor):
    def __init__(self, v=0.0):
        self.v = float(v)

    def learn_one(self, x, y, t=None):
        return self

    def predict_one(self, x, t=None, return_std=False):
        return self.v


class _BadPredict(Regressor):
    def learn_one(self, x, y, t=None):
        return self

    def predict_one(self, x, t=None, return_std=False):
        raise TypeError("bad")


class _BadLearn(Regressor):
    def learn_one(self, x, y, t=None):
        raise AttributeError("no learn")

    def predict_one(self, x, t=None, return_std=False):
        return 0.0


def test_bound_matches_paper_formula():
    ewa = ExponentiallyWeightedAverageRegressor([_Const(0.0), _Const(1.0)], eta=0.5)
    for _ in range(30):
        _ = ewa.learn_one({"x": 0.0}, 1.0)
    expected = (0.5 / (1.0 - np.exp(-0.5))) * min(ewa.losses_) + np.log(2) / (
        1.0 - np.exp(-0.5)
    )
    assert abs(ewa.bound() - expected) < 1e-9


def test_bound_tiny_eta_uses_expm1():
    ewa = ExponentiallyWeightedAverageRegressor([_Const(0.0), _Const(1.0)], eta=1e-17)
    for _ in range(3):
        _ = ewa.learn_one({"x": 0.0}, 1.0)
    b = ewa.bound()
    assert np.isfinite(b)
    assert b >= 0.0


def test_bound_responds_to_eta():
    a = ExponentiallyWeightedAverageRegressor([_Const(0.0), _Const(1.0)], eta=0.1)
    b = ExponentiallyWeightedAverageRegressor([_Const(0.0), _Const(1.0)], eta=1.0)
    for _ in range(30):
        _ = a.learn_one({"x": 0.0}, 1.0)
        _ = b.learn_one({"x": 0.0}, 1.0)
    assert a.bound() != b.bound()


def test_weights_keep_the_best_member_with_large_losses():
    """The reviewer case: three members at 0, 50, 90, truth 100."""
    ewa = ExponentiallyWeightedAverageRegressor(
        [_Const(0.0), _Const(50.0), _Const(90.0)], eta=0.5
    )
    for _ in range(30):
        _ = ewa.learn_one({"x": 0.0}, 100.0)
    pred = ewa.predict_one({"x": 0.0})
    assert pred > 85.0


def test_prediction_is_not_a_simple_mean():
    ewa = ExponentiallyWeightedAverageRegressor(
        [_Const(0.0), _Const(50.0), _Const(90.0)], eta=0.5
    )
    for _ in range(30):
        _ = ewa.learn_one({"x": 0.0}, 100.0)
    pred = ewa.predict_one({"x": 0.0})
    assert abs(pred - (0.0 + 50.0 + 90.0) / 3.0) > 30.0


def test_best_member_wins_on_a_clean_stream():
    ewa = ExponentiallyWeightedAverageRegressor([_Const(0.0), _Const(1.0)], eta=0.5)
    for _ in range(30):
        _ = ewa.learn_one({"x": 0.0}, 1.0)
    assert ewa.predict_one({"x": 0.0}) > 0.99
    assert ewa.losses_[0] > 10.0
    assert ewa.losses_[1] == 0.0


def test_predict_before_learning_returns_zero():
    ewa = ExponentiallyWeightedAverageRegressor([_Const(0.0)])
    assert ewa.predict_one({"x": 0.0}) == 0.0


def test_bound_before_learning_is_zero():
    ewa = ExponentiallyWeightedAverageRegressor([_Const(0.0)])
    assert ewa.bound() == 0.0


def test_return_std():
    ewa = ExponentiallyWeightedAverageRegressor([_Const(0.0), _Const(1.0)], eta=0.5)
    for _ in range(5):
        _ = ewa.learn_one({"x": 0.0}, 1.0)
    _y, s = ewa.predict_one({"x": 0.0}, return_std=True)
    assert s >= 0.0


def test_member_predict_raises_is_zero():
    ewa = ExponentiallyWeightedAverageRegressor([_BadPredict])
    for _ in range(3):
        _ = ewa.learn_one({"x": 0.0}, 1.0)
    assert ewa.predict_one({"x": 0.0}) == 0.0


def test_member_learn_raises_is_caught():
    ewa = ExponentiallyWeightedAverageRegressor([_BadLearn])
    _ = ewa.learn_one({"x": 0.0}, 1.0)
    assert ewa.n_seen_ == 1


def test_nan_counted():
    ewa = ExponentiallyWeightedAverageRegressor([_Const(0.0)])
    _ = ewa.learn_one({"x": float("nan")}, 0.0)
    assert ewa.n_missing == 1


def test_clone():
    ewa = ExponentiallyWeightedAverageRegressor([_Const(0.0)], eta=0.3)
    assert ewa.clone().eta == 0.3


def test_rejects_bad_eta_and_empty_models():
    for bad_eta, models in [(0.0, [_Const]), (-1.0, [_Const]), (1.0, [])]:
        try:
            ExponentiallyWeightedAverageRegressor(models, eta=bad_eta)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_skip_set_is_empty():
    assert ExponentiallyWeightedAverageRegressor([_Const])._unit_test_skips() == set()
