"""EWA: the Hedge bound of Freund-Schapire 1997 (Eq. 9)."""

import numpy as np

from dense_armor.roles import Regressor
from dense_armor.utility.ensemble.ewa import ExponentiallyWeightedAverageRegressor


class _Const(Regressor):
    def __init__(self, v=0.0):
        self.v = float(v)

    def learn_one(self, x, y, t=None):
        return self

    def predict_one(self, x, t=None, return_std=False):
        return self.v


def test_bound_matches_paper_formula():
    """``bound()`` equals Eq. 9 of the paper for the given losses."""
    ewa = ExponentiallyWeightedAverageRegressor([_Const(0.0), _Const(1.0)], eta=0.5)
    for _ in range(30):
        _ = ewa.learn_one({"x": 0.0}, 1.0)
    expected = (0.5 / (1.0 - np.exp(-0.5))) * min(ewa.losses_) + np.log(2) / (
        1.0 - np.exp(-0.5)
    )
    assert abs(ewa.bound() - expected) < 1e-9


def test_bound_responds_to_eta():
    """Changing eta changes the bound (catches a constant return)."""
    a = ExponentiallyWeightedAverageRegressor([_Const(0.0), _Const(1.0)], eta=0.1)
    b = ExponentiallyWeightedAverageRegressor([_Const(0.0), _Const(1.0)], eta=1.0)
    for _ in range(30):
        _ = a.learn_one({"x": 0.0}, 1.0)
        _ = b.learn_one({"x": 0.0}, 1.0)
    assert a.bound() != b.bound()


def test_best_member_wins():
    ewa = ExponentiallyWeightedAverageRegressor([_Const(0.0), _Const(1.0)], eta=0.5)
    for _ in range(30):
        _ = ewa.learn_one({"x": 0.0}, 1.0)
    assert ewa.predict_one({"x": 0.0}) > 0.99
    assert ewa.losses_[0] > 10.0
    assert ewa.losses_[1] == 0.0


def test_bound_is_finite_and_positive():
    ewa = ExponentiallyWeightedAverageRegressor([_Const(0.0), _Const(1.0)], eta=0.5)
    for _ in range(10):
        _ = ewa.learn_one({"x": 0.0}, 1.0)
    b = ewa.bound()
    assert np.isfinite(b)
    assert b >= 0


def test_nan_counted():
    ewa = ExponentiallyWeightedAverageRegressor([_Const(0.0)])
    _ = ewa.learn_one({"x": float("nan")}, 0.0)
    assert ewa.n_missing == 1


def test_clone():
    ewa = ExponentiallyWeightedAverageRegressor([_Const(0.0)], eta=0.3)
    c = ewa.clone()
    assert c.eta == 0.3
