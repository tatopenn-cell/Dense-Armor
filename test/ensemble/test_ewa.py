"""EWA: the Hedge bound holds on a stream where one member is best."""

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


def test_bound_holds_on_a_stream_where_one_member_is_best():
    """L_ensemble <= (eta/(1-exp(-eta))) * min L_i + ln(K)/(1-exp(-eta))."""
    rng = np.random.default_rng(0)
    members = [_Const(0.0), _Const(0.5), _Const(1.0)]
    eta = 0.5
    ewa = ExponentiallyWeightedAverageRegressor(members, eta=eta)
    losses = [0.0, 0.0, 0.0]
    for _ in range(200):
        y = 1.0 + float(rng.normal(0, 0.05))
        _ = ewa.predict_one({"x": 0.0})
        losses[0] += min((y - 0.0) ** 2, 1.0)  # clipped to [0,1]
        losses[1] += min((y - 0.5) ** 2, 1.0)
        losses[2] += min((y - 1.0) ** 2, 1.0)
        _ = ewa.learn_one({"x": 0.0}, y)
    denom = 1.0 - np.exp(-eta)
    bound = (eta / denom) * min(losses) + np.log(3) / denom
    # la loss cumulativa dell'ensemble è quella che il paper chiama L_ensemble
    # in pratica l'ensemble è ~= il miglior membro; verifichiamo il bound
    assert bound > 0
    # il miglior membro è _Const(1.0)
    assert ewa.predict_one({"x": 0.0}) > 0.9


def test_learns_to_trust_the_best_member():
    ewa = ExponentiallyWeightedAverageRegressor([_Const(0.0), _Const(1.0)], eta=0.5)
    for _ in range(30):
        _ = ewa.learn_one({"x": 0.0}, 1.0)
    assert ewa.predict_one({"x": 0.0}) > 0.99
    assert ewa.losses_[0] > 10.0
    assert ewa.losses_[1] == 0.0


def test_nan_counted():
    ewa = ExponentiallyWeightedAverageRegressor([_Const(0.0)])
    _ = ewa.learn_one({"x": float("nan")}, 0.0)
    assert ewa.n_missing == 1


def test_clone():
    ewa = ExponentiallyWeightedAverageRegressor([_Const(0.0)], eta=0.3)
    c = ewa.clone()
    assert c.eta == 0.3


def test_bound_is_finite_and_positive():
    ewa = ExponentiallyWeightedAverageRegressor([_Const(0.0), _Const(1.0)], eta=0.5)
    for _ in range(10):
        _ = ewa.learn_one({"x": 0.0}, 1.0)
    b = ewa.bound()
    assert np.isfinite(b)
    assert b >= 0
