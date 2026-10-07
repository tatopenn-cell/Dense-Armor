"""Tests for dense_armor/roles/uncertainty.py."""
import doctest

import jax.numpy as jnp
import numpy as np
import pytest

import dense_armor.roles.uncertainty as unc_mod
from dense_armor.roles.uncertainty import (
    AdaptiveConformalRegressor,
    Estimate,
    OnlinePlattScaling,
)
from dense_armor.roles.signal import Signal
from dense_armor.roles import Regressor
from dense_armor.utility.stats.moments import RunningMoments


class _Mean(Regressor):
    """Last-seen value, the simplest regressor for tests."""

    def __init__(self) -> None:
        self.m = RunningMoments()
        self._last = 0.0

    def learn_one(self, x, y, t=None):
        self.m.learn_one(x, t=t)
        self._last = float(y)
        return self

    def predict_one(self, x, t=None, return_std=False):
        return self._last


def _sig(i: int) -> Signal:
    return Signal(
        values=jnp.array([float(i)]),
        names=["x"],
        units=[""],
        t=i * 0.01,
    )


def _stream(n: int = 300, noise: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed)
    for i in range(n):
        yield _sig(i), float(i) + float(rng.normal(0.0, noise))


def test_estimate_std():
    e = Estimate(mean=1.0, var=4.0)
    assert e.std == pytest.approx(2.0)
    assert "std=2.0" in repr(e)


def test_estimate_negative_variance_is_clamped():
    e = Estimate(mean=0.0, var=-1.0)
    assert e.std == 0.0


def test_aci_coverage_close_to_target_on_stationary_stream():
    aci = AdaptiveConformalRegressor(
        _Mean(), alpha=0.1, gamma=0.01, window=100
    )
    for x, y in _stream(2000, noise=0.2, seed=1):
        aci.learn_one(x, y)
    assert abs(aci.coverage - 0.9) < 0.05, aci.coverage


def test_aci_interval_width_increases_with_noise():
    aci_lo = AdaptiveConformalRegressor(_Mean(), alpha=0.1, gamma=0.01)
    aci_hi = AdaptiveConformalRegressor(_Mean(), alpha=0.1, gamma=0.01)
    for x, y in _stream(1000, noise=0.1, seed=2):
        aci_lo.learn_one(x, y)
    for x, y in _stream(1000, noise=1.0, seed=2):
        aci_hi.learn_one(x, y)
    lo_w = aci_lo.predict_interval(_sig(0))
    hi_w = aci_hi.predict_interval(_sig(0))
    assert (hi_w[1] - hi_w[0]) > (lo_w[1] - lo_w[0])


def test_aci_alpha_t_stays_in_0_1():
    aci = AdaptiveConformalRegressor(_Mean(), alpha=0.1, gamma=0.5)
    for x, y in _stream(500):
        aci.learn_one(x, y)
        assert 0.0 <= aci.alpha_t_ <= 1.0


def test_aci_alpha_t_reacts_to_shift():
    aci = AdaptiveConformalRegressor(_Mean(), alpha=0.1, gamma=0.05)
    for x, y in _stream(500, noise=0.01, seed=3):
        aci.learn_one(x, y)
    low_noise_alpha = aci.alpha_t_
    for x, y in _stream(500, noise=5.0, seed=4):
        aci.learn_one(x, y)
    high_noise_alpha = aci.alpha_t_
    assert low_noise_alpha != high_noise_alpha


def test_aci_predict_returns_estimate_when_asked():
    aci = AdaptiveConformalRegressor(_Mean())
    for x, y in _stream(100):
        aci.learn_one(x, y)
    est = aci.predict_one(_sig(0), return_estimate=True)
    assert isinstance(est, Estimate)
    assert est.var >= 0.0
    mean, std = aci.predict_one(_sig(0), return_std=True)
    assert isinstance(mean, float) and isinstance(std, float)
    assert std >= 0.0
    plain = aci.predict_one(_sig(0))
    assert isinstance(plain, float)


def test_aci_transform_one_reports_alpha_t_and_coverage():
    aci = AdaptiveConformalRegressor(_Mean())
    for x, y in _stream(200):
        aci.learn_one(x, y)
    d = aci.transform_one(_sig(0))
    assert set(d) == {"lo", "hi", "alpha_t", "coverage"}
    assert d["lo"] <= d["hi"]


def test_aci_has_robot_contract():
    aci = AdaptiveConformalRegressor(_Mean())
    assert aci.budget_s is not None
    assert aci.memory_class == "O(window)"


def test_oplatt_is_reexported():
    assert OnlinePlattScaling is unc_mod.OnlinePlattScaling


def test_doctests():
    assert doctest.testmod(unc_mod).failed == 0
