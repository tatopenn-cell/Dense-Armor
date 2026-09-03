# -*- coding: utf-8 -*-
"""
test/test_rate_limiter.py
===========================
Tests for utility/rate_limiter.py's rate_limited_follower, promoted
from Dense-Evolution-Discovery after validation on two independent
real physical domains (SO-101, ALOHA -- see utility/rate_limiter.py's
own docstring for the real numbers).
"""
import numpy as np

from dense_armor.utility.rate_limiter import rate_limited_follower


def test_causal_output_unaffected_by_truncating_future_data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50).cumsum()
    i = 30
    out_full = rate_limited_follower(x, max_vel=1.0, max_accel=0.5)
    out_truncated = rate_limited_follower(x[:i + 1], max_vel=1.0, max_accel=0.5)
    assert np.isclose(out_full[i], out_truncated[i])


def test_velocity_never_exceeds_max_vel():
    x = np.concatenate([np.zeros(5), np.full(20, 1000.0)])
    out = rate_limited_follower(x, max_vel=2.0, max_accel=1.0)
    assert np.all(np.abs(np.diff(out)) <= 2.0 + 1e-9)


def test_acceleration_never_exceeds_max_accel():
    x = np.concatenate([np.zeros(5), np.full(20, 1000.0)])
    out = rate_limited_follower(x, max_vel=100.0, max_accel=0.3)
    accel = np.diff(np.diff(out))
    assert np.all(np.abs(accel) <= 0.3 + 1e-9)


def test_converges_to_a_constant_target_when_limits_are_generous():
    x = np.concatenate([np.zeros(5), np.full(50, 10.0)])
    out = rate_limited_follower(x, max_vel=100.0, max_accel=100.0)
    assert np.isclose(out[-1], 10.0, atol=1e-6)


def test_isolated_spike_capped_not_fully_passed_through():
    x = np.zeros(30)
    x[15] = 1000.0
    out = rate_limited_follower(x, max_vel=1.0, max_accel=0.5)
    assert out[15] < 100.0


def test_smooth_real_shaped_signal_tracked_closely_when_within_limits():
    t = np.linspace(0, 2 * np.pi, 100)
    x = np.sin(t)
    out = rate_limited_follower(x, max_vel=1.0, max_accel=1.0)
    assert np.sqrt(np.mean((out - x) ** 2)) < 0.05


def test_max_instantaneous_jump_stays_bounded_at_max_vel_under_a_real_spike():
    """The actual promoted property: a single, large real spike still
    produces an applied jump no larger than max_vel per step, unlike a
    moving median which can still let a partial jump through."""
    x = np.zeros(20)
    x[10] = 500.0
    out = rate_limited_follower(x, max_vel=3.0, max_accel=1.0)
    assert np.max(np.abs(np.diff(out))) <= 3.0 + 1e-9
