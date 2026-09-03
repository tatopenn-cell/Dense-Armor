# -*- coding: utf-8 -*-
"""
test/test_cbf_filter.py
=========================
Tests for utility/cbf_filter.py's cbf_safety_filter/cbf_filtered_trajectory,
promoted from Dense-Evolution-Discovery after validation on two independent
real physical domains (SO-101, ALOHA -- see utility/cbf_filter.py's own
docstring for the real numbers).
"""
import numpy as np

from dense_armor.utility.cbf_filter import cbf_safety_filter, cbf_filtered_trajectory


def test_safe_command_passes_through_unchanged():
    assert cbf_safety_filter(x=0.0, u_des=1.0, obstacle=100.0, safe_dist=1.0, alpha_gain=1.0) == 1.0


def test_unsafe_command_toward_obstacle_gets_corrected():
    u_des = -5.0
    u = cbf_safety_filter(x=5.0, u_des=u_des, obstacle=0.0, safe_dist=2.0, alpha_gain=2.0)
    assert u != u_des
    assert u > u_des


def test_command_moving_away_from_obstacle_is_never_blocked():
    u_des = 5.0
    assert cbf_safety_filter(x=1.0, u_des=u_des, obstacle=0.0, safe_dist=2.0, alpha_gain=1.0) == u_des


def test_forward_invariance_from_a_safe_start_never_breached():
    rng = np.random.default_rng(0)
    x_raw = np.cumsum(rng.normal(scale=2.0, size=200))
    obstacle = float(np.median(x_raw))
    safe_dist = 1.0
    assert (x_raw[0] - obstacle) ** 2 - safe_dist ** 2 >= 0
    filtered = cbf_filtered_trajectory(x_raw, obstacle=obstacle, safe_dist=safe_dist, alpha_gain=2.0, n_substeps=20)
    h = (filtered - obstacle) ** 2 - safe_dist ** 2
    assert np.all(h >= -1e-6)


def test_single_large_step_needs_substeps_to_stay_invariant():
    x_raw = np.array([10.0, -10.0])
    obstacle, safe_dist = 0.0, 2.0
    one_step = cbf_filtered_trajectory(x_raw, obstacle, safe_dist, alpha_gain=2.0, n_substeps=1)
    many_steps = cbf_filtered_trajectory(x_raw, obstacle, safe_dist, alpha_gain=2.0, n_substeps=20)
    h_one = (one_step - obstacle) ** 2 - safe_dist ** 2
    h_many = (many_steps - obstacle) ** 2 - safe_dist ** 2
    assert h_many[-1] >= -1e-6
    assert h_one[-1] < h_many[-1]


def test_trajectory_never_approaching_obstacle_is_untouched():
    x_raw = np.linspace(0, 1, 50)
    filtered = cbf_filtered_trajectory(x_raw, obstacle=1000.0, safe_dist=1.0, alpha_gain=1.0, n_substeps=10)
    assert np.allclose(filtered, x_raw)


def test_default_substeps_is_20():
    import inspect
    sig = inspect.signature(cbf_filtered_trajectory)
    assert sig.parameters["n_substeps"].default == 20
