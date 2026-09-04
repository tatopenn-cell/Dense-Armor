# -*- coding: utf-8 -*-
"""
test/test_trajectory.py
=========================
Tests for utility/trajectory.py's quintic_trajectory, promoted from
Dense-Evolution-Discovery after validation on two independent real
physical domains (SO-101, ALOHA -- see utility/trajectory.py's own
docstring for the real numbers).
"""
import numpy as np

from dense_armor.utility.trajectory import quintic_trajectory


def test_boundary_conditions_satisfied_exactly():
    q0, qf, T = [0.0], [10.0], 2.0
    v0, a0, vf, af = [1.0], [0.5], [-2.0], [0.3]
    t, q, v, a = quintic_trajectory(q0, qf, T, v0, a0, vf, af, n_samples=200)
    assert abs(q[0, 0] - q0[0]) < 1e-9
    assert abs(v[0, 0] - v0[0]) < 1e-9
    assert abs(a[0, 0] - a0[0]) < 1e-9
    assert abs(q[-1, 0] - qf[0]) < 1e-9
    assert abs(v[-1, 0] - vf[0]) < 1e-9
    assert abs(a[-1, 0] - af[0]) < 1e-6


def test_matches_independent_numeric_differentiation():
    t, q, v, a = quintic_trajectory([0.0], [10.0], 2.0, [1.0], [0.5], [-2.0], [0.3], n_samples=200)
    dt = t[1] - t[0]
    v_numeric = np.gradient(q[:, 0], dt)
    a_numeric = np.gradient(v_numeric, dt)
    assert np.max(np.abs(v[5:-5, 0] - v_numeric[5:-5])) < 0.01
    assert np.max(np.abs(a[5:-5, 0] - a_numeric[5:-5])) < 0.1


def test_default_boundary_conditions_are_zero():
    t, q, v, a = quintic_trajectory([0.0], [1.0], 1.0)
    assert abs(v[0, 0]) < 1e-9
    assert abs(a[0, 0]) < 1e-9
    assert abs(v[-1, 0]) < 1e-9
    assert abs(a[-1, 0]) < 1e-9


def test_works_for_any_number_of_dof():
    n_dof = 14
    q0 = np.zeros(n_dof)
    qf = np.arange(n_dof, dtype=float)
    t, q, v, a = quintic_trajectory(q0, qf, 3.0, n_samples=50)
    assert q.shape == (50, n_dof)
    assert np.allclose(q[0], q0)
    assert np.allclose(q[-1], qf)


def test_stationary_start_end_gives_a_real_trajectory_shape_not_a_straight_line():
    t, q, v, a = quintic_trajectory([0.0], [1.0], 1.0, n_samples=101)
    mid = q[50, 0]
    assert abs(mid - 0.5) < 1e-6
    assert v[25, 0] < v[50, 0]
    assert v[50, 0] > v[75, 0]


def test_mismatched_dof_raises():
    import pytest
    with pytest.raises(AssertionError):
        quintic_trajectory([0.0, 0.0], [1.0], 1.0)
