# -*- coding: utf-8 -*-
"""
test/test_kinematic_controller.py
====================================
Tests for utility/kinematic_controller.py's kinematic_tracking_controller,
promoted from Dense-Evolution-Discovery after validation on two
independent real physical domains (SO-101, ALOHA -- see
utility/kinematic_controller.py's own docstring for the real numbers).
"""
import numpy as np

from dense_armor.utility.kinematic_controller import kinematic_tracking_controller


def test_exact_exponential_convergence_constant_reference():
    q = np.array([5.0])
    q_ref = np.array([0.0])
    qd_ref = np.array([0.0])
    kp = 2.0
    dt = 0.001
    errors = []
    for _ in range(5000):
        e = q_ref - q
        errors.append(e[0])
        u = kinematic_tracking_controller(q, q_ref, qd_ref, kp)
        q = q + u * dt
    errors = np.array(errors)
    t = np.arange(len(errors)) * dt
    theory = errors[0] * np.exp(-kp * t)
    assert np.max(np.abs(errors - theory)) < 0.05


def test_exact_convergence_time_varying_reference():
    t = np.linspace(0, 3, 3000)
    dt = t[1] - t[0]
    q_ref_traj = np.sin(t)
    qd_ref_traj = np.cos(t)
    q = np.array([5.0])
    kp = 8.0
    for i in range(len(t)):
        q_ref = np.array([q_ref_traj[i]])
        qd_ref = np.array([qd_ref_traj[i]])
        u = kinematic_tracking_controller(q, q_ref, qd_ref, kp)
        q = q + u * dt
    final_error = abs(q_ref_traj[-1] - q[0])
    assert final_error < 0.01


def test_zero_gain_gives_pure_feedforward():
    q = np.array([0.0, 0.0])
    q_ref = np.array([1.0, 2.0])
    qd_ref = np.array([3.0, 4.0])
    u = kinematic_tracking_controller(q, q_ref, qd_ref, kp=0.0)
    assert np.allclose(u, qd_ref)


def test_works_for_any_number_of_dof():
    n = 14
    q = np.zeros(n)
    q_ref = np.arange(n, dtype=float)
    qd_ref = np.zeros(n)
    u = kinematic_tracking_controller(q, q_ref, qd_ref, kp=1.0)
    assert u.shape == (n,)
    assert np.allclose(u, q_ref)


def test_mismatched_shapes_raise():
    import pytest
    with pytest.raises(AssertionError):
        kinematic_tracking_controller([0.0, 0.0], [1.0], [0.0], kp=1.0)


def test_chains_with_quintic_trajectory_recovers_from_real_perturbation():
    from dense_armor.utility.trajectory import quintic_trajectory
    t, q_ref, qd_ref, _ = quintic_trajectory([0.0], [10.0], 3.0, n_samples=1000)
    dt = t[1] - t[0]
    kp = 5.0
    q = 2.0  # real disclosed nonzero initial tracking error
    for i in range(len(t)):
        u = kinematic_tracking_controller([q], q_ref[i], qd_ref[i], kp=kp)
        q = q + float(u[0]) * dt
    final_error = abs(float(q_ref[-1, 0]) - q)
    assert final_error < 0.01
