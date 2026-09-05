# -*- coding: utf-8 -*-
"""
test/test_six_dof_pbc_cbf_controller.py
==========================================
Tests for dynamics/six_dof_pbc_cbf_controller.py's solve_control_qp, promoted
from Dense-Evolution-Discovery Experiment 65 -- see that module's own
docstring for the real numbers this reproduces.
"""
import os

import jax.numpy as jnp
import numpy as np

from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
from dense_armor.dynamics.six_dof_pbc_cbf_controller import solve_control_qp, rotation_error

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "urdf")


def test_zero_error_gives_pure_gravity_compensation():
    """
    At the exact desired pose (position and orientation) with zero velocity,
    the solved torque must equal gravity compensation exactly (qdd=0, no
    tracking terms active) -- a real correctness check of the whole 6-DoF
    pipeline (spatial Jacobian, rotation error, task-space Lambda), not just
    "doesn't crash".
    """
    model = RigidBodyModel(os.path.join(FIXTURES, "GEN3_URDF_V12.urdf"))
    link = "end_effector_link"
    q = jnp.array([0.3, -0.6, 0.2, -1.1, 0.4, 0.8, -0.3])
    qd = jnp.zeros(7)
    p, r = model.link_pose(q, link)

    qdd, tau, mu, h = solve_control_qp(model, link, q, qd, p, jnp.zeros(3), jnp.zeros(3),
                                        r, jnp.zeros(3), jnp.zeros(3), eps=0.03)
    grav = model.gravity_forces(q)

    assert np.max(np.abs(np.array(qdd))) < 1e-9
    assert np.max(np.abs(np.array(tau) - np.array(grav))) < 1e-9


def test_rotation_error_zero_iff_equal():
    model = RigidBodyModel(os.path.join(FIXTURES, "GEN3_URDF_V12.urdf"))
    q = jnp.array([0.1, 0.2, 0.3, -0.5, 0.2, 0.6, -0.1])
    _, r = model.link_pose(q, "end_effector_link")
    assert np.max(np.abs(np.array(rotation_error(r, r)))) < 1e-12


def test_six_dof_stays_finite_on_a_different_robot():
    """
    Same real singular configuration as test_passivity_cbf_controller.py's
    3-robot validation bar (Gen3 6-DoF, driven toward its own true
    singularity) -- but through the full 6-DoF pipeline (spatial Jacobian,
    rotation error, task-space Lambda) instead of the 3-DoF one. Also the
    regression test for a real second-level OSQP infeasibility: the 6-DoF
    manipulability measure at this q is well below the 3-DoF one (0.0176 vs
    0.1128), so the CBF+box fallback is itself infeasible and the module
    needs its own third-level (CBF-only) fallback -- see the module's own
    docstring/comments.
    """
    model = RigidBodyModel(os.path.join(FIXTURES, "GEN3-6DOF.urdf"))
    link = "bracelet_with_vision_link"
    q = jnp.array([0.51570841, 0.7814598, -1.17051634, 0.37482289, -0.25813235, 0.34260431])
    qd = jnp.zeros(6)
    p_des = jnp.array([0.0, 0.00135062, 1.11510006])
    _, r_des = model.link_pose(q, link)

    qdd, tau, mu, h = solve_control_qp(model, link, q, qd, p_des, jnp.zeros(3), jnp.zeros(3),
                                        r_des, jnp.zeros(3), jnp.zeros(3), eps=0.03)

    assert np.all(np.isfinite(qdd))
    assert np.all(np.isfinite(tau))
    assert np.linalg.norm(qdd) < 1e4


def test_six_dof_stays_finite_on_a_third_robot_different_manufacturer():
    """Third robot, a different manufacturer (Franka Panda) -- same real
    singular configuration as test_passivity_cbf_controller.py's own third
    test, through the full 6-DoF pipeline."""
    model = RigidBodyModel(os.path.join(FIXTURES, "panda.urdf"))
    link = "panda_hand"
    q = jnp.array([-0.19360032, 0.89941805, -0.27271074, -1.15251618,
                   -2.92540148, 2.89498581, 2.73584013, 0.0, 0.0])
    qd = jnp.zeros(9)
    p_des = jnp.array([-0.2295431, -0.47162945, 1.00885136])
    _, r_des = model.link_pose(q, link)

    qdd, tau, mu, h = solve_control_qp(model, link, q, qd, p_des, jnp.zeros(3), jnp.zeros(3),
                                        r_des, jnp.zeros(3), jnp.zeros(3), eps=0.03)

    assert np.all(np.isfinite(qdd))
    assert np.all(np.isfinite(tau))
    assert np.linalg.norm(qdd) < 1e4


def test_position_and_orientation_converge():
    """
    Real closed-loop check: a real disclosed 10cm/10cm position offset and a
    real disclosed 30-degree orientation offset (about world z) both
    converge under RK4-integrated real dynamics -- not just a single QP call.
    """
    from functools import partial
    import jax

    @partial(jax.jit, static_argnames=("model", "n_substeps"))
    def rk4_n_substeps(model, q, qd, tau, dt, n_substeps):
        def deriv(q, qd):
            return qd, model.forward_dynamics(q, qd, tau)

        def one_step(carry, _):
            q, qd = carry
            k1q, k1v = deriv(q, qd)
            k2q, k2v = deriv(q + 0.5 * dt * k1q, qd + 0.5 * dt * k1v)
            k3q, k3v = deriv(q + 0.5 * dt * k2q, qd + 0.5 * dt * k2v)
            k4q, k4v = deriv(q + dt * k3q, qd + dt * k3v)
            q_next = q + (dt / 6.0) * (k1q + 2 * k2q + 2 * k3q + k4q)
            qd_next = qd + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
            return (q_next, qd_next), None

        (q_out, qd_out), _ = jax.lax.scan(one_step, (q, qd), None, length=n_substeps)
        return q_out, qd_out

    def axis_angle_to_matrix(axis, angle):
        axis = axis / np.linalg.norm(axis)
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)

    model = RigidBodyModel(os.path.join(FIXTURES, "GEN3_URDF_V12.urdf"))
    link = "end_effector_link"
    q = jnp.array([0.3, -0.6, 0.2, -1.1, 0.4, 0.8, -0.3])
    qd = jnp.zeros(7)
    p0, r0 = model.link_pose(q, link)
    p_des = np.array(p0) + np.array([0.1, 0.0, 0.1])
    r_des = axis_angle_to_matrix(np.array([0.0, 0.0, 1.0]), np.deg2rad(30.0)) @ np.array(r0)

    dt_ctrl = 0.01
    n_substeps = 5
    dt_phys = dt_ctrl / n_substeps
    for k in range(1000):
        qdd, tau, mu, h = solve_control_qp(
            model, link, q, qd, jnp.asarray(p_des), jnp.zeros(3), jnp.zeros(3),
            jnp.asarray(r_des), jnp.zeros(3), jnp.zeros(3), eps=0.03)
        q, qd = rk4_n_substeps(model, q, qd, jnp.asarray(tau), dt_phys, n_substeps)

    p, r = model.link_pose(q, link)
    pos_err = float(jnp.linalg.norm(p - jnp.asarray(p_des)))
    rot_err = float(jnp.linalg.norm(rotation_error(r, jnp.asarray(r_des))))
    assert pos_err < 1e-6
    assert rot_err < 1e-4
