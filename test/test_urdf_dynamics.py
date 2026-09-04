# -*- coding: utf-8 -*-
"""
test/test_urdf_dynamics.py
=============================
Tests for dynamics/urdf_dynamics.py's RigidBodyModel, promoted from
Dense-Evolution-Discovery Experiment 62 after validation on three
independent real robots (Kinova Gen3 7-DoF, Kinova Gen3 6-DoF, Franka
Emika Panda -- see dynamics/urdf_dynamics.py's own docstring for the
real numbers).
"""
import os

import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.dynamics.urdf_dynamics import RigidBodyModel

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "urdf")

ROBOTS = [
    ("GEN3_URDF_V12.urdf", 7),
    ("GEN3-6DOF.urdf", 6),
    ("panda.urdf", 9),
]


@pytest.mark.parametrize("urdf_file,expected_n", ROBOTS)
def test_dof_count_matches_the_real_urdf(urdf_file, expected_n):
    m = RigidBodyModel(os.path.join(FIXTURES, urdf_file))
    assert m.n == expected_n


@pytest.mark.parametrize("urdf_file,expected_n", ROBOTS)
def test_mass_matrix_symmetric_positive_definite(urdf_file, expected_n):
    m = RigidBodyModel(os.path.join(FIXTURES, urdf_file))
    rng = np.random.default_rng(0)
    for _ in range(10):
        q = jnp.array(rng.uniform(-1.5, 1.5, m.n))
        mat = np.array(m.mass_matrix(q))
        assert np.max(np.abs(mat - mat.T)) < 1e-9
        assert np.linalg.eigvalsh(mat).min() > 0


@pytest.mark.parametrize("urdf_file,expected_n", ROBOTS)
def test_energy_conserved_under_free_dynamics(urdf_file, expected_n):
    import jax
    from functools import partial

    m = RigidBodyModel(os.path.join(FIXTURES, urdf_file))
    rng = np.random.default_rng(1)
    q0 = jnp.array(rng.uniform(-1.0, 1.0, m.n))
    qd0 = jnp.array(rng.uniform(-0.5, 0.5, m.n))

    @jax.jit
    def rk4_step(q, qd, dt):
        tau = jnp.zeros(m.n)

        def deriv(q, qd):
            return qd, m.forward_dynamics(q, qd, tau)

        k1q, k1v = deriv(q, qd)
        k2q, k2v = deriv(q + 0.5 * dt * k1q, qd + 0.5 * dt * k1v)
        k3q, k3v = deriv(q + 0.5 * dt * k2q, qd + 0.5 * dt * k2v)
        k4q, k4v = deriv(q + dt * k3q, qd + dt * k3v)
        return (q + (dt / 6.0) * (k1q + 2 * k2q + 2 * k3q + k4q),
                qd + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v))

    @partial(jax.jit, static_argnames=("n_steps",))
    def simulate(q0, qd0, dt, n_steps):
        def scan_fn(carry, _):
            q, qd = carry
            q_next, qd_next = rk4_step(q, qd, dt)
            return (q_next, qd_next), m.total_energy(q_next, qd_next)

        _, energies = jax.lax.scan(scan_fn, (q0, qd0), None, length=n_steps)
        return energies

    e0 = float(m.total_energy(q0, qd0))
    drifts = []
    for dt in (1e-2, 1e-3):
        energies = simulate(q0, qd0, dt, int(round(0.3 / dt)))
        max_drift = float(jnp.max(jnp.abs(energies - e0)))
        drifts.append(max_drift / abs(e0))

    assert drifts[0] < 1.0
    assert drifts[1] < drifts[0] / 50


def test_branching_tree_and_prismatic_joints_parse_correctly():
    """The Panda has independent left/right finger prismatic joints off the
    same parent link (panda_hand) -- a real branching tree, not a chain."""
    m = RigidBodyModel(os.path.join(FIXTURES, "panda.urdf"))
    joint_types = {j["name"]: j["type"] for j in m.dof_joints}
    assert joint_types["panda_finger_joint1"] == "prismatic"
    assert joint_types["panda_finger_joint2"] == "prismatic"
    assert m.link_parent_joint["panda_leftfinger"]["parent"] == "panda_hand"
    assert m.link_parent_joint["panda_rightfinger"]["parent"] == "panda_hand"


@pytest.mark.parametrize("urdf_file,expected_n", ROBOTS)
def test_link_jacobian_matches_autodiff_of_link_position(urdf_file, expected_n):
    """Cross-check: model.link_jacobian must equal jax.jacfwd of model.link_position,
    for every link, not only an end-effector -- a real correctness check, not just a
    coverage exercise."""
    import jax

    m = RigidBodyModel(os.path.join(FIXTURES, urdf_file))
    rng = np.random.default_rng(3)
    q = jnp.array(rng.uniform(-1.0, 1.0, m.n))

    for link_name in m.link_names:
        jac_analytic = np.array(m.link_jacobian(q, link_name))
        jac_autodiff = np.array(jax.jacfwd(lambda qq: m.link_position(qq, link_name))(q))
        assert np.max(np.abs(jac_analytic - jac_autodiff)) < 1e-8


@pytest.mark.parametrize("urdf_file,expected_n", ROBOTS)
def test_com_positions_shape_and_finite(urdf_file, expected_n):
    m = RigidBodyModel(os.path.join(FIXTURES, urdf_file))
    rng = np.random.default_rng(4)
    q = jnp.array(rng.uniform(-1.0, 1.0, m.n))
    com = np.array(m.com_positions(q))
    assert com.shape == (len(m.link_names), 3)
    assert np.all(np.isfinite(com))
