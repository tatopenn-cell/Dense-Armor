# -*- coding: utf-8 -*-
"""
test/test_mimic_joints.py
============================
Tests for urdf_dynamics.RigidBodyModel's <mimic> joint support, promoted
from Dense-Evolution-Discovery Experiment 67.
"""
import os

import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.dynamics.urdf_dynamics import RigidBodyModel

XACRO_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "urdf",
                           "xacro_panda", "panda_arm_hand.urdf.xacro")


def test_mimic_joint_removed_from_independent_dof():
    """
    hand.xacro's <mimic joint="panda_finger_joint1"/> on the second finger
    must not become a second independent coordinate: n counts real DOF, not
    <joint> tags.
    """
    m = RigidBodyModel(XACRO_PATH)
    assert "panda_finger_joint2" not in m.dof_index
    assert m.mimic_map["panda_finger_joint2"] == (m.dof_index["panda_finger_joint1"], 1.0, 0.0)


def test_mimic_joint_moves_with_its_master():
    """
    Driving the real, independent finger_joint1 must move both fingers --
    the second finger's own <mimic> multiplier=1, offset=0 (URDF defaults)
    means it tracks finger_joint1 exactly, closing the gripper symmetrically
    (opposite local axes, "0 1 0" vs "0 -1 0" in hand.xacro, rotated into
    world frame by the hand's own -45-degree yaw).
    """
    m = RigidBodyModel(XACRO_PATH)
    q0 = jnp.zeros(m.n)
    q1 = q0.at[m.dof_index["panda_finger_joint1"]].set(0.02)

    left0, _ = m.link_pose(q0, "panda_leftfinger")
    left1, _ = m.link_pose(q1, "panda_leftfinger")
    right0, _ = m.link_pose(q0, "panda_rightfinger")
    right1, _ = m.link_pose(q1, "panda_rightfinger")

    d_left = np.array(left1 - left0)
    d_right = np.array(right1 - right0)
    assert np.linalg.norm(d_left) == pytest.approx(0.02, abs=1e-9)
    assert np.allclose(d_right, -d_left, atol=1e-9)


def test_mimic_jacobian_matches_autodiff_of_link_position():
    """
    The hand-built geometric Jacobian's mimic chain-rule column must match a
    real finite-difference derivative of link_pose w.r.t. the master
    coordinate -- not just look plausible.
    """
    m = RigidBodyModel(XACRO_PATH)
    rng = np.random.default_rng(4)
    q = jnp.array(rng.uniform(-0.5, 0.5, m.n))
    master_idx = m.dof_index["panda_finger_joint1"]

    jv = m.link_jacobian(q, "panda_rightfinger")

    eps = 1e-6
    q_plus = q.at[master_idx].add(eps)
    q_minus = q.at[master_idx].add(-eps)
    pos_plus, _ = m.link_pose(q_plus, "panda_rightfinger")
    pos_minus, _ = m.link_pose(q_minus, "panda_rightfinger")
    fd = np.array((pos_plus - pos_minus) / (2 * eps))

    assert np.max(np.abs(np.array(jv[:, master_idx]) - fd)) < 1e-5


def test_mass_matrix_symmetric_positive_definite_with_mimic():
    m = RigidBodyModel(XACRO_PATH)
    rng = np.random.default_rng(5)
    for _ in range(10):
        q = jnp.array(rng.uniform(-1.5, 1.5, m.n))
        mat = np.array(m.mass_matrix(q))
        assert np.max(np.abs(mat - mat.T)) < 1e-9
        assert np.linalg.eigvalsh(mat).min() > 0
