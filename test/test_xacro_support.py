# -*- coding: utf-8 -*-
"""
test/test_xacro_support.py
============================
Tests for urdf_dynamics.RigidBodyModel's `.xacro` support, promoted from
Dense-Evolution-Discovery Experiment 66 -- see that module's own docstring
for the real inconsistency found and fixed in the Franka Panda's own
published macros along the way.
"""
import os

import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.dynamics.urdf_dynamics import RigidBodyModel

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "urdf")
XACRO_PATH = os.path.join(FIXTURES, "xacro_panda", "panda_arm_hand.urdf.xacro")


def test_xacro_expands_to_full_panda_model():
    """
    xacro.process_file() must expand the real, unmodified Franka Panda macro
    source (arm macro + hand macro, via xacro:include) into the 7-arm-joint
    model, plus one independent gripper coordinate -- hand.xacro's second
    finger is a real <mimic> of the first (see test_mimic_joints.py), not a
    second independent DOF.
    """
    m = RigidBodyModel(XACRO_PATH)
    assert m.n == 8
    names = [j["name"] for j in m.dof_joints]
    assert names == [
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7",
        "panda_finger_joint1",
    ]


def test_xacro_joint_limits_expanded_correctly():
    m = RigidBodyModel(XACRO_PATH)
    assert float(m.q_min[3]) == pytest.approx(-3.0718)
    assert float(m.q_max[3]) == pytest.approx(-0.0698)


def test_xacro_hand_link_reachable():
    """
    Regression guard for the real link8-disconnection issue: hand.xacro's
    connected_to="panda_link8" requires that link to exist in the expanded
    tree, otherwise link_pose('panda_hand') raises KeyError.
    """
    m = RigidBodyModel(XACRO_PATH)
    q0 = jnp.zeros(m.n)
    pos, _ = m.link_pose(q0, "panda_hand")
    assert np.allclose(np.array(pos), [0.088, -0.1, 0.593], atol=1e-9)


def test_xacro_expansion_is_deterministic():
    """
    Two independent expansions of the same xacro source must produce
    identical models -- xacro.process_file() must not depend on hidden
    global state across calls.
    """
    m1 = RigidBodyModel(XACRO_PATH)
    m2 = RigidBodyModel(XACRO_PATH)
    rng = np.random.default_rng(3)
    for _ in range(10):
        q = jnp.array(rng.uniform(-1.5, 1.5, m1.n))
        mass1 = np.array(m1.mass_matrix(q))
        mass2 = np.array(m2.mass_matrix(q))
        assert np.max(np.abs(mass1 - mass2)) == 0.0
