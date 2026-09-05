# -*- coding: utf-8 -*-
"""
test/test_passivity_cbf_controller.py
========================================
Tests for dynamics/passivity_cbf_controller.py's solve_control_qp, promoted
from Dense-Evolution-Discovery Experiment 63 after validation on three
independent real robots (Kinova Gen3 7-DoF, Kinova Gen3 6-DoF, Franka Emika
Panda -- see dynamics/passivity_cbf_controller.py's own docstring for the
real numbers).
"""
import os
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np

import dense_armor.dynamics.passivity_cbf_controller as pcc
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
from dense_armor.dynamics.passivity_cbf_controller import solve_control_qp, manipulability

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "urdf")


def test_controller_stays_finite_near_a_documented_infeasible_state():
    """
    Regression test for the real OSQP-infeasibility bug found in Discovery
    Experiment 61/63: at this exact joint state, the joint QP was primal
    infeasible under OSQP 1.1.3 as run in Discovery's environment. Re-verified
    here: with this repo's own OSQP 1.1.3 install, the same state solves
    successfully instead (a razor-edge numerical case, sensitive to tiny
    floating-point differences between environments/BLAS builds -- not
    reliably reproducible everywhere). Kept as a real-world sanity check;
    see test_osqp_infeasibility_fallback_path_is_exercised below for a
    deterministic test of the fallback branch itself.
    """
    model = RigidBodyModel(os.path.join(FIXTURES, "GEN3_URDF_V12.urdf"))
    q = jnp.array([0.024434754271692588, 0.21033097888352037, -0.013211504941075125,
                   -1.0242941965027483, -0.007324506055652218, 1.1038655977606235,
                   -0.9812817032340545])
    qd = jnp.array([0.02551702814644204, -0.09708874319385337, -0.00864499017131824,
                    0.9385065660201053, -0.010950684566497454, -0.7280601780288772,
                    -0.3852007482333173])
    p_des = jnp.array([-0.09341796453028546, -0.02485595, 1.072509868409771])
    pd_des = jnp.zeros(3)
    pdd_des = jnp.zeros(3)

    qdd, tau, mu, h = solve_control_qp(model, "end_effector_link", q, qd, p_des, pd_des, pdd_des, eps=0.03)

    assert np.all(np.isfinite(qdd))
    assert np.all(np.isfinite(tau))
    assert np.linalg.norm(qdd) < 1e4


def test_controller_stays_finite_on_a_structurally_different_robot():
    model = RigidBodyModel(os.path.join(FIXTURES, "GEN3-6DOF.urdf"))
    q = jnp.array([0.51570841, 0.7814598, -1.17051634, 0.37482289, -0.25813235, 0.34260431])
    qd = jnp.zeros(6)
    p_des = jnp.array([0.0, 0.00135062, 1.11510006])
    pd_des = jnp.zeros(3)
    pdd_des = jnp.zeros(3)

    qdd, tau, mu, h = solve_control_qp(model, "bracelet_with_vision_link", q, qd, p_des, pd_des, pdd_des, eps=0.03)

    assert np.all(np.isfinite(qdd))
    assert np.all(np.isfinite(tau))
    assert np.linalg.norm(qdd) < 1e4


def test_controller_stays_finite_on_a_third_robot_different_manufacturer():
    model = RigidBodyModel(os.path.join(FIXTURES, "panda.urdf"))
    q = jnp.array([-0.19360032, 0.89941805, -0.27271074, -1.15251618,
                   -2.92540148, 2.89498581, 2.73584013, 0.0, 0.0])
    qd = jnp.zeros(9)
    p_des = jnp.array([-0.2295431, -0.47162945, 1.00885136])
    pd_des = jnp.zeros(3)
    pdd_des = jnp.zeros(3)

    qdd, tau, mu, h = solve_control_qp(model, "panda_hand", q, qd, p_des, pd_des, pdd_des, eps=0.03)

    assert np.all(np.isfinite(qdd))
    assert np.all(np.isfinite(tau))
    assert np.linalg.norm(qdd) < 1e4


def test_manipulability_zero_at_a_true_singularity():
    model = RigidBodyModel(os.path.join(FIXTURES, "GEN3_URDF_V12.urdf"))
    q_straight = jnp.zeros(model.n)
    mu = float(manipulability(model.link_jacobian(q_straight, "end_effector_link")))
    assert mu < 1e-6


def test_osqp_infeasibility_fallback_path_is_exercised():
    """
    Deterministic test of the infeasibility-fallback branch itself: forces
    OSQP to report the first solve as not-solved (regardless of what it
    actually computed), verifying solve_control_qp genuinely re-solves with
    only the hard CBF constraint and still returns a finite, valid result --
    without depending on a real robot state that may or may not trigger
    genuine infeasibility on a given OSQP version/environment (see the razor
    edge documented above).
    """
    real_osqp_cls = pcc.osqp.OSQP
    calls = {"n": 0}

    class ForcedInfeasibleFirstCallOSQP:
        def __init__(self):
            self._real = real_osqp_cls()

        def setup(self, *args, **kwargs):
            self._real.setup(*args, **kwargs)

        def solve(self):
            calls["n"] += 1
            res = self._real.solve()
            if calls["n"] == 1:
                res.info.status_val = -999
            return res

    model = RigidBodyModel(os.path.join(FIXTURES, "GEN3_URDF_V12.urdf"))
    q = jnp.array([0.3, -0.6, 0.2, -1.1, 0.4, 0.8, -0.3])
    qd = jnp.array([0.4, -0.2, 0.5, 0.1, -0.3, 0.2, 0.6])
    p_des = jnp.array([0.5, 0.1, 0.9])
    pd_des = jnp.zeros(3)
    pdd_des = jnp.zeros(3)

    with patch.object(pcc.osqp, "OSQP", ForcedInfeasibleFirstCallOSQP):
        qdd, tau, mu, h = solve_control_qp(model, "end_effector_link", q, qd, p_des, pd_des, pdd_des, eps=0.03)

    assert calls["n"] == 2, "expected exactly one fallback re-solve after the forced infeasibility"
    assert np.all(np.isfinite(qdd))
    assert np.all(np.isfinite(tau))
