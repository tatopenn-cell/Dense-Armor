"""The two CBF controllers on a six-joint arm: every branch of the QP, with the solver forced.

The tests of the real robots (Kinova, Panda) are marked ``heavy``. These run on the six-joint
Kinova fixture, on a copy of it without limits and on a two-joint arm, so the branches of
``solve_control_qp`` are reached in the fast tests too.
"""

import re
from pathlib import Path
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

import dense_armor.dynamics.passivity_cbf_controller as pcc
import dense_armor.dynamics.six_dof_pbc_cbf_controller as spc
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "urdf" / "GEN3-6DOF.urdf"
LINK = "end_effector_link"

TWO_JOINTS = """<?xml version="1.0"?>
<robot name="two">
  <link name="base"/>
  <link name="a"><inertial><origin xyz="0 0 0.1"/><mass value="1"/>
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>
  <link name="b"><inertial><origin xyz="0 0 0.1"/><mass value="1"/>
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>
  <joint name="j1" type="revolute"><parent link="base"/><child link="a"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
  <joint name="j2" type="revolute"><parent link="a"/><child link="b"/>
    <origin xyz="0 0 0.2"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
</robot>
"""


@pytest.fixture(scope="module")
def limited():
    return RigidBodyModel(str(FIXTURE))


@pytest.fixture(scope="module")
def free(tmp_path_factory):
    text = FIXTURE.read_text(encoding="utf-8")
    text = re.sub(r"<limit[^>]*/>", "", text)
    path = tmp_path_factory.mktemp("urdf") / "free.urdf"
    path.write_text(text, encoding="utf-8")
    return RigidBodyModel(str(path))


@pytest.fixture(scope="module")
def two(tmp_path_factory):
    path = tmp_path_factory.mktemp("urdf2") / "two.urdf"
    path.write_text(TWO_JOINTS, encoding="utf-8")
    return RigidBodyModel(str(path))


Q = jnp.array([0.3, 0.6, -0.4, 0.5, 0.8, -0.3])
QD = jnp.array([0.1, -0.1, 0.1, 0.05, -0.1, 0.1])


def _fake_osqp(real, infeasible):
    """An OSQP class whose solves listed in ``infeasible`` (0-based) report failure."""
    calls = []

    class Fake:
        def __init__(self):
            self._real = real()

        def setup(self, *a, **k):
            self._real.setup(*a, **k)

        def solve(self):
            res = self._real.solve()
            if len(calls) in infeasible:
                res.info.status_val = -999
            calls.append(1)
            return res

    Fake.calls = calls
    return Fake


def _passivity(model, q=Q, qd=QD, **kw):
    p = model.link_position(q, LINK) + jnp.array([0.02, 0.0, 0.03])
    z = jnp.zeros(3)
    return pcc.solve_control_qp(model, LINK, q, qd, p, z, z, eps=0.0, **kw)


def _six_dof(model, q=Q, qd=QD, **kw):
    p, r = model.link_pose(q, LINK)
    z = jnp.zeros(3)
    return spc.solve_control_qp(model, LINK, q, qd, p + jnp.array([0.02, 0.0, 0.03]), z, z, r, z, z, eps=0.0, **kw)


def _forced(module, infeasible):
    return patch.object(module.osqp, "OSQP", _fake_osqp(module.osqp.OSQP, infeasible))


def test_manipulability_is_the_volume_of_the_jacobian_ellipsoid():
    j = jnp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    assert float(pcc.manipulability(j)) == pytest.approx(2.0)
    assert float(spc.manipulability(j)) == pytest.approx(2.0)
    assert float(pcc.manipulability(jnp.zeros((2, 3)))) == 0.0


def test_rotation_error_is_zero_for_equal_attitudes_and_signed_otherwise():
    eye = jnp.eye(3)
    assert np.allclose(np.asarray(spc.rotation_error(eye, eye)), 0.0)
    c, s = np.cos(0.4), np.sin(0.4)
    rz = jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    e = np.asarray(spc.rotation_error(rz, eye))
    assert e[2] == pytest.approx(np.sin(0.4))
    assert np.allclose(e[:2], 0.0)
    assert np.allclose(np.asarray(spc.rotation_error(eye, rz)), -e)


@pytest.mark.parametrize("solve", [_passivity, _six_dof])
def test_a_feasible_tick_returns_a_finite_command(limited, solve):
    qdd, tau, mu, h = solve(limited)
    assert qdd.shape == tau.shape == (6,)
    assert np.all(np.isfinite(qdd)) and np.all(np.isfinite(tau))
    assert mu > 0.0
    assert h == pytest.approx(mu)


@pytest.mark.parametrize("solve", [_passivity, _six_dof])
def test_without_limits_the_box_rows_are_not_added(free, solve):
    qdd, tau, mu, h = solve(free)
    assert np.all(np.isfinite(qdd)) and np.all(np.isfinite(tau))
    assert mu > 0.0


@pytest.mark.parametrize("solve,module", [(_passivity, pcc), (_six_dof, spc)])
@pytest.mark.parametrize("model_name", ["limited", "free"])
def test_the_passivity_row_is_dropped_when_the_first_solve_fails(request, solve, module, model_name):
    model = request.getfixturevalue(model_name)
    with _forced(module, {0}) as fake:
        qdd, tau, _, _ = solve(model)
        calls = len(module.osqp.OSQP.calls)
    assert calls == 2
    assert np.all(np.isfinite(qdd)) and np.all(np.isfinite(tau))


@pytest.mark.parametrize("solve,module", [(_passivity, pcc), (_six_dof, spc)])
def test_only_the_cbf_is_kept_when_the_box_is_also_infeasible(limited, solve, module):
    with _forced(module, {0, 1}):
        qdd, tau, _, _ = solve(limited)
        calls = len(module.osqp.OSQP.calls)
    assert calls == 3
    assert np.all(np.isfinite(qdd)) and np.all(np.isfinite(tau))


@pytest.mark.parametrize("solve,module", [(_passivity, pcc), (_six_dof, spc)])
@pytest.mark.parametrize("model_name,failing", [("limited", {0, 1, 2}), ("free", {0, 1})])
def test_an_unsolved_cbf_problem_raises(request, solve, module, model_name, failing):
    model = request.getfixturevalue(model_name)
    with _forced(module, failing):
        with pytest.raises(RuntimeError, match="control QP not solved"):
            solve(model)


def test_passivity_needs_three_joints(two):
    z = jnp.zeros(3)
    with pytest.raises(ValueError, match="model.n >= 3"):
        pcc.solve_control_qp(two, "b", jnp.zeros(2), jnp.zeros(2), z, z, z)


def test_six_dof_needs_six_joints(two):
    z = jnp.zeros(3)
    with pytest.raises(ValueError, match="model.n >= 6"):
        spc.solve_control_qp(two, "b", jnp.zeros(2), jnp.zeros(2), z, z, z, jnp.eye(3), z, z)
