"""Drift of RigidBodyModel.forward_dynamics in float32 against float64.

The public methods of RigidBodyModel run in float64 (a local x64 context). The
float32 run uses a subclass whose methods are the undecorated ones, under
``jax.enable_x64(False)``, with float32 inputs. Both runs integrate the same
joint-space PD controller with gravity compensation from the same state.
"""

import os
from pathlib import Path

import jax
import numpy as np
import pytest

from dense_armor.dynamics.urdf_dynamics import RigidBodyModel

URDF = str(Path(__file__).resolve().parents[1] / "fixtures" / "urdf" / "panda.urdf")
STEPS = int(os.environ.get("URDF_DRIFT_STEPS", "10000"))
DT = 1e-3
THRESHOLD = 1e-6
NAMES = (
    "forward_dynamics mass_matrix bias_forces gravity_forces forward_kinematics link_jacobian "
    "link_pose link_position link_spatial_jacobian kinetic_energy potential_energy total_energy "
    "com_positions"
).split()


class _Float32Model(RigidBodyModel):
    pass


for _name in NAMES:
    setattr(_Float32Model, _name, getattr(RigidBodyModel, _name).__wrapped__)


def _run(model, dtype, steps):
    n = model.n
    q_target = np.linspace(0.2, 0.5, n).astype(dtype)
    q = np.zeros(n, dtype=dtype)
    qd = np.zeros(n, dtype=dtype)
    kp, kd = dtype(50.0), dtype(10.0)
    traj = []
    first_qdd = None
    for _ in range(steps):
        tau = np.asarray(model.gravity_forces(q), dtype=dtype) + kp * (q_target - q) - kd * qd
        qdd = np.asarray(model.forward_dynamics(q, qd, tau), dtype=dtype)
        if first_qdd is None:
            first_qdd = qdd
        qd = qd + dtype(DT) * qdd
        q = q + dtype(DT) * qd
        traj.append(q.copy())
    return np.array(traj, dtype=np.float64), first_qdd.astype(np.float64)


@pytest.mark.heavy
def test_forward_dynamics_float32_drift_is_below_threshold():
    traj64, qdd64 = _run(RigidBodyModel(URDF), np.float64, STEPS)
    with jax.enable_x64(False):
        traj32, qdd32 = _run(_Float32Model(URDF), np.float32, STEPS)
    rel_single = float(np.linalg.norm(qdd32 - qdd64) / np.linalg.norm(qdd64))
    rel_traj = float(
        np.max(np.linalg.norm(traj32 - traj64, axis=1)) / np.max(np.linalg.norm(traj64, axis=1))
    )
    print(f"DRIFT steps={STEPS} single_call={rel_single:.3e} trajectory={rel_traj:.3e}")
    assert rel_single < THRESHOLD, f"single call: {rel_single:.3e}"
    assert rel_traj < THRESHOLD, f"trajectory over {STEPS} steps: {rel_traj:.3e}"
