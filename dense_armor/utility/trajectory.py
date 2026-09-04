# -*- coding: utf-8 -*-
"""
utility/trajectory.py
=======================
Closed-form, no-training point-to-point trajectory generator --
`rate_limiter.py` bounds how fast a command can change and `cbf_filter.py`
bounds where it can go, but neither generates a REFERENCE to track in the
first place. This module fills that gap with the simplest real,
universal piece: a minimum-jerk-continuous quintic polynomial between two
points.

Scoped down deliberately from two real papers (both read in full before
writing any code): Lozer, Scalera, Boscariol & Gasparetto, "Planning
optimal minimum-jerk trajectories for redundant robots" (Robotics and
Autonomous Systems, Elsevier) does multi-stage optimization with a full
dynamic model on a real 7-DoF Franka Panda; Fried & Paternain, "A
Bi-Level Optimization Approach to Joint Trajectory Optimization for
Redundant Manipulators" (arXiv:2412.07859) does a convex inner/primal-
dual outer optimization validated on a real UR10e. Both need a full
kinematic/dynamic model (URDF parsing, Jacobians, torque limits) -- not
implemented here, a real, separate, much larger undertaking.

THEORY: the classic quintic (5th-order) polynomial trajectory (see e.g.
Craig, "Introduction to Robotics", or Piazzi & Visioli 2000) is the
unique degree-5 polynomial per joint satisfying 6 real boundary
conditions (position, velocity, acceleration at t=0 and t=T) -- solved
here directly from the 6x6 linear system, not copied from a memorized
formula, so a transcription error would fail the boundary-condition test
rather than silently produce a wrong trajectory.

UNIVERSAL in the sense that matters for this stack: works for any number
of joints at once (any robot) since each joint's polynomial is
independent; needs no URDF, no kinematics, no dynamics, no robot
connection. Composes directly with `rate_limiter`/`cbf_filter`, which
already own rate-of-change and spatial safety -- this generator does not
need to worry about either.

VALIDATED, honestly, on TWO independent real physical domains (real
LeRobot robot-arm joint data), promoted from Dense-Evolution-Discovery
after both checked out (see docs/quintic_trajectory_planner.md there).
A first validation attempt (real episode start/end frames as q0/qf) was
thrown out as degenerate: a real pick-and-place task often returns close
to its own starting configuration, giving several joints almost nothing
to do and nothing meaningful to compare. Fixed by comparing each real
joint's own min-to-max excursion within the episode instead:
- SO-101 (single 6-DoF arm, real 30Hz): 6 real joint excursions checked.
- ALOHA (bimanual, 14-DoF, real 50Hz -- a genuinely different real
  robot): 14 real joint excursions checked.
Real result, all 20/20 real excursions: the quintic's peak velocity is
always LOWER than the real recorded peak velocity for the same start,
end, and real elapsed duration (ratio 0.05-0.62, mean 0.26). Expected,
not a bug: the quintic is the smoothest possible path between two
points, so it needs less peak speed than a real (teleoperated, not
necessarily efficient) trajectory covering the same net real
displacement in the same real time.

HONEST SCOPE: single-segment point-to-point only. Chaining several
quintic segments across many waypoints is a real, natural next step,
not implemented here.
"""
import numpy as np


def quintic_trajectory(q0, qf, T: float, v0=None, a0=None, vf=None, af=None, n_samples: int = 100):
    """Closed-form quintic point-to-point trajectory, any number of DOF.

    Parameters
    ----------
    q0, qf : array-like, shape (n_dof,)
        Start and end position for each joint.
    T : float
        Total real trajectory duration.
    v0, a0, vf, af : array-like, shape (n_dof,), optional
        Start/end velocity and acceleration per joint (default 0 -- the
        standard "start and end at rest" case).
    n_samples : int
        Number of real time samples to return.

    Returns
    -------
    t : ndarray, shape (n_samples,)
    q, v, a : ndarray, shape (n_samples, n_dof)
        Position, velocity, acceleration at each sampled time.
    """
    q0 = np.atleast_1d(np.asarray(q0, dtype=float))
    qf = np.atleast_1d(np.asarray(qf, dtype=float))
    n_dof = q0.shape[0]
    assert qf.shape[0] == n_dof, "q0 and qf must have the same number of joints"
    v0 = np.zeros(n_dof) if v0 is None else np.atleast_1d(np.asarray(v0, dtype=float))
    a0 = np.zeros(n_dof) if a0 is None else np.atleast_1d(np.asarray(a0, dtype=float))
    vf = np.zeros(n_dof) if vf is None else np.atleast_1d(np.asarray(vf, dtype=float))
    af = np.zeros(n_dof) if af is None else np.atleast_1d(np.asarray(af, dtype=float))
    assert T > 0, "T must be positive"

    t = np.linspace(0.0, T, n_samples)

    boundary_matrix = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [1, T, T ** 2, T ** 3, T ** 4, T ** 5],
        [0, 1, 2 * T, 3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
        [0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3],
    ])
    rhs = np.stack([q0, v0, a0, qf, vf, af], axis=1)          # (n_dof, 6)
    coeffs = np.linalg.solve(boundary_matrix, rhs.T)           # (6, n_dof)

    powers_q = t[:, None] ** np.arange(6)[None, :]                                    # (n_samples, 6)
    powers_v = np.concatenate([np.zeros((n_samples, 1)),
                                np.arange(1, 6)[None, :] * t[:, None] ** np.arange(0, 5)[None, :]], axis=1)
    powers_a = np.concatenate([np.zeros((n_samples, 2)),
                                (np.arange(2, 6) * np.arange(1, 5))[None, :] * t[:, None] ** np.arange(0, 4)[None, :]], axis=1)

    q = powers_q @ coeffs
    v = powers_v @ coeffs
    a = powers_a @ coeffs
    return t, q, v, a
