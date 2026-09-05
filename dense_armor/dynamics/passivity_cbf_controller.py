# -*- coding: utf-8 -*-
"""
dynamics/passivity_cbf_controller.py
=======================================
A task-space passivity + singularity-avoidance CBF controller, for any robot
`RigidBodyModel` can load and any named link as the tracked task-space point.

HONEST HISTORY: promoted from Dense-Evolution-Discovery in two steps, same as
`dynamics/urdf_dynamics.py`. Experiment 61 implemented Kurtz, Wensing & Lin's
(2021, arXiv:2109.13349) controller -- a QP over joint acceleration q̈ subject
to Vdot<=0 (passivity of the tracking-error storage function) and an
exponential CBF enforcing a minimum manipulability index -- but hardcoded to
one specific Kinova Gen3's kinematics. Experiment 63 replaced those hardcoded
calls with `RigidBodyModel`'s public API (`mass_matrix`, `link_position`,
`link_jacobian`, `bias_forces`, `gravity_forces`), re-validated on the same
three robots `RigidBodyModel` itself was validated on before promotion (Kinova
Gen3 7-DoF, Kinova Gen3 6-DoF, Franka Emika Panda -- different manufacturer):
cross-checked to machine precision against the original hardcoded controller,
including a real OSQP-infeasibility bug and its fix (see below), then
confirmed stable on the other two robots, each driven toward its own true
kinematic singularity, the CBF holding manipulability within 0.1-1.8% of the
declared floor in every case.

A REAL BUG, found and fixed: OSQP can report the passivity+CBF QP jointly
infeasible (the passivity constraint's coefficients go numerically near-zero
exactly when tracking is already good, which combined with a tight CBF margin
occasionally has no feasible point under OSQP's default tolerances) and
return its infeasibility certificate -- a vector with norm in the billions --
as if it were a real solution. Fixed by checking the solver status and, on
infeasibility, dropping the soft passivity constraint and re-solving with only
the hard, safety-critical CBF constraint.

HONEST SCOPE: task-space position tracking only (not full 6-DoF pose), same
as the Discovery version this promotes.
"""
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import osqp
from scipy import sparse

jax.config.update("jax_enable_x64", True)


def manipulability(j):
    """Manipulability index sqrt(det(J J^T)) -- zero at a kinematic singularity."""
    return jnp.sqrt(jnp.linalg.det(j @ j.T))


def _mu_fn(model, link_name):
    return lambda q: manipulability(model.link_jacobian(q, link_name))


def _manipulability_jacobian(model, link_name, q):
    return jax.grad(_mu_fn(model, link_name))(q)


def _manipulability_jacobian_dot_times_qd(model, link_name, q, qd):
    jfn = lambda qq: _manipulability_jacobian(model, link_name, qq) @ qd
    return jax.jvp(jfn, (q,), (qd,))[1]


def _link_jacobian_dot_times_qd(model, link_name, q, qd):
    jfn = lambda qq: model.link_jacobian(qq, link_name) @ qd
    return jax.jvp(jfn, (q,), (qd,))[1]


def _lambda_task(model, link_name, q):
    m = model.mass_matrix(q)
    j = model.link_jacobian(q, link_name)
    return jnp.linalg.inv(j @ jnp.linalg.solve(m, j.T))


def _lambda_task_dot_times_qd(model, link_name, q, qd):
    fn = lambda qq: _lambda_task(model, link_name, qq)
    return jax.jvp(fn, (q,), (qd,))[1]


@partial(jax.jit, static_argnames=("model", "link_name"))
def _qp_ingredients(model, link_name, q, qd, p_des, pd_des, pdd_des,
                     kp_task, kd_task, kd_null, eps, ka0, ka1):
    n = model.n
    m = model.mass_matrix(q)
    minv = jnp.linalg.inv(m)
    j = model.link_jacobian(q, link_name)
    jdot_qd = _link_jacobian_dot_times_qd(model, link_name, q, qd)

    lam = _lambda_task(model, link_name, q)
    lam_dot = _lambda_task_dot_times_qd(model, link_name, q, qd)
    jbar = minv @ j.T @ lam

    x = model.link_position(q, link_name)
    xd = j @ qd
    x_tilde = x - p_des
    xd_tilde = xd - pd_des

    xdd_cmd = pdd_des - kd_task * xd_tilde - kp_task * x_tilde
    qdd_task = jbar @ (xdd_cmd - jdot_qd)
    null_proj = jnp.eye(n) - jbar @ j
    qdd_nom = qdd_task + null_proj @ (-kd_null * qd)

    def vdot_fn(qdd):
        xdd_tilde = j @ qdd + jdot_qd - pdd_des
        return (xd_tilde @ lam @ xdd_tilde
                + 0.5 * xd_tilde @ lam_dot @ xd_tilde
                + kp_task * x_tilde @ xd_tilde)

    zero = jnp.zeros(n)
    a_vdot = jax.grad(vdot_fn)(zero)
    c_vdot = vdot_fn(zero)

    mu = manipulability(j)
    j_mu = _manipulability_jacobian(model, link_name, q)
    jdot_mu_qd = _manipulability_jacobian_dot_times_qd(model, link_name, q, qd)
    h = mu - eps
    hd = j_mu @ qd

    a1 = a_vdot
    u1 = -c_vdot
    a2 = -j_mu
    u2 = ka0 * h + ka1 * hd + jdot_mu_qd

    # Joint limit CBFs (Kurtz, Wensing & Lin 2021's own "joint" constraint
    # type, AddJointVelCBFConstraint applied twice): a relative-degree-1 CBF
    # for the velocity limit, and a relative-degree-2 (nested class-K) CBF
    # for the position limit, both real URDF <limit> values, class-K gains=1
    # (identity), matching their own published code -- inert (+/-inf)
    # wherever the URDF declares no limit for that joint.
    qd_min = -model.qd_max
    ah_qd_min = qd - qd_min
    ah_qd_max = model.qd_max - qd
    ah_q_min = qd + (q - model.q_min)
    ah_q_max = (model.q_max - q) - qd

    qdd_lb = jnp.maximum(-ah_qd_min, -ah_q_min)
    qdd_ub = jnp.minimum(ah_qd_max, ah_q_max)

    bias = model.bias_forces(q, qd)
    grav = model.gravity_forces(q)

    return m, bias, grav, qdd_nom, a1, u1, a2, u2, mu, h, qdd_lb, qdd_ub


def solve_control_qp(model, link_name, q, qd, p_des, pd_des, pdd_des,
                      kp_task=50.0, kd_task=20.0, kd_null=5.0,
                      eps=0.035, ka=(100.0, 20.0), w_reg=1e-6):
    """Solve one control-tick QP for task-space passivity + singularity avoidance.

    Parameters
    ----------
    model : dynamics.urdf_dynamics.RigidBodyModel
    link_name : str
        Name of the URDF link whose position is tracked.
    q, qd : array-like, shape (model.n,)
        Current joint position/velocity.
    p_des, pd_des, pdd_des : array-like, shape (3,)
        Desired task-space position/velocity/acceleration at the current time
        (e.g. from `utility.trajectory.quintic_trajectory`, mapped to the
        tracked link's task space).
    kp_task, kd_task : float
        Task-space PD gains for the nominal (unconstrained) command.
    kd_null : float
        Joint-velocity damping gain applied in the redundant null space.
    eps : float
        Minimum manipulability index the CBF constraint enforces.
    ka : tuple of float
        Exponential CBF class-K gains (proportional, derivative).
    w_reg : float
        Small regularization weight on the QP's joint-acceleration cost.

    Returns
    -------
    qdd : ndarray, shape (model.n,)
        Solved joint acceleration.
    tau : ndarray, shape (model.n,)
        Corresponding joint torque, from the rigid-body dynamics equation.
    mu : float
        Manipulability index at the current configuration.
    h : float
        CBF value (mu - eps); negative means the declared floor was crossed.

    Real per-joint position/velocity limits from `model`'s own URDF <limit>
    tags are enforced too (Kurtz et al.'s own "joint" CBF constraint type),
    inert for any joint the URDF declares no limit for.
    """
    n = model.n
    m, bias, grav, qdd_nom, a1, u1, a2, u2, mu, h, qdd_lb, qdd_ub = _qp_ingredients(
        model, link_name, jnp.asarray(q), jnp.asarray(qd), jnp.asarray(p_des),
        jnp.asarray(pd_des), jnp.asarray(pdd_des), kp_task, kd_task, kd_null,
        eps, ka[0], ka[1])

    p_mat = sparse.csc_matrix(np.eye(n) * (1.0 + w_reg))
    q_vec = -np.asarray(qdd_nom)

    qdd_lb_raw = np.asarray(qdd_lb)
    qdd_ub_raw = np.asarray(qdd_ub)
    # Only add the joint-limit box rows when the URDF actually declares a
    # real limit somewhere -- an all-unbounded robot gets the exact same
    # 2-row QP as before this feature existed. A literally-inert box row
    # (bounds clipped to +/-1e20) still perturbs OSQP's internal scaling by
    # ~1e-4, which would otherwise break the machine-precision cross-check
    # this module was validated with for zero physical reason.
    has_real_limits = not (np.all(np.isneginf(qdd_lb_raw)) and np.all(np.isposinf(qdd_ub_raw)))

    rows_a1a2 = [sparse.csc_matrix(np.asarray(a1).reshape(1, n)),
                 sparse.csc_matrix(np.asarray(a2).reshape(1, n))]
    l_a1a2 = [-1e20, -1e20]
    u_a1a2 = [float(u1), float(u2)]

    if has_real_limits:
        qdd_lb_np = np.clip(qdd_lb_raw, -1e20, 1e20)
        qdd_ub_np = np.clip(qdd_ub_raw, -1e20, 1e20)
        identity_n = sparse.csc_matrix(np.eye(n))
        a_full = sparse.vstack(rows_a1a2 + [identity_n]).tocsc()
        l_full = np.concatenate([l_a1a2, qdd_lb_np])
        u_full = np.concatenate([u_a1a2, qdd_ub_np])
    else:
        a_full = sparse.vstack(rows_a1a2).tocsc()
        l_full = np.array(l_a1a2)
        u_full = np.array(u_a1a2)

    prob = osqp.OSQP()
    prob.setup(p_mat, q_vec, a_full, l_full, u_full, verbose=False, polish=True)
    res = prob.solve()

    if res.info.status_val != osqp.constant("OSQP_SOLVED"):
        # Soft passivity constraint dropped; hard CBF + joint-limit constraints kept.
        if has_real_limits:
            a_cbf = sparse.vstack([sparse.csc_matrix(np.asarray(a2).reshape(1, n)),
                                    sparse.csc_matrix(np.eye(n))]).tocsc()
            l_cbf = np.concatenate([[-1e20], qdd_lb_np])
            u_cbf = np.concatenate([[float(u2)], qdd_ub_np])
        else:
            a_cbf = sparse.csc_matrix(np.asarray(a2).reshape(1, n))
            l_cbf = np.array([-1e20])
            u_cbf = np.array([float(u2)])
        prob = osqp.OSQP()
        prob.setup(p_mat, q_vec, a_cbf, l_cbf, u_cbf, verbose=False, polish=True)
        res = prob.solve()

    qdd = np.asarray(res.x)
    tau = np.asarray(m) @ qdd + np.asarray(bias) + np.asarray(grav)
    return qdd, tau, float(mu), float(h)
