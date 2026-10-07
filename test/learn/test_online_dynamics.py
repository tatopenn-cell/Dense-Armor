# -*- coding: utf-8 -*-
"""Unit tests for dense_armor/dynamics/online_dynamics.py."""
import doctest
import importlib
import pickle
import sys
from copy import deepcopy

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import dense_armor.utility.learn.online_dynamics as online_dynamics  # noqa: E402
from dense_armor.utility.learn.online_dynamics import (  # noqa: E402
    RecursiveLeastSquares, ResidualDynamicsLearner,
    DriftAwareResidualDynamicsLearner, write_minimal_urdf,
)
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel  # noqa: E402


def _jit_patch(m):
    m.mass_matrix = jax.jit(m.mass_matrix)
    m.bias_forces = jax.jit(m.bias_forces)
    m.gravity_forces = jax.jit(m.gravity_forces)


def _two_link():
    path_nom = write_minimal_urdf()
    path_pay = write_minimal_urdf(payload_mass=2.0)
    model_nom = RigidBodyModel(path_nom)
    model_true = RigidBodyModel(path_pay)
    _jit_patch(model_nom)
    _jit_patch(model_true)
    z = jnp.zeros(model_nom.n)
    _ = model_nom.mass_matrix(z), model_nom.bias_forces(z, z), model_nom.gravity_forces(z)
    _ = model_true.mass_matrix(z), model_true.bias_forces(z, z), model_true.gravity_forces(z)
    return model_nom, model_true


def test_rls_matches_lstsq_lam1_large_delta():
    rng = np.random.default_rng(0)
    n, d = 200, 5
    X = rng.standard_normal((n, d))
    y = X @ rng.standard_normal(d)
    rls = RecursiveLeastSquares(lam=1.0, delta=1e10,
                                 feature_keys=[f"f{i}" for i in range(d)])
    for i in range(n):
        _ = rls.learn_one({f"f{j}": float(X[i, j]) for j in range(d)},
                          float(y[i]))
    ls, *_ = np.linalg.lstsq(X, y, rcond=None)
    assert np.max(np.abs(rls._w - ls)) < 1e-8


def test_rls_learns_simple_linear_relation():
    rls = RecursiveLeastSquares(lam=1.0, delta=1e10,
                                 feature_keys=["a", "b"])
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]])
    y = X @ np.array([2.0, -3.0])
    for i in range(len(X)):
        _ = rls.learn_one({"a": float(X[i, 0]), "b": float(X[i, 1])},
                          float(y[i]))
    assert np.allclose(rls._w, np.array([2.0, -3.0]), atol=1e-6)


def test_rls_predict_before_learning_is_zero():
    rls = RecursiveLeastSquares(feature_keys=["a"])
    assert rls.predict_one({"a": 1.0}) == 0.0


def test_rls_tracks_changing_parameter():
    rng = np.random.default_rng(1)
    n = 2000
    w1 = np.array([1.0, -1.0])
    w2 = np.array([3.0, 2.0])
    X = rng.standard_normal((n, 2))
    y = np.where(np.arange(n) < 1000, X @ w1, X @ w2)
    rls_forget = RecursiveLeastSquares(lam=0.95, delta=1.0,
                                        feature_keys=["a", "b"])
    rls_noforget = RecursiveLeastSquares(lam=1.0, delta=1.0,
                                         feature_keys=["a", "b"])
    for i in range(n):
        d = {"a": float(X[i, 0]), "b": float(X[i, 1])}
        _ = rls_forget.learn_one(d, float(y[i]))
        _ = rls_noforget.learn_one(d, float(y[i]))
    err_forget = float(np.max(np.abs(rls_forget._w - w2)))
    err_noforget = float(np.max(np.abs(rls_noforget._w - w2)))
    assert err_forget < err_noforget


def test_rls_p_symmetrised():
    rls = RecursiveLeastSquares(lam=0.99, delta=1e3, feature_keys=["a", "b"])
    rng = np.random.default_rng(0)
    for _ in range(50):
        _ = rls.learn_one({"a": float(rng.standard_normal()),
                           "b": float(rng.standard_normal())}, 1.0)
    assert np.max(np.abs(rls._P - rls._P.T)) < 1e-12



def test_rls_clone_pickle_repr():
    m = RecursiveLeastSquares(lam=0.99, delta=1e4, feature_keys=["a", "b"])
    m2 = deepcopy(m)
    assert m2.lam == m.lam and m2.delta == m.delta
    blob = pickle.dumps(m)
    m3 = pickle.loads(blob)
    assert m3.lam == m.lam and m3.feature_keys == m.feature_keys
    assert isinstance(repr(m), str)


def test_doctest():
    assert doctest.testmod(online_dynamics).failed == 0


def test_check_estimator():
    from dense_armor.checks import check_estimator
    check_estimator(RecursiveLeastSquares(lam=1.0, delta=1e6,
                                          feature_keys=["a", "b"]))


def test_residual_learner_reduces_rmse():
    model_nom, model_true = _two_link()
    n_dof = model_nom.n

    def true_tau(q, qd, qdd):
        qj, qdj, qddj = jnp.asarray(q), jnp.asarray(qd), jnp.asarray(qdd)
        return (np.asarray(model_true.mass_matrix(qj) @ qddj
                           + model_true.bias_forces(qj, qdj)
                           + model_true.gravity_forces(qj))
                + 0.5 * qd + 0.1 * np.sign(qd))

    rng = np.random.default_rng(2)
    n = 200
    Q = rng.uniform(-1.2, 1.2, (n, n_dof))
    QD = rng.uniform(-1.0, 1.0, (n, n_dof))
    QDD = rng.uniform(-1.0, 1.0, (n, n_dof))
    TAU = np.array([true_tau(Q[i], QD[i], QDD[i]) for i in range(n)])

    learner = ResidualDynamicsLearner(model_nom, lam=1.0, delta=1e6)
    nom = np.zeros_like(TAU)
    learned = np.zeros_like(TAU)
    for i in range(n):
        nom[i] = np.asarray(model_nom.mass_matrix(jnp.asarray(Q[i])) @ jnp.asarray(QDD[i])
                            + model_nom.bias_forces(jnp.asarray(Q[i]), jnp.asarray(QD[i]))
                            + model_nom.gravity_forces(jnp.asarray(Q[i])))
        learned[i] = learner.predict_torque(Q[i], QD[i], QDD[i])
        _ = learner.learn_one(Q[i], QD[i], QDD[i], TAU[i])

    rmse_nom = np.sqrt(np.mean((nom - TAU) ** 2))
    rmse_learned = np.sqrt(np.mean((learned - TAU) ** 2))
    assert rmse_learned < rmse_nom


def test_drift_aware_runs_and_produces_finite_output():
    model_nom, model_true = _two_link()
    n_dof = model_nom.n

    def true_tau(q, qd, qdd, b_v, c_c):
        qj, qdj, qddj = jnp.asarray(q), jnp.asarray(qd), jnp.asarray(qdd)
        return (np.asarray(model_true.mass_matrix(qj) @ qddj
                           + model_true.bias_forces(qj, qdj)
                           + model_true.gravity_forces(qj))
                + b_v * qd + c_c * np.sign(qd))

    rng = np.random.default_rng(3)
    n = 400
    Q = rng.uniform(-1.2, 1.2, (n, n_dof))
    QD = rng.uniform(-1.0, 1.0, (n, n_dof))
    QDD = rng.uniform(-1.0, 1.0, (n, n_dof))
    TAU = np.zeros((n, n_dof))
    for i in range(n):
        b, c = (0.5, 0.1) if i < 200 else (1.5, 0.3)
        TAU[i] = true_tau(Q[i], QD[i], QDD[i], b, c)

    learner = DriftAwareResidualDynamicsLearner(
        model_nom, lam=0.99, fast_lam=0.9, warmup_steps=30,
        trigger="hampel", hampel_radius=20, hampel_n_sigmas=3.0,
        detector_warmup=150,
    )
    pred = np.zeros((n, n_dof))
    for i in range(n):
        pred[i] = learner.predict_torque(Q[i], QD[i], QDD[i])
        _ = learner.learn_one(Q[i], QD[i], QDD[i], TAU[i])
    assert np.all(np.isfinite(pred))
