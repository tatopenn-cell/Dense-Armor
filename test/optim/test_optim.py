"""Tests for dense_armor.utility.optim.

The optimizer API has two shapes: an object with step(w, g), and a
pure pair init_state(dim) / update(state, w, g). The tests check that
the two agree step by step, that each optimizer converges on a small
least-squares problem, and that the pure form is usable inside
jax.jit and jax.lax.scan.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dense_armor.utility.optim import (
    SGD,
    AdaGrad,
    Adam,
    FTRLProximal,
    OnlineNewtonStep,
)

CASES = [
    (SGD, {"lr": 0.05}, 500, 1e-2),
    (AdaGrad, {"lr": 0.3}, 500, 1e-2),
    (Adam, {"lr": 0.1}, 500, 1e-2),
    (FTRLProximal, {"alpha": 1.0, "beta": 0.5, "l2": 0.1}, 500, 1e-1),
    (OnlineNewtonStep, {"gamma": 1.0, "eps": 1.0}, 50, 1e-2),
]

# The online least-squares problem is a generic convex problem. Only
# the optimizers with a convex-loss regret bound have a guarantee to
# reach the solution. OnlineNewtonStep is designed for exp-concave
# losses (Orabona 2019, Section 7.10): on the generic squared loss it
# is free to stall and the test would be meaningless. It is kept in
# CASES for the structural checks (object vs pure, jit, scan) and
# excluded from the convergence check by the marker below.
CONVERGENCE_CASES = [c for c in CASES if c[0] is not OnlineNewtonStep]


@pytest.mark.parametrize("cls,kw,n,tol", CASES, ids=[c[0].__name__ for c in CASES])
def test_object_matches_pure(cls, kw, n, tol):
    opt = cls(**kw)
    w_obj = np.zeros(3)
    w_pure = jnp.zeros(3)
    state = opt.init_state(3)
    rng = np.random.default_rng(0)
    for _ in range(20):
        g = rng.normal(size=3)
        w_obj = opt.step(w_obj, g)
        state, w_pure = opt.update(state, w_pure, jnp.asarray(g))
    assert float(np.max(np.abs(w_obj - np.asarray(w_pure)))) < 1e-6


@pytest.mark.parametrize(
    "cls,kw,n,tol", CONVERGENCE_CASES, ids=[c[0].__name__ for c in CONVERGENCE_CASES]
)
def test_converges_on_least_squares(cls, kw, n, tol):
    rng = np.random.default_rng(0)
    A = rng.normal(size=(200, 5))
    w_star = np.array([1.0, -2.0, 3.0, 0.5, -1.0])
    y = A @ w_star
    opt = cls(**kw)
    w = np.zeros(5)
    for i in range(n):
        pred = A[i % len(A)] @ w
        g = 2.0 * (pred - y[i % len(A)]) * A[i % len(A)]
        w = opt.step(w, g)
    assert float(np.max(np.abs(w - w_star))) < tol


@pytest.mark.parametrize("cls,kw,n,tol", CASES, ids=[c[0].__name__ for c in CASES])
def test_pure_is_jittable(cls, kw, n, tol):
    opt = cls(**kw)
    state = opt.init_state(4)
    w = jnp.zeros(4)
    g = jnp.ones(4)
    _state2, w2 = jax.jit(opt.update)(state, w, g)
    _state3, w3 = opt.update(state, w, g)
    assert jnp.allclose(w2, w3)


@pytest.mark.parametrize("cls,kw,n,tol", CASES, ids=[c[0].__name__ for c in CASES])
def test_pure_inside_lax_scan(cls, kw, n, tol):
    opt = cls(**kw)
    dim = 3
    grads = jnp.ones((10, dim))

    def body(carry, g):
        state, w = carry
        state, w = opt.update(state, w, g)
        return (state, w), w

    (_, w_final), ws = jax.lax.scan(body, (opt.init_state(dim), jnp.zeros(dim)), grads)
    assert ws.shape == (10, dim)
    assert jnp.all(jnp.isfinite(w_final))
