"""Object form and pure form of glm.py agree with intercept and Adam."""

import jax
import jax.numpy as jnp
import numpy as np

from dense_armor.utility.linear.glm import LinearRegression, LogisticRegression
from dense_armor.utility.optim import Adam


def test_linear_object_matches_pure_with_intercept_and_adam():
    rng = np.random.default_rng(0)
    samples = [
        ({"a": float(rng.normal()), "b": float(rng.normal())}, float(rng.normal()))
        for _ in range(30)
    ]

    reg = LinearRegression(optimizer=Adam(lr=0.05), intercept=True)
    for x, y in samples:
        reg.learn_one(x, y)
    w_obj = reg._w.copy()

    reg_pure = LinearRegression(optimizer=Adam(lr=0.05), intercept=True)
    state = reg_pure.init_state(dim=2)
    for x, y in samples:
        v = {**x, "y": y}
        state, _ = reg_pure.step(state, v)
    w_pure = np.asarray(state[0])

    assert np.max(np.abs(w_obj - w_pure)) < 1e-5


def test_logistic_object_matches_pure_with_intercept_and_adam():
    rng = np.random.default_rng(0)
    samples = [
        (
            {"a": float(rng.normal()), "b": float(rng.normal())},
            1 if rng.normal() > 0 else 0,
        )
        for _ in range(30)
    ]

    clf = LogisticRegression(optimizer=Adam(lr=0.05), intercept=True)
    for x, y in samples:
        clf.learn_one(x, y)
    w_obj = clf._w.copy()

    clf_pure = LogisticRegression(optimizer=Adam(lr=0.05), intercept=True)
    state = clf_pure.init_state(dim=2)
    for x, y in samples:
        v = {**x, "y": float(y)}
        state, _ = clf_pure.step(state, v)
    w_pure = np.asarray(state[0])

    assert np.max(np.abs(w_obj - w_pure)) < 1e-5


def test_pure_and_object_forms_agree_up_to_dtype():
    """The two forms agree; the pure dtype depends on the JAX backend.

    In JAX 32-bit mode (default) the pure form is float32 and the
    object form float64. With X64 enabled both are float64. The test
    checks the agreement on 30 samples, not the specific dtype, so
    it passes in both configurations.
    """
    rng = np.random.default_rng(1)
    samples = [({"a": float(rng.normal())}, float(rng.normal())) for _ in range(30)]

    reg = LinearRegression(optimizer=Adam(lr=0.05), intercept=True)
    for x, y in samples:
        reg.learn_one(x, y)

    reg_pure = LinearRegression(optimizer=Adam(lr=0.05), intercept=True)
    state = reg_pure.init_state(dim=1)
    for x, y in samples:
        state, _ = reg_pure.step(state, {**x, "y": y})

    d_pure = np.asarray(state[0]).dtype
    d_obj = reg._w.dtype
    # pure form is float32 or float64 depending on the JAX config
    assert d_pure in (np.float32, np.float64)
    assert d_obj == np.float64
    assert np.max(np.abs(np.asarray(state[0]) - reg._w)) < 1e-5


def test_pure_form_inside_jit_runs():
    reg = LinearRegression(optimizer=Adam(lr=0.05), intercept=True)
    state = reg.init_state(dim=2)
    v = {"a": 0.5, "b": -0.5, "y": 1.0}

    @jax.jit
    def one_step(state, v):
        return reg.step(state, v)

    new_state, pred = one_step(state, v)
    assert jnp.isfinite(pred)
    assert new_state[0].shape == (3,)
