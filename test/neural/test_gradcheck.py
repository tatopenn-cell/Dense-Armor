"""Tests for gradcheck: correct pair, wrong pair, precision, shape."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from dense_armor.utility.neural.gradcheck import gradcheck


def _sq_f(x):
    return x * x


def _sq_b(x, y, g):
    del y
    return g * 2.0 * x


def _bad_b(x, y, g):
    del y
    return g * x


def test_correct_pair_below_threshold():
    x = jnp.array([1.0, 2.0, -3.0, 0.5])
    err = gradcheck(_sq_f, _sq_b, x)
    assert err < 1e-7


def test_wrong_pair_above_threshold():
    x = jnp.array([1.0, 2.0, -3.0, 0.5])
    err = gradcheck(_sq_f, _bad_b, x)
    assert err > 1e-1


def test_dtype_outside_call_is_preserved():
    before = jnp.zeros(1).dtype
    x = jnp.array([1.0, 2.0])
    gradcheck(_sq_f, _sq_b, x)
    after = jnp.zeros(1).dtype
    assert before == after


def test_shape_mismatch_raises():
    def bad_shape_b(x, y, g):
        del y
        return g[:-1] * 2.0 * x[:-1]

    with pytest.raises(ValueError, match="backward shape"):
        gradcheck(_sq_f, bad_shape_b, jnp.array([1.0, 2.0, 3.0]))


def test_x64_is_restored_after_call():
    before = bool(jax.config.jax_enable_x64)
    x = jnp.array([1.0, 2.0])
    gradcheck(_sq_f, _sq_b, x)
    after = bool(jax.config.jax_enable_x64)
    assert before == after
