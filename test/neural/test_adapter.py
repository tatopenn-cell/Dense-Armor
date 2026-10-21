"""Tests for AsTransformer."""

from __future__ import annotations

import pickle

import jax
import jax.numpy as jnp

from dense_armor.checks import check_estimator
from dense_armor.utility.neural.adapter import AsTransformer
from dense_armor.utility.neural.layers import Dense
from dense_armor.utility.neural.norm import BatchNorm, LayerNorm


def _make():
    return AsTransformer(
        Dense(3, 2, jax.random.PRNGKey(0)),
        ["a", "b", "c"],
        prefix="z",
    )


def test_check_estimator_passes_with_empty_skips():
    est = _make()
    out = check_estimator(est)
    assert len(out["run"]) > 0
    assert out["not_applicable"] == []
    assert est._unit_test_skips() == set()


def test_output_matches_forward_on_the_same_vector():
    est = _make()
    x = {"a": 1.0, "b": -2.0, "c": 0.5}
    out = est.transform_one(x)
    vec = jnp.array([1.0, -2.0, 0.5], dtype=jnp.float32)
    y = est.layer.forward(vec, training=False)
    for i in range(2):
        assert abs(out[f"z{i}"] - float(y[i])) < 1e-6


def test_missing_key_nan_none_and_empty_all_return_empty():
    est = _make()
    assert est.transform_one({"a": 1.0}) == {}
    assert est.transform_one({"a": 1.0, "b": float("nan"), "c": 0.0}) == {}
    assert est.transform_one({"a": 1.0, "b": None, "c": 0.0}) == {}
    assert est.transform_one({}) == {}
    assert est.n_missing == 4


def test_sum_of_two_transformers_unions_keys():
    a = _make()
    b = AsTransformer(
        Dense(3, 2, jax.random.PRNGKey(1)),
        ["a", "b", "c"],
        prefix="w",
    )
    both = a + b
    out = both.transform_one({"a": 1.0, "b": 0.0, "c": 0.0})
    assert set(out) == {"z0", "z1", "w0", "w1"}


def test_layernorm_through_the_adapter():
    a = AsTransformer(LayerNorm(3), ["a", "b", "c"])
    out = a.transform_one({"a": 1.0, "b": 2.0, "c": 3.0})
    assert len(out) == 3


def test_batchnorm_eval_through_the_adapter():
    a = AsTransformer(BatchNorm(3), ["a", "b", "c"])
    out = a.transform_one({"a": 1.0, "b": 2.0, "c": 3.0})
    assert len(out) == 3


def test_clone_and_pickle_roundtrip():
    est = _make()
    c = est.clone()
    assert type(c) is type(est)
    assert repr(c) == repr(est)
    assert c.get_params() == est.get_params()
    p = pickle.loads(pickle.dumps(est))
    assert repr(p) == repr(est)
    x = {"a": 1.0, "b": 0.0, "c": 0.0}
    assert est.transform_one(x) == c.transform_one(x)
    assert est.transform_one(x) == p.transform_one(x)


def test_learn_one_changes_nothing():
    est = _make()
    before = repr(est)
    est.learn_one({"a": 1.0, "b": 0.0, "c": 0.0}, 1)
    assert repr(est) == before


def test_budget_s_is_at_least_one_millisecond():
    assert AsTransformer.budget_s >= 1e-3
    assert AsTransformer.memory_class == "O(1)"
