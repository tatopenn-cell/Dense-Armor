"""Tests for batching.py."""

from __future__ import annotations

import jax
import pytest

from dense_armor.utility.neural.batching import minibatches


def test_no_key_returns_indices_in_order():
    batches = [b.tolist() for b in minibatches(10, 3)]
    assert batches == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


def test_no_key_sizes_are_3_3_3_1():
    sizes = [len(b) for b in minibatches(10, 3)]
    assert sizes == [3, 3, 3, 1]


def test_drop_last_sizes():
    sizes = [len(b) for b in minibatches(10, 3, drop_last=True)]
    assert sizes == [3, 3, 3]


def test_batch_size_one():
    sizes = [len(b) for b in minibatches(5, 1)]
    assert sizes == [1, 1, 1, 1, 1]


def test_batch_size_above_n():
    batches = [b.tolist() for b in minibatches(3, 20)]
    assert batches == [[0, 1, 2]]


def test_key_same_twice():
    k = jax.random.PRNGKey(0)
    b1 = [b.tolist() for b in minibatches(10, 3, key=k)]
    b2 = [b.tolist() for b in minibatches(10, 3, key=k)]
    assert b1 == b2


def test_two_different_keys():
    k1 = jax.random.PRNGKey(0)
    k2 = jax.random.PRNGKey(1)
    b1 = [b.tolist() for b in minibatches(10, 3, key=k1)]
    b2 = [b.tolist() for b in minibatches(10, 3, key=k2)]
    assert b1 != b2


def test_key_covers_all_indices():
    k = jax.random.PRNGKey(7)
    flat = sorted(x for b in minibatches(20, 4, key=k) for x in b.tolist())
    assert flat == list(range(20))


def test_no_key_covers_all_indices():
    flat = sorted(x for b in minibatches(17, 5) for x in b.tolist())
    assert flat == list(range(17))


def test_value_errors():
    with pytest.raises(ValueError, match="n must be at least 1"):
        list(minibatches(0, 3))
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        list(minibatches(10, 0))
    with pytest.raises(ValueError):
        list(minibatches(-1, 3))
    with pytest.raises(ValueError):
        list(minibatches(10, -1))
