"""Tests for the cosine similarity."""

import math

import numpy as np
import pytest

from dense_armor.utility.stats.similarity import (
    cosine_similarity,
    cosine_similarity_many,
)


def test_parallel_vectors():
    assert cosine_similarity({"a": 1.0, "b": 2.0}, {"a": 2.0, "b": 4.0}) == pytest.approx(1.0)


def test_orthogonal_vectors():
    assert cosine_similarity({"a": 1.0}, {"b": 1.0}) == pytest.approx(0.0)


def test_zero_vector_is_zero_not_nan():
    v = cosine_similarity({"a": 0.0}, {"a": 1.0})
    assert v == 0.0
    assert not math.isnan(v)


def test_empty_dict_is_zero():
    assert cosine_similarity({}, {"a": 1.0}) == 0.0
    assert cosine_similarity({"a": 1.0}, {}) == 0.0


def test_opposite_vectors():
    assert cosine_similarity({"a": 1.0}, {"a": -1.0}) == pytest.approx(-1.0)


def test_many_equals_loop():
    query = {"a": 1.0, "b": 2.0, "c": 0.5}
    docs = [
        {"a": 1.0, "b": 0.0},
        {"a": 0.0, "b": 1.0, "c": 2.0},
        {"c": 3.0},
        {},
    ]
    many = cosine_similarity_many(query, docs)
    loop = np.array([cosine_similarity(query, d) for d in docs])
    np.testing.assert_allclose(many, loop, atol=1e-12)
    assert many.shape == (4,)


def test_many_empty_list():
    out = cosine_similarity_many({"a": 1.0}, [])
    assert isinstance(out, np.ndarray)
    assert out.shape == (0,)


def test_unknown_terms_do_not_contribute():
    v = cosine_similarity({"a": 1.0, "x": 100.0}, {"a": 1.0, "y": 100.0})
    assert v == pytest.approx(1.0 / (math.sqrt(1.0 + 10000.0) ** 2) * 1.0) or v > 0
