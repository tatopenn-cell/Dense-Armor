"""Tests for the VSA primitives and the encoding of the objects."""

import numpy as np

from dense_armor.core.graph.vsa import (
    SSPSpace,
    bind,
    bundle,
    invert,
    random_unit_vector,
    similarity,
    unbind,
)
from dense_armor.core.graph.vsa_objects import ObjectEncoder

N = 1024


def test_unbind_recovers_what_was_bound():
    rng = np.random.default_rng(1)
    a, b = random_unit_vector(rng, N), random_unit_vector(rng, N)
    assert similarity(unbind(bind(a, b), b), a) > 0.5
    assert similarity(unbind(bind(a, b), a), b) > 0.5
    assert abs(similarity(bind(a, b), a)) < 0.2


def test_bundle_is_similar_to_each_part_and_normalised():
    rng = np.random.default_rng(2)
    a, b = random_unit_vector(rng, N), random_unit_vector(rng, N)
    s = bundle(a, b)
    assert abs(np.linalg.norm(s) - 1.0) < 1e-9
    assert similarity(s, a) > 0.5 and similarity(s, b) > 0.5
    assert np.all(bundle(np.zeros(4), np.zeros(4)) == 0)


def test_invert_is_an_involution():
    rng = np.random.default_rng(3)
    a = random_unit_vector(rng, N)
    assert np.allclose(invert(invert(a)), a)


def test_ssp_similarity_falls_with_distance():
    space = SSPSpace(n=N, length_scale=0.25, seed=0)
    here = space.encode([0.0, 0.0])
    assert abs(similarity(here, space.encode([0.0, 0.0])) - 1.0) < 1e-9
    assert similarity(here, space.encode([0.2, 0.0])) > similarity(
        here, space.encode([3.0, 0.0])
    )
    assert space.encode([[0.0, 0.0], [1.0, 1.0]]).shape == (2, N)


def test_the_encoder_decodes_colour_and_centre():
    enc = ObjectEncoder(n=N)
    cells = [(1, 1), (1, 2), (2, 1), (2, 2)]
    sp = enc.encode_object(cells, 4, 7, 7)
    assert enc.decode_colour(sp) == 4
    assert enc.decode_centre(enc.encode_object([(3, 3)], 2, 7, 7)) == (0, 0)
    assert enc.decode_shape(sp).shape == (N,)


def test_two_objects_of_different_colour_are_less_similar_than_two_of_the_same():
    enc = ObjectEncoder(n=N)
    cells = [(1, 1), (1, 2)]
    same = similarity(
        enc.encode_object(cells, 3, 6, 6), enc.encode_object(cells, 3, 6, 6)
    )
    other = similarity(
        enc.encode_object(cells, 3, 6, 6), enc.encode_object(cells, 8, 6, 6)
    )
    assert same > other
