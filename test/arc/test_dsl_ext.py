"""Tests for the object primitives in dense_armor.core.graph.dsl."""

import pytest

from dense_armor.core.graph import dsl
from dense_armor.core.graph.grid import Grid


def _g():
    return Grid.from_list(
        [
            [1, 1, 0, 2],
            [1, 0, 0, 0],
            [0, 0, 3, 3],
            [0, 0, 3, 3],
        ]
    )


def test_fill_objects_all():
    out = dsl.fill_objects(_g(), 7)
    assert out.to_list() == [[7, 7, 0, 7], [7, 0, 0, 0], [0, 0, 7, 7], [0, 0, 7, 7]]


def test_fill_objects_predicate():
    out = dsl.fill_objects(_g(), 9, lambda o: o.area == 1)
    assert out.at(0, 3) == 9
    assert out.at(0, 0) == 1


def test_recolor_objects():
    out = dsl.recolor_objects(_g(), {1: 8, 2: 8, 3: 8})
    assert out.to_list() == [[8, 8, 0, 8], [8, 0, 0, 0], [0, 0, 8, 8], [0, 0, 8, 8]]


def test_recolor_objects_subset():
    out = dsl.recolor_objects(_g(), {3: 5})
    assert out.at(2, 2) == 5
    assert out.at(0, 0) == 1


def test_move_objects_down_one():
    out = dsl.move_objects(_g(), 1, 0)
    assert out.to_list() == [[0, 0, 0, 0], [1, 1, 0, 2], [1, 0, 0, 0], [0, 0, 3, 3]]


def test_delete_objects_size1():
    out = dsl.delete_objects(_g(), lambda o: o.area == 1)
    assert out.at(0, 3) == 0
    assert out.at(0, 0) == 1


def test_crop_to_object():
    g = Grid.from_list(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 7, 7, 0],
            [0, 0, 7, 7, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    out = dsl.crop_to_object(g, lambda o: o.color == 7)
    assert out.to_list() == [[7, 7], [7, 7]]


def test_crop_to_object_raises_when_none_match():
    with pytest.raises(ValueError):
        dsl.crop_to_object(_g(), lambda o: o.color == 9)


def test_apply_per_object_identity():
    g = _g()
    out = dsl.apply_per_object(g, lambda o: [(r, c, g.at(r, c)) for r, c in o.cells])
    assert out == g


def test_apply_per_object_recolor():
    g = Grid.from_list([[1, 0], [0, 2]])
    out = dsl.apply_per_object(
        g,
        lambda o: [(r, c, 5 if o.color == 1 else o.color) for r, c in o.cells],
    )
    assert out.to_list() == [[5, 0], [0, 2]]


def test_default_primitives_returns_non_empty():
    prims = dsl.default_primitives()
    assert len(prims) >= 10
    for p in prims:
        assert callable(p)
        assert hasattr(p, "__name__")
