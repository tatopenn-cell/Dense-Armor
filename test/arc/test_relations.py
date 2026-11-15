"""Tests for core/graph/relations.py."""

from dense_armor.core.graph import relations as R
from dense_armor.core.graph.grid import Grid
from dense_armor.core.graph.objects import extract_objects


def _objects(rows):
    return extract_objects(Grid.from_list(rows))


def test_adjacent_positive():
    objs = _objects([[1, 2, 0], [0, 0, 0], [0, 0, 0]])
    a = [o for o in objs if o.color == 1][0]
    b = [o for o in objs if o.color == 2][0]
    assert R.adjacent(a, b)


def test_adjacent_negative():
    objs = _objects([[1, 0, 2], [0, 0, 0], [0, 0, 0]])
    a = [o for o in objs if o.color == 1][0]
    b = [o for o in objs if o.color == 2][0]
    assert not R.adjacent(a, b)


def test_inside():
    objs = _objects([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2]])
    a = [o for o in objs if o.color == 1][0]
    b = [o for o in objs if o.color == 2][0]
    assert not R.inside(a, b)


def test_same_shape():
    objs = _objects([[1, 1, 0, 2, 2], [0, 0, 0, 0, 0]])
    a = [o for o in objs if o.color == 1][0]
    b = [o for o in objs if o.color == 2][0]
    assert R.same_shape(a, b)


def test_same_color():
    objs = _objects([[1, 0, 1], [0, 0, 0]])
    ones = [o for o in objs if o.color == 1]
    assert len(ones) == 2
    assert R.same_color(ones[0], ones[1])


def test_distance():
    objs = _objects([[1, 0, 0, 2]])
    a = [o for o in objs if o.color == 1][0]
    b = [o for o in objs if o.color == 2][0]
    assert R.distance(a, b) == 3
