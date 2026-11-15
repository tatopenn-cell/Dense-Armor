"""Tests for core/graph/objects.py."""

from dense_armor.core.graph.grid import Grid
from dense_armor.core.graph.objects import extract_objects


def test_two_separated_objects():
    g = Grid.from_list([[1, 0, 2], [0, 0, 0], [0, 0, 0]])
    objs = extract_objects(g)
    colors = sorted(o.color for o in objs)
    assert colors.count(1) == 1
    assert colors.count(2) == 1


def test_same_color_components():
    g = Grid.from_list([[1, 1, 0], [1, 0, 0], [0, 0, 1]])
    objs = extract_objects(g, by_color=True)
    ones = [o for o in objs if o.color == 1]
    assert len(ones) == 2


def test_adjacency_4_vs_8():
    g = Grid.from_list([[1, 0], [0, 1]])
    objs4 = extract_objects(g, connectivity=4, by_color=True)
    objs8 = extract_objects(g, connectivity=8, by_color=True)
    assert len([o for o in objs4 if o.color == 1]) == 2
    assert len([o for o in objs8 if o.color == 1]) == 1


def test_area_and_bbox():
    g = Grid.from_list([[1, 1, 0], [1, 0, 0], [0, 0, 0]])
    ones = [o for o in extract_objects(g, by_color=True) if o.color == 1]
    assert len(ones) == 1
    assert ones[0].area == 3
    assert ones[0].bbox == (0, 0, 2, 2)


def test_touches_border():
    g = Grid.from_list([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    o = [x for x in extract_objects(g) if x.color == 1][0]
    assert o.touches_border((3, 3))


def test_is_rectangular():
    g = Grid.from_list([[1, 1], [1, 0]])
    o = [x for x in extract_objects(g) if x.color == 1][0]
    assert not o.is_rectangular()


def test_holes():
    g = Grid.from_list([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    ones = [o for o in extract_objects(g) if o.color == 1]
    assert len(ones) == 1
    holes = ones[0].holes(g)
    assert len(holes) == 1
    assert holes[0] == frozenset({(1, 1)})
