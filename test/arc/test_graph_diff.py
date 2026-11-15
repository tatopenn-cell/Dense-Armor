"""Tests for graph_diff."""

from dense_armor.core.graph.graph_diff import (
    diff_graphs,
    hungarian_match,
    level2_nodes,
)
from dense_armor.core.graph.grid import Grid
from dense_armor.core.graph.objects import extract_objects


def test_level2_bg():
    g = Grid.from_list([[0, 0], [0, 1]])
    objs = extract_objects(g)
    nodes = level2_nodes(g, objs)
    assert "bg" in nodes
    assert nodes["bg"]["state"]["color"] == 0


def test_level2_frame():
    g = Grid.from_list(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
    )
    objs = extract_objects(g)
    nodes = level2_nodes(g, objs)
    frames = [k for k, v in nodes.items() if v["type"] == "frame"]
    assert len(frames) == 1


def test_level2_alignment():
    g = Grid.from_list(
        [
            [1, 0, 2],
            [0, 0, 0],
        ]
    )
    objs = extract_objects(g)
    nodes = level2_nodes(g, objs)
    aligns = [k for k, v in nodes.items() if v["type"] == "alignment_group"]
    assert len(aligns) >= 1


def test_hungarian_identity():
    g = Grid.from_list([[1, 0], [0, 2]])
    # extract_objects includes background cells as objects too:
    # 4 objects (1, 2, and two 0s). hungarian_match must match all 4.
    objs = extract_objects(g)
    assert len(objs) == 4
    matched, created, deleted = hungarian_match(objs, objs)
    assert len(matched) == 4
    assert created == []
    assert deleted == []


def test_hungarian_delete():
    g1 = Grid.from_list([[1, 0], [0, 2]])
    g2 = Grid.from_list([[1, 0], [0, 0]])
    o1 = extract_objects(g1)
    o2 = extract_objects(g2)
    matched, created, deleted = hungarian_match(o1, o2)
    assert len(matched) == 2  # 1->1, 0->0 (both have color 0, area 1)


def test_diff_graphs_same():
    g = Grid.from_list([[1, 0], [0, 0]])
    d = diff_graphs(g, g)
    assert d["created"] == []
    assert d["deleted"] == []
