"""Tests for core/graph/grid_graph.py."""

from dense_armor.core.graph.grid import Grid
from dense_armor.core.graph.grid_graph import build_grid_graph


def test_nodes_for_every_cell_and_object():
    g = Grid.from_list([[1, 0], [0, 0]])
    graph = build_grid_graph(g)
    cell_names = [
        n for n in graph.graph.nodes if graph.graph.nodes[n]["type"] == "cell"
    ]
    obj_names = [
        n for n in graph.graph.nodes if graph.graph.nodes[n]["type"] == "object"
    ]
    assert len(cell_names) == 4
    assert len(obj_names) >= 2


def test_graph_validates_on_clean_grid():
    g = Grid.from_list([[1, 1, 0], [0, 0, 0]])
    graph = build_grid_graph(g)
    assert graph.validate() == []


def test_graph_has_belongs_to_edges():
    g = Grid.from_list([[1, 0], [0, 0]])
    graph = build_grid_graph(g)
    has_belongs = any(
        d.get("relation") == "belongs_to"
        for _u, _v, _k, d in graph.graph.edges(keys=True, data=True)
    )
    assert has_belongs


def test_has_hole_is_computed():
    grid_hole = Grid.from_list([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    g = build_grid_graph(grid_hole)
    objs = [
        n
        for n in g.graph.nodes
        if g.graph.nodes[n]["type"] == "object"
        and g.graph.nodes[n]["state"]["color"] == 1
    ]
    assert len(objs) == 1
    assert g.graph.nodes[objs[0]]["state"]["has_hole"] is True

    grid_solid = Grid.from_list([[1, 1], [1, 0]])
    g = build_grid_graph(grid_solid)
    objs = [
        n
        for n in g.graph.nodes
        if g.graph.nodes[n]["type"] == "object"
        and g.graph.nodes[n]["state"]["color"] == 1
    ]
    assert len(objs) == 1
    assert g.graph.nodes[objs[0]]["state"]["has_hole"] is False


def test_belongs_to_constraint_is_real():
    g = Grid.from_list([[1, 0], [0, 0]])
    graph = build_grid_graph(g)
    assert graph.validate() == []
    obj_name = next(
        n
        for n in graph.graph.nodes
        if graph.graph.nodes[n]["type"] == "object"
        and graph.graph.nodes[n]["state"]["color"] == 1
    )
    cells = list(graph.graph.nodes[obj_name]["state"]["cells"])
    graph.update(obj_name, cells=tuple(cells[1:]) if len(cells) > 1 else ())
    assert graph.validate(), "togliere una cella deve rompere belongs_to"
