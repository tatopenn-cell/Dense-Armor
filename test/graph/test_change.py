"""Tests for structural_change, with values worked out by hand."""

import math

import pytest

from dense_armor.core.graph import Graph, structural_change


def _before():
    return Graph(4, [(0, 1, 1.0), (1, 2, 2.0), (2, 3, 1.0)])


def _after():
    return Graph(4, [(0, 1, 1.0), (1, 2, 3.0), (1, 3, 1.0)])


def test_edges_added_removed_and_reweighted():
    c = structural_change(_before(), _after())
    assert c["added"] == [(1, 3)]
    assert c["removed"] == [(2, 3)]
    assert c["reweighted"] == [(1, 2, 2.0, 3.0)]


def test_topology_change_is_the_jaccard_distance():
    c = structural_change(_before(), _after())
    assert c["topology_change"] == pytest.approx(1 - 2 / 4)


def test_weight_change_by_hand():
    c = structural_change(_before(), _after())
    diff = math.sqrt(0**2 + 1**2 + 1**2 + 1**2)
    den = math.sqrt(1 + 9 + 1)
    assert c["weight_change"] == pytest.approx(diff / den)


def test_same_graph_has_no_change():
    c = structural_change(_before(), _before())
    assert c["added"] == c["removed"] == c["reweighted"] == []
    assert c["topology_change"] == 0.0
    assert c["weight_change"] == 0.0


def test_edge_order_and_direction_do_not_matter():
    a = Graph(3, [(0, 1, 1.0), (1, 2, 1.0)])
    b = Graph(3, [(2, 1, 1.0), (1, 0, 1.0)])
    assert structural_change(a, b)["topology_change"] == 0.0


def test_no_edges_in_either_graph():
    c = structural_change(Graph(3, []), Graph(3, []))
    assert c["topology_change"] == 0.0 and c["weight_change"] == 0.0


def test_disjoint_edges_are_at_distance_one():
    c = structural_change(Graph(3, [(0, 1, 1.0)]), Graph(3, [(1, 2, 1.0)]))
    assert c["topology_change"] == 1.0


def test_tolerance_hides_a_small_reweighting():
    a = Graph(2, [(0, 1, 1.0)])
    b = Graph(2, [(0, 1, 1.05)])
    assert structural_change(a, b)["reweighted"] != []
    assert structural_change(a, b, tol=0.1)["reweighted"] == []


def test_bad_arguments_are_refused():
    with pytest.raises(ValueError):
        structural_change(Graph(2, []), Graph(3, []))
    with pytest.raises(ValueError):
        structural_change(Graph(2, []), Graph(2, []), tol=-1.0)
