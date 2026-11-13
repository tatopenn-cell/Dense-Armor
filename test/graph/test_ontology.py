"""Tests for TypedGraph."""

import pytest

from dense_armor.core.graph import TypedGraph

SCHEMA = {
    "sensor": {"battery": float, "status": str},
    "joint": {"angle": float},
    "system": {"mode": str},
}


def _valid_graph():
    g = TypedGraph(schema=SCHEMA)
    g.add_node("s1", "sensor", battery=80.0, status="active")
    g.add_node("sys", "system", mode="maintenance")
    g.add_edge("s1", "sys", "part_of")
    return g


def test_valid_graph():
    g = _valid_graph()
    assert g.validate() == []


def test_unknown_type():
    g = TypedGraph(schema=SCHEMA)
    with pytest.raises(ValueError):
        g.add_node("x", "unknown")


def test_field_not_in_schema():
    g = TypedGraph(schema=SCHEMA)
    with pytest.raises(ValueError):
        g.add_node("s1", "sensor", foo=1.0)


def test_wrong_field_type():
    g = TypedGraph(schema=SCHEMA)
    with pytest.raises(ValueError):
        g.add_node("s1", "sensor", battery="high")


def test_constraint_failure():
    g = _valid_graph()
    g.add_node("s2", "sensor", battery=10.0, status="active")
    g.add_edge(
        "s2", "sys", "part_of", constraint=lambda src, dst: src["battery"] > 20.0
    )
    failures = g.validate()
    assert len(failures) == 1
    assert failures[0][0] == "s2"


def test_to_networkx():
    g = _valid_graph()
    G = g.to_networkx()
    assert "s1" in G.nodes
    assert G.nodes["s1"]["type"] == "sensor"
    assert G.edges["s1", "sys", 0]["relation"] == "part_of"
