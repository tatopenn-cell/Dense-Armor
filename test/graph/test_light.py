"""LightGraph: the same answers as networkx, and the ontology and the DAG run without networkx."""

import random
import sys

import networkx as nx
import numpy as np
import pytest

from dense_armor.core.graph import Action, Constraint, ExecutionDAG, TypedGraph, apply_action
from dense_armor.core.graph._light import LightGraph


def _random_dag(seed, n=12, p=0.25):
    rnd = random.Random(seed)
    names = [f"n{i}" for i in range(n)]
    rnd.shuffle(names)
    edges = [(names[i], names[j]) for i in range(n) for j in range(i + 1, n) if rnd.random() < p]
    return names, edges


@pytest.mark.parametrize("seed", range(25))
def test_topological_order_and_reach_agree_with_networkx(seed):
    names, edges = _random_dag(seed)
    light, ref = LightGraph(), nx.DiGraph()
    light.add_nodes_from(names)
    ref.add_nodes_from(names)
    for u, v in edges:
        light.add_edge(u, v)
        ref.add_edge(u, v)
    assert light.topological_order() == list(nx.topological_sort(ref))
    assert light.is_acyclic()
    for name in names:
        assert light.ancestors(name) == nx.ancestors(ref, name)
        assert light.descendants(name) == nx.descendants(ref, name)
        assert light.predecessors(name) == list(ref.predecessors(name))
        assert light.successors(name) == list(ref.successors(name))
        assert light.in_degree(name) == ref.in_degree(name)


def test_a_cycle_is_found():
    g = LightGraph()
    for u, v in [("a", "b"), ("b", "c"), ("c", "a")]:
        g.add_edge(u, v)
    assert not g.is_acyclic()
    with pytest.raises(ValueError, match="cycle"):
        g.topological_order()


def test_parallel_edges_get_their_own_key_like_networkx():
    light, ref = LightGraph(), nx.MultiDiGraph()
    for g in (light, ref):
        g.add_node("a", type="t")
        g.add_node("b", type="t")
        g.add_edge("a", "b", relation="r1")
        g.add_edge("a", "b", relation="r2")
        g.add_edge("b", "a", relation="r3")
    assert light.edges(keys=True, data=True) == list(ref.edges(keys=True, data=True))
    assert light.edges() == list(ref.edges())
    assert light.edges(data="relation") == list(ref.edges(data="relation"))
    assert light.in_edges("b", keys=True, data=True) == list(ref.in_edges("b", keys=True, data=True))
    assert light.out_edges("a", keys=True, data=True) == list(ref.out_edges("a", keys=True, data=True))
    assert light.number_of_edges() == ref.number_of_edges() == 3
    assert light.nodes["a"] == ref.nodes["a"]
    assert light.nodes(data=True) == list(ref.nodes(data=True))


def test_copy_is_independent_and_to_networkx_keeps_the_attributes():
    g = LightGraph()
    g.add_node("a", state={"x": 1})
    g.add_node("b", state={"x": 2})
    g.add_edge("a", "b", relation="r")
    c = g.copy()
    c.nodes["a"]["state"]["x"] = 99
    assert g.nodes["a"]["state"]["x"] == 1
    n = g.to_networkx()
    assert isinstance(n, nx.MultiDiGraph)
    assert n.nodes["a"]["state"] == {"x": 1}
    assert list(n.edges(keys=True, data="relation")) == [("a", "b", 0, "r")]


@pytest.fixture
def no_networkx(monkeypatch):
    for name in [m for m in sys.modules if m == "networkx" or m.startswith("networkx.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "networkx", None)


def test_the_typed_graph_runs_without_networkx(no_networkx):
    g = TypedGraph(schema={"machine": {"load": float, "mode": str}})
    g.add_node("m1", "machine", load=0.5, mode="run")
    g.add_node("m2", "machine", load=0.9, mode="run")
    g.add_edge("m1", "m2", "feeds", Constraint("the load rises", lambda s, d: s["load"] <= d["load"]))
    assert g.validate() == []
    stop = Action("stop", preconditions=[("m1", "mode", "==", "run")], effects=[("m1", "mode", "set", "stop")])
    g2 = apply_action(g, stop)
    assert g2.graph.nodes["m1"]["state"]["mode"] == "stop"
    assert g.graph.nodes["m1"]["state"]["mode"] == "run"
    g.update("m1", load=0.99)
    assert g.validate() == [("m1", "m2", "feeds")]
    assert g.validate(nodes=["m2"]) == [("m1", "m2", "feeds")]


class _Add:
    def __init__(self, k):
        self.k = k

    def learn_one(self, x, y=None, t=None):
        return self

    def transform_one(self, x, t=None):
        return {k: v + self.k for k, v in x.items()}


def test_the_execution_dag_runs_without_networkx(no_networkx):
    dag = ExecutionDAG({"a": _Add(1), "b": _Add(2), "c": _Add(3)}, [("a", "b"), ("b", "c")])
    assert dag.topological_order() == ["a", "b", "c"]
    assert dag.ancestors("c") == ["a", "b"]
    assert dag.descendants("a") == ["b", "c"]
    assert dag.is_acyclic()
    with pytest.raises(ValueError, match="cycle"):
        ExecutionDAG({"a": _Add(1), "b": _Add(2)}, [("a", "b"), ("b", "a")])


def test_to_networkx_still_works_and_needs_it():
    g = TypedGraph(schema={"t": {"v": float}})
    g.add_node("a", "t", v=1.0)
    g.add_node("b", "t", v=2.0)
    g.add_edge("a", "b", "r")
    assert isinstance(g.to_networkx(), nx.MultiDiGraph)
    assert np.isfinite(sum(d["state"]["v"] for _, d in g.to_networkx().nodes(data=True)))


def test_a_graph_can_be_iterated_counted_and_asked_for_its_degrees():
    g = LightGraph()
    g.add_edge("a", "b")
    g.add_edge("a", "c")
    g.add_edge("b", "c")
    assert list(g) == ["a", "b", "c"]
    assert len(g) == g.number_of_nodes() == 3
    assert g.number_of_edges() == 3
    assert [g.out_degree(n) for n in "abc"] == [2, 1, 0]
    assert [g.in_degree(n) for n in "abc"] == [0, 1, 2]
    assert g.out_degree("missing") == 0


def test_the_simple_graph_for_networkx_keeps_nodes_and_edges():
    g = LightGraph()
    g.add_node("a", role="source")
    g.add_edge("a", "b", weight=2.0)
    n = g.to_networkx(directed_simple=True)
    assert isinstance(n, nx.DiGraph) and not n.is_multigraph()
    assert n.nodes["a"]["role"] == "source"
    assert list(n.edges(data="weight")) == [("a", "b", 2.0)]


def test_the_execution_dag_gives_a_networkx_digraph():
    dag = ExecutionDAG({"a": _Add(1), "b": _Add(2), "c": _Add(3)}, [("a", "b"), ("a", "c")])
    n = dag.to_networkx()
    assert isinstance(n, nx.DiGraph)
    assert sorted(n.edges()) == [("a", "b"), ("a", "c")]
    assert list(nx.topological_sort(n)) == dag.topological_order()
