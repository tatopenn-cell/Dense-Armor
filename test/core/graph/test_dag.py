"""Tests for ExecutionDAG: topological order, ancestors, run, errors."""

import networkx as nx
import pytest

from dense_armor.core.graph import ExecutionDAG
from dense_armor.roles import Signal


class _Pass:
    """A pass-through estimator: learn_one is a no-op, transform_one
    returns the input dict."""

    def learn_one(self, x, y=None):
        return self

    def transform_one(self, x):
        return dict(x) if isinstance(x, dict) else {"x": float(x)}


def _signals(n=3):
    return {f"n{i}": Signal([float(i)], names=[f"k{i}"], units=[""]) for i in range(n)}


def test_topological_order_diamond():
    dag = ExecutionDAG(
        nodes={"a": _Pass(), "b": _Pass(), "c": _Pass(), "d": _Pass()},
        edges=[("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")],
    )
    order = dag.topological_order()
    assert len(order) == 4
    pos = {n: i for i, n in enumerate(order)}
    assert pos["a"] < pos["b"] < pos["d"]
    assert pos["a"] < pos["c"] < pos["d"]


def test_ancestors_descendants():
    dag = ExecutionDAG(
        nodes={n: _Pass() for n in "abcd"},
        edges=[("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")],
    )
    assert dag.ancestors("d") == ["a", "b", "c"]
    assert dag.descendants("a") == ["b", "c", "d"]
    assert dag.ancestors("a") == []
    assert dag.descendants("d") == []


def test_cycle_refused():
    with pytest.raises(ValueError):
        ExecutionDAG(
            nodes={"a": _Pass(), "b": _Pass()},
            edges=[("a", "b"), ("b", "a")],
        )


def test_unknown_node_refused():
    with pytest.raises(ValueError):
        ExecutionDAG(
            nodes={"a": _Pass()},
            edges=[("a", "b")],
        )


def test_run_path_three_nodes():
    dag = ExecutionDAG(
        nodes={"a": _Pass(), "b": _Pass(), "c": _Pass()},
        edges=[("a", "b"), ("b", "c")],
    )
    sigs = _signals(1)
    out = dag.run({"a": sigs["n0"]})
    assert out["a"] is not None
    assert out["b"] is not None
    assert out["c"] is not None


def test_all_topological_sorts_count():
    G = nx.DiGraph([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])
    assert len(list(nx.all_topological_sorts(G))) == 2


class _Reg:
    """A regressor that records the order of its calls."""

    def __init__(self):
        self.calls = []

    def predict_one(self, x, t=None):
        self.calls.append("predict")
        return 1.5

    def learn_one(self, x, y=None, t=None):
        self.calls.append(("learn", y))
        return self


def test_regressor_predicts_before_it_learns_and_needs_a_label():
    reg = _Reg()
    dag = ExecutionDAG(nodes={"r": reg}, edges=[])
    sig = Signal([1.0], names=["x"], units=[""])
    out = dag.run({"r": sig})
    assert out["r"]["r"] == 1.5
    assert reg.calls == ["predict"]
    dag.run({"r": sig}, labels={"r": 2.0})
    assert reg.calls == ["predict", "predict", ("learn", 2.0)]


def test_repeated_edges_are_one_edge():
    dag = ExecutionDAG(
        nodes={"a": _Pass(), "b": _Pass()}, edges=[("a", "b"), ("a", "b")]
    )
    assert dag.edges == [("a", "b")]
    assert dag.run({"a": _signals(1)["n0"]})["b"] is not None
