"""Tests for Dataflow."""

import pytest

from dense_armor.core.graph import Dataflow, Provenance
from dense_armor.roles import Signal


class _Inc:
    def learn_one(self, x, y=None):
        return self

    def transform_one(self, x):
        return {k: float(v) + 1.0 for k, v in dict(x).items()}


def _flow():
    return Dataflow().source("a", _Inc()).then("b", _Inc(), "a").then("c", _Inc(), "b")


def test_lineage_goes_from_the_source_to_the_node():
    d = _flow()
    assert d.lineage("c") == ["a", "b", "c"]
    assert d.lineage("a") == ["a"]


def test_edges_in_declaration_order():
    assert _flow().edges() == [("a", "b"), ("b", "c")]


def test_build_runs_like_a_hand_made_dag():
    sig = Signal([1.0], names=["x"], units=[""])
    out = _flow().build().run({"a": sig})
    assert [float(out[n]["x"]) for n in "abc"] == [2.0, 3.0, 4.0]


def test_build_passes_the_provenance():
    prov = Provenance()
    _flow().build(provenance=prov).run({"a": Signal([1.0], names=["x"], units=[""])})
    assert [r["node"] for r in prov.records()] == ["a", "b", "c"]


def test_a_name_used_twice_is_refused():
    with pytest.raises(ValueError):
        Dataflow().source("a", _Inc()).source("a", _Inc())


def test_a_node_after_an_unknown_name_is_refused():
    with pytest.raises(ValueError):
        Dataflow().then("b", _Inc(), "a")


def test_lineage_of_an_unknown_node_is_refused():
    with pytest.raises(ValueError):
        _flow().lineage("z")
