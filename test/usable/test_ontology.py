"""Tests for the ontology (item E)."""

from pathlib import Path

import pytest

from dense_armor.usable import concepts
from dense_armor.usable.onto_select import load_ontology

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------
# load_ontology: errori chiari (mai criptici)
# --------------------------------------------------------------
def test_load_ontology_missing_repo_raises_clear_error():
    with pytest.raises(RuntimeError, match="Dense-Armor not found"):
        load_ontology(Path("/nonexistent/repo"))


def test_load_ontology_missing_ontology_file_raises_clear_error(tmp_path):
    repo = tmp_path / "fake_repo"
    (repo / "dense_armor" / "core" / "graph").mkdir(parents=True)
    # niente ontology.py
    with pytest.raises(RuntimeError, match="the ontology file was not found"):
        load_ontology(repo)


def test_load_ontology_real_repo_has_typed_graph_and_constraint():
    mod = load_ontology(REPO)
    assert hasattr(mod, "TypedGraph"), "manca TypedGraph in ontology"
    assert hasattr(mod, "Constraint"), "manca Constraint in ontology"


# --------------------------------------------------------------
# Propagation: the mass propagates along the edges
# --------------------------------------------------------------
def test_mass_propagates_along_synonym_edges():
    """The synonym 'cut ~ attenuate' carries +0.7 of the mass to the neighbour."""
    # control: both words are in ontology.json['synonyms']
    g = concepts.build(REPO, [["cut", 1]], "en")
    sp = concepts.spread(g)
    assert "cut" in sp["mass"], "missing 'cut' in the mass"
    assert "attenuate" in sp["mass"], "missing 'attenuate' in the mass (synonym)"
    assert abs(sp["mass"]["cut"] - 1.0) < 1e-9
    assert abs(sp["mass"]["attenuate"] - 0.7) < 1e-9


def test_mass_does_not_propagate_through_unrelated_edges():
    """The graph knows 'cut -> attenuate' but not 'cut -> elephant'."""
    g = concepts.build(REPO, [["cut", 1]], "en")
    sp = concepts.spread(g)
    assert "elephant" not in sp["mass"]


# --------------------------------------------------------------
# Antonym: negative mass, and it lowers the class it names
# --------------------------------------------------------------
def test_antonym_has_negative_mass():
    """The antonym 'low != high' carries -0.5 to the mass of the neighbour."""
    g = concepts.build(REPO, [["low", 1]], "en")
    sp = concepts.spread(g)
    assert "low" in sp["mass"]
    assert "high" in sp["mass"]
    assert sp["mass"]["high"] < 0, f"massa di 'high': {sp['mass']['high']}"
    assert abs(sp["mass"]["high"] + 0.5) < 1e-9


def test_antonym_lowers_the_class_it_reaches():
    """A concept with negative mass lowers the class it names
    (boost < 0) through a relates_primary link."""
    onto = load_ontology(REPO)
    g = onto.TypedGraph(concepts.SCHEMA)
    g.add_node("t:low", "term", word="low", lang="en", weight=1.0)
    g.add_node("c:low", "concept", label="low")
    g.add_node("c:high", "concept", label="high")
    g.add_node("p:MyClass", "primitive", label="MyClass")
    g.add_edge("t:low", "c:low", "expresses")
    g.add_edge("c:low", "c:high", "antonym")
    g.add_edge("c:high", "p:MyClass", "relates_primary")

    sp = concepts.spread(g)
    assert sp["mass"]["high"] == -0.5
    assert "MyClass" in sp["links"], sp["links"]

    row = {"class": "MyClass", "name": "MyClass"}
    boost_value, _hits = concepts.boost(row, sp)
    assert boost_value < 0, f"boost di MyClass: {boost_value}"


# --------------------------------------------------------------
# Orphan concept: changes nothing
# --------------------------------------------------------------
def test_orphan_concept_does_not_change_anything():
    """A concept without edges carries its mass (1.0) but produces no
    link to classes: p and the mass of the classes stay identical."""
    onto = load_ontology(REPO)
    g = onto.TypedGraph(concepts.SCHEMA)
    g.add_node("t:foo", "term", word="foo", lang="en", weight=1.0)
    g.add_node("c:foo", "concept", label="foo")
    g.add_edge("t:foo", "c:foo", "expresses")

    sp = concepts.spread(g)
    assert sp["mass"].get("foo", 0) == 1.0
    assert sp["links"] == {}, f"links inattesi: {sp['links']}"
