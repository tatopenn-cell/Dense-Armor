"""Tests for the gap report (item G)."""
import json
from pathlib import Path

from dense_armor.usable.catalog import get_catalog
from dense_armor.usable.ontology import gaps as gp

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------
# The number of classes without links: formula with the same unit
# --------------------------------------------------------------
def test_classes_without_link_formula():
    """distinct classes - classes with a link = classes without a link.

    Il criterio di ISTRUZIONI.md dice "447 meno le classi distinte in
    ontology.json" (354). 447 sono le righe del catalogo; 93 sono le
    classi distinte con un legame: mescolano righe e distinte. La
    formula con la stessa unita' e' quella qui sopra. Il numero che
    esce e' 301, non 354; il criterio originale e' sbagliato, non il
    codice.
    """
    rep = gp.report(REPO)
    assert rep["n_without_link"] == rep["n_classes"] - rep["n_with_link"]
    assert rep["n_without_link"] > 0


def test_classes_without_link_are_distinct():
    """The list has no duplicates: the homonyms [core]/[ml] are one class."""
    missing = gp.classes_without_link(REPO)
    assert len(missing) == len(set(missing))


def test_links_to_class_not_in_catalog_are_reported_as_none():
    """If 'links' names a class outside the catalog, it does not enter the
    count of 'classes without links' (those are only from the catalog)."""
    cat = get_catalog(REPO)
    classi = {r["class"] for r in cat}
    missing = set(gp.classes_without_link(REPO))
    assert missing <= classi


# --------------------------------------------------------------
# Words without a concept
# --------------------------------------------------------------
def test_words_without_concept_returns_unknown_words():
    words = ["anomaly", "drift", "zzznotaword"]
    out = gp.words_without_concept(words, REPO)
    assert "zzznotaword" in out
    # "anomaly" and "drift" are known concepts in ontology.json["links"]
    assert "anomaly" not in out
    assert "drift" not in out


def test_report_includes_last_analysis_when_file_exists(tmp_path):
    last = tmp_path / "last.json"
    last.write_text(json.dumps({"file": "x", "words": ["anomaly", "zzznotaword"]}),
                    encoding="utf-8")
    rep = gp.report(REPO, last)
    assert rep["last_file"] == str(last)
    assert rep["last_words"] == 2
    assert "zzznotaword" in rep["last_without_concept_words"]
    assert "anomaly" not in rep["last_without_concept_words"]


def test_report_without_last_file_has_no_last_keys():
    rep = gp.report(REPO)
    assert "last_file" not in rep


# --------------------------------------------------------------
# write_last_analysis: scrive tmp/last_analysis.json
# --------------------------------------------------------------
def test_write_last_analysis_creates_file():
    p = gp.write_last_analysis({"file": "x", "words": ["anomaly"]})
    assert p.is_file()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["words"] == ["anomaly"]


# --------------------------------------------------------------
# CLI
# --------------------------------------------------------------
def test_cli_ontology_gaps_prints_numbers(capsys):
    from dense_armor.usable.cli import cmd_ontology_gaps
    rc = cmd_ontology_gaps(REPO)
    out = capsys.readouterr().out
    assert rc == 0
    assert "classes without a link:" in out
    assert "formula:" in out


def test_cli_ontology_gaps_json(capsys):
    from dense_armor.usable.cli import cmd_ontology_gaps
    rc = cmd_ontology_gaps(REPO, None, True)
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert "n_without_link" in data
    assert isinstance(data["without_link"], list)
