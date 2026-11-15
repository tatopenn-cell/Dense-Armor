"""Tests for the gap report."""

import json
from pathlib import Path

from dense_armor.usable.catalog import get_catalog
from dense_armor.usable.ontology import gaps as gp

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------
# The number of classes without links: formula with the same unit
# --------------------------------------------------------------
def test_classes_without_link_formula():
    """distinct classes of the catalog - distinct classes with a link = classes without a link."""
    rep = gp.report(REPO)
    assert rep["n_without_link"] == rep["n_classes"] - rep["n_with_link"]


def test_every_class_has_a_link():
    """The manual links and the links of level auto (the domain of the class) reach every class."""
    rep = gp.report(REPO)
    assert rep["n_without_link"] == 0, rep["without_link"][:10]
    assert rep["n_with_link"] == rep["n_classes"]


def test_auto_links_are_never_above_a_manual_one():
    """A link of level auto has rank 0.25 at most, a source, and does not replace a manual link."""
    from dense_armor.usable import concepts

    ont = Path(concepts.__file__).resolve().parent
    manual = json.loads((ont / "ontology.json").read_text(encoding="utf-8"))["links"]
    auto = json.loads(
        (ont / "ontology" / "auto_links.json").read_text(encoding="utf-8")
    )["links"]
    merged = concepts._links()
    assert auto
    for name, cls, rank, source in auto:
        assert rank <= 0.25
        assert source.get("source") and source.get("license")
        if cls in manual.get(name, {}):
            assert merged[name][cls] == manual[name][cls]
    for name, targets in manual.items():
        for cls, rank in targets.items():
            assert merged[name][cls] == rank


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
    last.write_text(
        json.dumps({"file": "x", "words": ["anomaly", "zzznotaword"]}), encoding="utf-8"
    )
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
