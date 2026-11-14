"""Tests for the scripts that build the ontology data: they run on small inputs, offline, and write only to temporary folders."""

import gzip
import io
import json
import sys

import pytest

from dense_armor.usable.tools import (
    build_class_links as bcl,
)
from dense_armor.usable.tools import (
    build_conceptnet_subset as bcs,
)
from dense_armor.usable.tools import (
    build_dictionary as bd,
)
from dense_armor.usable.tools import (
    build_ontology_data as bod,
)

SUBSET_HEADER = "# relation\tlang_start\tstart\tlang_end\tend\tweight\n"


# --------------------------------------------------------------
# build_class_links: a link of level auto for every class that has none
# --------------------------------------------------------------
def test_class_links_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bcl, "OUT", tmp_path / "auto_links.json")
    assert bcl.main(["--dry-run"]) == 0
    assert "classes" in capsys.readouterr().out
    assert not (tmp_path / "auto_links.json").exists()


def test_class_links_cover_the_classes_without_a_manual_link(tmp_path, monkeypatch):
    monkeypatch.setattr(bcl, "OUT", tmp_path / "auto_links.json")
    assert bcl.main([]) == 0
    rows = json.loads((tmp_path / "auto_links.json").read_text(encoding="utf-8"))[
        "links"
    ]
    manual = json.loads((bcl.USABLE / "ontology.json").read_text(encoding="utf-8"))[
        "links"
    ]
    manual_classes = {c for targets in manual.values() for c in targets}
    assert rows
    for concept, cls, rank, source in rows:
        assert cls not in manual_classes
        assert rank == bcl.RANK
        assert source["source"] and source["license"]


# --------------------------------------------------------------
# build_dictionary: Italian -> English from the ConceptNet subset
# --------------------------------------------------------------
def test_dictionary_needs_the_subset(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bd, "SUBSET", tmp_path / "missing.tsv")
    monkeypatch.setattr(bd, "OUTPUT", tmp_path / "dictionary.tsv")
    assert bd.main(["--dry-run"]) == 1
    assert bd.main([]) == 1
    assert "subset missing" in capsys.readouterr().out


def test_dictionary_keeps_the_best_translation_of_a_catalog_word(
    tmp_path, monkeypatch, capsys
):
    subset = tmp_path / "subset.tsv"
    subset.write_text(
        SUBSET_HEADER
        + "Synonym\tit\tanomalia\ten\tanomaly\t2.0\n"
        + "RelatedTo\tit\tanomalia\ten\tanomaly\t5.0\n"
        + "Antonym\tit\tnormale\ten\tanomaly\t3.0\n"
        + "Synonym\tfr\tanomalie\ten\tanomaly\t3.0\n"
        + "Synonym\tit\tzzz\ten\tqqqqunknownword\t3.0\n"
        + "Synonym\tit\trumore\ten\tnoise\tnot-a-number\n"
        + "short\tline\n",
        encoding="utf-8",
    )
    out = tmp_path / "dictionary.tsv"
    monkeypatch.setattr(bd, "SUBSET", subset)
    monkeypatch.setattr(bd, "OUTPUT", out)
    assert bd.main(["--dry-run"]) == 0
    assert "unique Italian words: 2" in capsys.readouterr().out
    assert not out.exists()
    assert bd.main([]) == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# italian")
    assert "anomalia\tanomaly\t5.0" in lines
    assert "rumore\tnoise\t1.0" in lines
    assert not any(line.startswith(("normale", "anomalie", "zzz")) for line in lines)


# --------------------------------------------------------------
# build_conceptnet_subset: the dump is streamed; here a fake dump of five lines
# --------------------------------------------------------------
def fake_dump() -> bytes:
    def edge(rel, a, b, meta='{"weight": 2.0}'):
        return f"/a/[/r/{rel}/,{a},{b}]\t/r/{rel}\t{a}\t{b}\t{meta}\n"

    lines = [
        edge("Synonym", "/c/en/noise/n", "/c/en/disturbance/n"),
        edge("Synonym", "/c/it/rumore/n", "/c/en/noise/n", "not json"),
        edge("Synonym", "/c/en/noise/n", "/c/en/disturbance/n"),
        edge("IsA", "/c/en/noise/n", "/c/en/sound/n"),
        edge("Synonym", "/c/fr/bruit/n", "/c/fr/son/n"),
        edge("Synonym", "/c/en/noise/n", "/c/en/noise/n"),
        "short\tline\n",
    ]
    return gzip.compress("".join(lines).encode("utf-8"))


@pytest.fixture
def fake_download(monkeypatch):
    monkeypatch.setattr(
        bcs.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(fake_dump())
    )


def test_conceptnet_subset_keeps_the_rows_of_the_catalog_words(
    tmp_path, fake_download, capsys
):
    out = tmp_path / "subset.tsv"
    assert bcs.build(out, None, False) == 0
    rows = [r.split("\t") for r in out.read_text(encoding="utf-8").splitlines()[1:]]
    assert ["Synonym", "en", "noise", "en", "disturbance", "2.0"] in rows
    assert ["Synonym", "it", "rumore", "en", "noise", "1.0"] in rows
    assert len(rows) == 2
    assert "CC BY-SA 4.0" in capsys.readouterr().out


def test_conceptnet_subset_limit_and_insecure_option(tmp_path, fake_download):
    out = tmp_path / "subset.tsv"
    assert bcs.main(["--output", str(out), "--limit", "1", "--insecure"]) == 0
    assert len(out.read_text(encoding="utf-8").splitlines()) == 2


def test_conceptnet_subset_dry_run_downloads_nothing(capsys, monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("the dry run must not download")

    monkeypatch.setattr(bcs.urllib.request, "urlopen", refuse)
    assert bcs.main(["--dry-run"]) == 0
    assert "expected size" in capsys.readouterr().out


# --------------------------------------------------------------
# build_ontology_data: auto.json and LICENZE.md from the sources that are present
# --------------------------------------------------------------
@pytest.fixture
def no_wordnet(monkeypatch):
    monkeypatch.setitem(sys.modules, "nltk", None)
    monkeypatch.setitem(sys.modules, "nltk.corpus", None)


def test_ontology_data_dry_run_reports_a_missing_subset(
    tmp_path, monkeypatch, capsys, no_wordnet
):
    monkeypatch.setattr(bod, "DATA", tmp_path / "data")
    monkeypatch.setattr(bod, "ONTO", tmp_path / "ontology")
    assert bod.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "terms of the catalog" in out
    assert "MISSING" in out
    assert not (tmp_path / "ontology").exists()


def test_ontology_data_reads_the_columns_of_the_subset(
    tmp_path, monkeypatch, capsys, no_wordnet
):
    data, onto = tmp_path / "data", tmp_path / "ontology"
    data.mkdir()
    a, b, c = bod._catalog_terms()[10:13]
    (data / "conceptnet_subset.tsv").write_text(
        SUBSET_HEADER
        + f"Synonym\ten\t{a}\ten\t{b}\t2.0\n"
        + f"Antonym\ten\t{a}\ten\t{c}\t2.0\n"
        + f"Synonym\ten\t{b}\ten\t{a}\t2.0\n"
        + f"Synonym\tit\t{a}\ten\t{c}\t2.0\n"
        + "Synonym\ten\tnotacatalogword\ten\talsonot\t1.0\n"
        + "short\tline\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(bod, "DATA", data)
    monkeypatch.setattr(bod, "ONTO", onto)
    assert bod.main(["--dry-run"]) == 0
    assert "ConceptNet subset:" in capsys.readouterr().out
    assert bod.main([]) == 0
    auto = json.loads((onto / "auto.json").read_text(encoding="utf-8"))
    assert [a, b, {"source": "conceptnet", "license": "CC BY-SA 4.0"}] in auto[
        "synonyms"
    ]
    assert [a, c, {"source": "conceptnet", "license": "CC BY-SA 4.0"}] in auto[
        "antonyms"
    ]
    assert len(auto["synonyms"]) == 1
    assert "CC BY-SA 4.0" in (data / "LICENZE.md").read_text(encoding="utf-8")
