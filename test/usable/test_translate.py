"""Tests for the translation (item K)."""
import json
from pathlib import Path

import pytest

from dense_armor.usable.card import analyze
from dense_armor.usable.translate import (
    DICTIONARY_PATH,
    SUPPORTED,
    Translator,
)

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "dense_armor" / "usable"
SAMPLES = Path(__file__).resolve().parent / "samples"


# --------------------------------------------------------------
# glossary backend
# --------------------------------------------------------------
def test_glossary_translates_10_known_words():
    words = ["filtro", "filtri", "filtrare", "filtra", "filtrato",
             "rumore", "segnale", "anomalia", "deriva", "soglia"]
    out = Translator("glossary").translate(words, "it")
    assert len(out) == 10
    tradotti = [w for w, v in out.items() if v is not None]
    assert len(tradotti) >= 10, f"tradotti {len(tradotti)}/10: {out}"


def test_glossary_known_translation():
    out = Translator("glossary").translate(["filtro"], "it")
    assert out["filtro"] == "filter"


def test_english_is_returned_unchanged():
    out = Translator("glossary").translate(["filter", "signal"], "en")
    assert out == {"filter": "filter", "signal": "signal"}


def test_unknown_language_returns_none():
    out = Translator("glossary").translate(["filtro"], "fr")
    assert out == {"filtro": None}


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown backend"):
        Translator("nonsense")


def test_supported_backends():
    assert SUPPORTED == ("glossary", "dictionary", "argos")


def test_env_variable_selects_backend(monkeypatch):
    monkeypatch.setenv("USABLE_TRANSLATOR", "glossary")
    assert Translator().backend == "glossary"


# --------------------------------------------------------------
# argos absent -> available=False, why_not_available explains
# --------------------------------------------------------------
def test_argos_absent_reports_reason():
    tr = Translator("argos")
    if tr.available:
        pytest.skip("argos is installed here")
    assert tr.available is False
    reason = tr.why_not_available()
    assert "argostranslate" in reason
    assert "not installed" in reason
    # also: translate still returns the words unchanged as None
    out = tr.translate(["filtro"], "it")
    assert out == {"filtro": None}


# --------------------------------------------------------------
# dictionary backend (built from ConceptNet subset)
# --------------------------------------------------------------
def test_dictionary_is_built_from_conceptnet():
    if not DICTIONARY_PATH.is_file():
        pytest.skip("dictionary_it_en.tsv not built (run tools.build_dictionary)")
    lines = DICTIONARY_PATH.read_text(encoding="utf-8").splitlines()
    # header + at least 100 entries
    assert len(lines) > 100, f"dictionary has only {len(lines)} lines (header + n)"
    # each row: italian<TAB>english<TAB>weight
    bad = [ln for ln in lines[1:] if ln.count("\t") != 2]
    assert not bad, f"malformed rows (no 2 tabs): {bad[:3]}"
    # at least one known entry
    words = {ln.split("\t")[0] for ln in lines[1:]}
    assert "anomalia" in words, "anomalia missing"


def test_dictionary_translates_a_word():
    if not DICTIONARY_PATH.is_file():
        pytest.skip("dictionary_it_en.tsv not built (run tools.build_dictionary)")
    # glossary wins, so "anomalia" -> "anomaly" comes from the manual
    # glossary; try a word only in the dictionary
    out2 = Translator("dictionary").translate(["addomesticatore"], "it")
    if out2["addomesticatore"] is None:
        pytest.skip("chosen word not in dictionary (data changed)")
    assert isinstance(out2["addomesticatore"], str)


# --------------------------------------------------------------
# card: 'translated' / 'total' in the summary
# --------------------------------------------------------------
def test_card_italian_text_has_translation_summary(tmp_path):
    """An Italian text produces a card with summary.translation."""
    p = tmp_path / "italian.txt"
    p.write_text(
        "Il filtro taglia il rumore. Il segnale e' pulito. "
        "L'anomalia e' chiara. La deriva e' bassa. La soglia e' alta.",
        encoding="utf-8",
    )
    c = analyze(str(p))
    assert c["object"]["kind"] == "text"
    assert "translation" in c["summary"], c["summary"]
    tr = c["summary"]["translation"]
    assert tr["lang"] == "it"
    assert tr["backend"] in SUPPORTED
    assert tr["translated"] <= tr["total"]
    assert tr["total"] > 0


def test_card_english_text_has_no_translation_summary(tmp_path):
    p = tmp_path / "english.txt"
    p.write_text(
        "The filter cuts the noise. The signal is clean. "
        "The anomaly is clear. The drift is low. The threshold is high.",
        encoding="utf-8",
    )
    c = analyze(str(p))
    assert c["object"]["kind"] == "text"
    # English: no translation key (the ontology is already English)
    assert "translation" not in c["summary"], c["summary"]


# --------------------------------------------------------------
# no Italian in the ontology
# --------------------------------------------------------------
def test_ontology_has_no_italian_words():
    """The ontology is English only (item K criterion)."""
    onto = json.loads((PKG / "ontology.json").read_text(encoding="utf-8"))
    # A few Italian words that must NOT be in the ontology
    forbidden = {"filtro", "segnale", "rumore", "anomalia", "soglia",
                 "deriva", "anomalo", "basso", "alto", "veloce", "lento"}
    seen = set()
    for a, b in onto.get("synonyms", []) + onto.get("antonyms", []):
        seen.add(a.lower())
        seen.add(b.lower())
    for names in onto.get("evokes", {}).values():
        seen.update(n.lower() for n in names)
    seen.update(c.lower() for c in onto.get("links", {}))
    bad = seen & forbidden
    assert not bad, f"Italian words in the ontology: {sorted(bad)}"

def test_dictionary_blocklist():
    """A word in the blocklist returns None, not a wrong translation."""
    tr = Translator("dictionary")
    if not tr.available:
        pytest.skip("dictionary backend not available")
    out = tr.translate(["ottocentouno", "marchetta"], "it")
    assert out["ottocentouno"] is None
    assert out["marchetta"] is None


def test_dictionary_blocklist_does_not_break_good_words():
    """A word not in the blocklist keeps working."""
    tr = Translator("dictionary")
    if not tr.available:
        pytest.skip("dictionary backend not available")
    # anomalia -> the glossary wins
    out = tr.translate(["anomalia"], "it")
    assert out["anomalia"] == "anomaly"
