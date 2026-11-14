"""Tests for the card."""
import json
from pathlib import Path

import pytest

from dense_armor.usable.card import analyze, to_json, validate

REPO_ROOT = Path(__file__).resolve().parents[2]

HERE = Path(__file__).resolve().parent
SAMPLES = HERE / "samples"
SCHEMA = REPO_ROOT / "dense_armor" / "usable" / "schema" / "card.schema.json"


def test_schema_file_exists():
    assert SCHEMA.exists(), f"{SCHEMA} non esiste"
    data = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert data["title"] == "usable card"
    assert "object" in data["properties"]
    assert "units" in data["properties"]


def test_audio_card_is_valid():
    c = analyze(str(SAMPLES / "tone.wav"))
    assert c["object"]["kind"] == "sound"
    assert len(c["units"]) == 7
    names = [u["name"] for u in c["units"]]
    for expected in ("waveform", "energy_bands", "mel_spectrogram",
                     "mfcc", "spectral", "onsets", "f0"):
        assert expected in names, f"{expected} assente: {names}"
    assert validate(c) == [], validate(c)


def test_text_card_tfidf_skipped_on_short_text():
    c = analyze(str(SAMPLES / "note.txt"))
    assert c["object"]["kind"] == "text"
    tfidf = next(u for u in c["units"] if u["name"] == "tfidf_top")
    assert "skipped" in tfidf["features"], tfidf["features"]


def test_text_card_tfidf_active_on_long_text(tmp_path):
    p = tmp_path / "long.txt"
    p.write_text(
        "The robot learns one sample at a time.\n"
        "The robot protects itself from drift.\n"
        "The library measures the signal.\n"
        "The signal changes when the market moves.\n"
        "The market is quiet today.\n",
        encoding="utf-8",
    )
    c = analyze(str(p))
    tfidf = next(u for u in c["units"] if u["name"] == "tfidf_top")
    assert "skipped" not in tfidf["features"], tfidf["features"]
    assert len(tfidf["features"]) > 0


def test_image_card_is_valid():
    c = analyze(str(SAMPLES / "pic.png"))
    assert c["object"]["kind"] == "image"
    img = next(u for u in c["units"] if u["name"] == "image")
    assert img["features"]["width"] > 0
    assert img["features"]["height"] > 0
    assert validate(c) == []


def test_table_card_has_one_unit_per_column():
    c = analyze(str(SAMPLES / "payments.csv"))
    assert c["object"]["kind"] == "table"
    cols = [u["name"] for u in c["units"]]
    assert "timestamp" in cols
    assert "amount" in cols
    assert "is_fraud" in cols


def test_code_card_has_one_unit_per_definition():
    c = analyze(str(SAMPLES / "jax_demo.py"))
    assert c["object"]["kind"] == "code"
    defs = [u["name"] for u in c["units"]]
    assert "truncate_bond" in defs
    assert "sample_shots" in defs


def test_unsupported_card_has_a_reason(tmp_path):
    p = tmp_path / "bin.dat"
    p.write_bytes(b"\x00\x01\x02\x03")
    c = analyze(str(p))
    # Either unsupported (with a reason), or a kind with valid units.
    if c["object"]["kind"] == "file":
        assert "unsupported" in c["object"].get("note", "") or validate(c) == []
    else:
        assert validate(c) == []


def test_folder_card_has_one_unit_per_file():
    c = analyze(str(SAMPLES))
    assert c["object"]["kind"] == "folder"
    assert c["summary"]["n_files"] >= 3
    files = [u for u in c["units"] if u["kind"] == "file"]
    assert len(files) == c["summary"]["n_files"]


def test_folder_card_validates():
    c = analyze(str(SAMPLES))
    assert validate(c) == []


def test_missing_object_raises():
    with pytest.raises(ValueError):
        analyze("/nonexistent/path/file.wav")


def test_to_json_roundtrip():
    c = analyze(str(SAMPLES / "tone.wav"))
    text = to_json(c)
    back = json.loads(text)
    assert back["object"]["kind"] == "sound"


def test_base_is_dense_armor_by_default():
    # With base=None the base is the installed dense_armor (or USABLE_REPO).
    # Only check that analyze does not fail with base=None
    # on a file (the ranking uses get_catalog, which may be empty).
    c = analyze(str(SAMPLES / "tone.wav"))
    assert c["object"]["kind"] == "sound"

# --------------------------------------------------------------
# Check 14 of 04_SELF_CHECK: 10x MAX_FILES is bounded, the card
# declares how many files it skipped, and MAX_BYTES is honoured.
# --------------------------------------------------------------
def test_folder_over_max_files_is_bounded(tmp_path):
    """A folder with 10x MAX_FILES keeps MAX_FILES units and declares
    the rest as skipped, in summary and in the note.
    """
    from dense_armor.usable.card import MAX_FILES
    n = MAX_FILES * 10
    for i in range(n):
        (tmp_path / f"f_{i:04d}.bin").write_bytes(b"\x00")
    c = analyze(str(tmp_path))
    assert c["object"]["kind"] == "folder"
    assert c["summary"]["n_files"] == MAX_FILES
    assert c["summary"]["n_files_seen"] == n
    assert c["summary"]["n_files_skipped"] == n - MAX_FILES
    assert c["summary"]["n_files_skipped_by_count"] == n - MAX_FILES
    assert "skipped" in c["object"]["note"]
    assert str(n - MAX_FILES) in c["object"]["note"]
    assert validate(c) == []


def test_folder_over_max_bytes_skips_big_file(tmp_path):
    """A file over MAX_BYTES is skipped and counted."""
    from dense_armor.usable.card import MAX_BYTES
    (tmp_path / "small.txt").write_text("x", encoding="utf-8")
    (tmp_path / "big.bin").write_bytes(b"x" * (MAX_BYTES + 1))
    c = analyze(str(tmp_path))
    assert c["summary"]["n_files"] == 1
    assert c["summary"]["n_files_skipped_by_size"] == 1
    assert "MAX_BYTES" in c["object"]["note"]
    assert validate(c) == []

def test_table_card_features_include_type_and_semantic():
    c = analyze(str(SAMPLES / "payments.csv"))
    by = {u["name"]: u for u in c["units"] if u["kind"] == "table"}
    amount = by["amount"]["features"]
    assert amount["type"] == "float"
    assert amount["semantic"] == "measure"
    assert amount["range"] is not None
    assert amount["range"]["min"] <= amount["range"]["max"]
    is_fraud = by["is_fraud"]["features"]
    assert is_fraud["semantic"] == "category"
    assert is_fraud["unique"] == 2
    assert is_fraud["top3"]
    assert validate(c) == []
