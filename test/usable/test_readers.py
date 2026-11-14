"""Tests for the readers (item C)."""
import types
from pathlib import Path

import pytest

from dense_armor.usable import readers
from dense_armor.usable.readers import (
    MAX_URL_BYTES,
    SUPPORTED,
    fetch,
    is_url,
    profile,
)

SAMPLES = Path(__file__).resolve().parent / "samples"


# --------------------------------------------------------------
# is_url
# --------------------------------------------------------------
def test_is_url_http_and_https():
    assert is_url("http://example.com/x")
    assert is_url("https://example.com/x")
    assert is_url("HTTPS://EXAMPLE.COM")
    assert not is_url("ftp://example.com/x")
    assert not is_url("example.com/x")
    assert not is_url("/tmp/x")
    assert not is_url("")


# --------------------------------------------------------------
# profile: one test per type, with samples/
# --------------------------------------------------------------
def test_profile_csv():
    pr = profile(SAMPLES / "payments.csv")
    assert pr["kind"] == "table"
    assert pr["rows"] > 0
    assert len(pr["columns"]) > 0


def test_profile_jsonl():
    pr = profile(SAMPLES / "rows.jsonl")
    assert pr["kind"] == "table"


def test_profile_json():
    pr = profile(SAMPLES / "expected.json")
    assert pr["kind"] == "table"


def test_profile_txt():
    pr = profile(SAMPLES / "note.txt")
    assert pr["kind"] == "text"
    assert pr["words"] > 0


def test_profile_pdf():
    pr = profile(SAMPLES / "report.pdf")
    assert pr["kind"] in ("text", "unsupported")  # pypdf puo' mancare


def test_profile_png():
    pr = profile(SAMPLES / "pic.png")
    assert pr["kind"] == "vision"


def test_profile_py():
    pr = profile(SAMPLES / "jax_demo.py")
    assert pr["kind"] == "code"
    assert len(pr["defs"]) > 0


def test_profile_wav():
    pr = profile(SAMPLES / "tone.wav")
    assert pr["kind"] == "sound"
    assert pr["rate"] > 0


# --------------------------------------------------------------
# Unsupported: non-empty reason + list of the types read
# --------------------------------------------------------------
def test_profile_unsupported_binary(tmp_path):
    """An unknown binary: unsupported with a non-empty reason and SUPPORTED."""
    p = tmp_path / "blob.bin"
    p.write_bytes(b"\x00\x01\x02\x03\x04\x05\x06\x07")
    pr = profile(p)
    assert pr["kind"] == "unsupported"
    assert pr["reason"]
    assert pr["reason"].strip() != ""
    # the list of the types read appears in the reason
    assert "Read:" in pr["reason"]
    assert "CSV" in pr["reason"]


def test_profile_nonexistent(tmp_path):
    pr = profile(tmp_path / "nope.xyz")
    assert pr["kind"] == "unsupported"
    assert pr["reason"]


def test_profile_sound_not_read_says_read(tmp_path):
    """A sound format not read: a reason that says what to do and what is read."""
    p = tmp_path / "x.mp3"
    p.write_bytes(b"ID3\x03\x00\x00\x00")
    pr = profile(p)
    assert pr["kind"] == "unsupported"
    assert "WAV" in pr["reason"]
    assert "Read:" in pr["reason"]


# --------------------------------------------------------------
# fetch: URL -> file scaricato in <package>/tmp/
# --------------------------------------------------------------
class _FakeResponse:
    """Fake response for fetch: .read, .geturl, .headers.get_content_type, context."""
    def __init__(self, data, url, ctype):
        self._data = data
        self._url = url
        self.headers = types.SimpleNamespace(
            get_content_type=lambda: ctype,
        )

    def geturl(self):
        return self._url

    def read(self, n=-1):
        if n < 0:
            return self._data
        return self._data[:n]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch_fetch(monkeypatch, response):
    """Replaces _open with a function that returns `response`."""
    captured = {}

    def fake_open(req, insecure):
        captured["url"] = req.full_url
        captured["insecure"] = insecure
        return response

    monkeypatch.setattr(
        "dense_armor.usable.readers._open", fake_open,
    )
    return captured


def test_fetch_downloads_to_package_tmp(monkeypatch):
    """The file goes in <package>/tmp/usable_XXX/, not in /tmp."""
    monkeypatch.setattr(
        "dense_armor.usable.readers._open",
        lambda req, insecure: _FakeResponse(b"hello", req.full_url, "text/plain"),
    )
    path, note = fetch("https://example.com/file.txt")
    package_tmp = Path(readers.__file__).resolve().parent / "tmp"
    assert package_tmp in path.parents or path.parent == package_tmp


def test_fetch_github_blob_becomes_raw(monkeypatch):
    """github.com/x/y/blob/main/file.md -> raw.githubusercontent.com/x/y/main/file.md."""
    captured = _patch_fetch(
        monkeypatch,
        _FakeResponse(b"hi", "https://raw.githubusercontent.com/u/r/main/x.md", "text/markdown"),
    )
    fetch("https://github.com/u/r/blob/main/x.md")
    assert captured["url"] == "https://raw.githubusercontent.com/u/r/main/x.md"


def test_fetch_over_max_bytes_raises(monkeypatch):
    """If the download exceeds MAX_URL_BYTES, ValueError."""
    _patch_fetch(
        monkeypatch,
        _FakeResponse(b"x" * (MAX_URL_BYTES + 1), "https://example.com/big", "text/plain"),
    )
    with pytest.raises(ValueError, match="larger than"):
        fetch("https://example.com/big.txt")


def test_fetch_insecure_is_passed_and_noted(monkeypatch):
    """insecure=True reaches _open and appears in the note."""
    captured = _patch_fetch(
        monkeypatch,
        _FakeResponse(b"data", "https://example.com/x.txt", "text/plain"),
    )
    path, note = fetch("https://example.com/x.txt", insecure=True)
    assert captured["insecure"] is True
    assert "certificate not checked" in note


def test_supported_lists_every_family():
    """SUPPORTED names every family read."""
    for parola in ("CSV", "JSON", "WAV", "TXT", "PDF", "PY", "PNG", "MP4"):
        assert parola in SUPPORTED, parola

def test_profile_csv_types_and_semantic():
    """payments.csv: amount float/measure, is_fraud int/category, hour category.

    ``timestamp`` is numeric in this file (0.0, 0.1, ...), not ISO:
    the correct type is float, the semantic is measure; monotonic and
    regular dt say so, but the type is not datetime.
    """
    pr = profile(SAMPLES / "payments.csv")
    by = {c["name"]: c for c in pr["columns"]}
    assert by["amount"]["type"] == "float"
    assert by["amount"]["semantic"] == "measure"
    assert by["is_fraud"]["type"] == "int"
    assert by["is_fraud"]["semantic"] == "category"
    assert by["hour"]["semantic"] == "category"
    assert by["timestamp"]["type"] == "float"
    assert by["timestamp"]["monotonic"] is True
    assert by["timestamp"]["dt"] is not None


def test_profile_csv_range_and_unique_ratio():
    """Numeric range on amount, unique_ratio and top3 on is_fraud."""
    pr = profile(SAMPLES / "payments.csv")
    by = {c["name"]: c for c in pr["columns"]}
    r = by["amount"]["range"]
    assert r is not None
    assert r["min"] <= r["median"] <= r["max"]
    assert r["std"] >= 0
    assert by["is_fraud"]["unique_ratio"] < 0.01
    assert by["is_fraud"]["top3"]
    assert by["timestamp"]["unique_ratio"] == 1.0


def test_profile_csv_datetime_column_range(tmp_path):
    """A column with real ISO dates produces range_t and semantic datetime."""
    p = tmp_path / "dates.csv"
    p.write_text(
        "ts,value\n"
        "2024-01-01T00:00:00,1\n"
        "2024-01-01T00:00:10,2\n"
        "2024-01-01T00:00:20,3\n"
        "2024-01-01T00:00:30,4\n",
        encoding="utf-8",
    )
    pr = profile(p)
    by = {c["name"]: c for c in pr["columns"]}
    assert by["ts"]["type"] == "datetime"
    assert by["ts"]["semantic"] == "datetime"
    rt = by["ts"]["range_t"]
    assert rt is not None
    assert rt["range_s"] == 30.0
    assert by["ts"]["monotonic"] is True
    assert by["ts"]["dt"] == 10.0
