"""Tests for the quality of the deliverable (item H).

Every check that the item names, one test: the README exists and says
the exit codes; every `--json` command produces valid JSON; the exit
codes are the documented ones; `ruff` and `mypy` pass when installed
(they are dev tools, not runtime dependencies).
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "dense_armor" / "usable"
SAMPLES = Path(__file__).resolve().parent / "samples"


def _run(args, timeout=120):
    """Run the CLI in a fresh interpreter; return (rc, stdout, stderr)."""
    r = subprocess.run(
        [sys.executable, "-m", "dense_armor.usable", *args],
        cwd=REPO, capture_output=True, text=True, timeout=timeout,
    )
    return r.returncode, r.stdout, r.stderr


# --------------------------------------------------------------
# README
# --------------------------------------------------------------
def test_readme_exists():
    p = PKG / "README.md"
    assert p.is_file(), f"{p} non esiste"
    txt = p.read_text(encoding="utf-8")
    assert "## Commands" in txt
    assert "## Exit codes" in txt
    assert "USABLE_REPO" in txt
    assert "ontology.json" in txt


def test_readme_documents_the_three_exit_codes():
    txt = (PKG / "README.md").read_text(encoding="utf-8")
    # i tre codici, ciascuno sulla sua riga
    for code, reason in [
        ("0", "ok"),
        ("1", "not found"),
        ("2", "wrong arguments"),
    ]:
        righe = [ln for ln in txt.splitlines() if ln.strip().startswith(code)]
        assert any(reason in ln for ln in righe), f"exit {code}: missing {reason!r}"


# --------------------------------------------------------------
# --json: every command produces valid JSON
# --------------------------------------------------------------
def test_list_json_is_valid():
    rc, out, err = _run(["list", "--domain", "drift", "--json"])
    assert rc == 0, err
    data = json.loads(out)
    assert isinstance(data, list)
    assert all("name" in r and "module" in r for r in data)


def test_card_json_is_valid():
    rc, out, err = _run(["card", str(SAMPLES / "tone.wav"), "--json"])
    assert rc == 0, err
    data = json.loads(out)
    assert data["schema_version"] == "1.0"
    assert data["object"]["kind"] == "sound"


def test_analyze_json_is_valid():
    rc, out, err = _run(["analyze", str(SAMPLES / "payments.csv"), "--json"])
    assert rc == 0, err
    data = json.loads(out)
    assert "needs" in data
    assert "primitives" in data


def test_ontology_gaps_json_is_valid():
    rc, out, err = _run(["ontology", "gaps", "--json"])
    assert rc == 0, err
    data = json.loads(out)
    assert "n_rows" in data
    assert "without_link" in data
    assert data["n_without_link"] == data["n_classes"] - data["n_with_link"]


# --------------------------------------------------------------
# Exit codes documentati
# --------------------------------------------------------------
def test_exit_help_is_zero():
    rc, _out, _err = _run(["help"])
    assert rc == 0


def test_exit_nonexistent_file_is_one():
    rc, _out, _err = _run(["analyze", "/nonexistent/file.wav"])
    assert rc == 1


def test_exit_bad_arguments_is_two():
    # `ontology` without `gaps` is an argument error
    rc, _out, _err = _run(["ontology"])
    assert rc == 2


def test_exit_unknown_command_is_not_zero():
    rc, _out, _err = _run(["nonsense"])
    assert rc != 0


# --------------------------------------------------------------
# ruff / mypy: strumenti di sviluppo, skip se assenti
# --------------------------------------------------------------
def test_ruff_check_passes():
    if shutil.which("ruff") is None:
        pytest.skip("ruff not installed (dev tool, not a runtime dependency)")
    rc = subprocess.run(
        ["ruff", "check", "dense_armor/usable/", "--exclude", "__pycache__"],
        cwd=REPO, capture_output=True, text=True,
    ).returncode
    assert rc == 0


def test_ruff_format_check_passes():
    if shutil.which("ruff") is None:
        pytest.skip("ruff not installed (dev tool, not a runtime dependency)")
    rc = subprocess.run(
        ["ruff", "format", "--check", "dense_armor/usable/", "--exclude", "__pycache__"],
        cwd=REPO, capture_output=True, text=True,
    ).returncode
    assert rc == 0


def test_mypy_passes():
    if shutil.which("mypy") is None:
        pytest.skip("mypy not installed (dev tool, not a runtime dependency)")
    r = subprocess.run(
        ["mypy", "dense_armor/usable", "--follow-imports=skip"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
