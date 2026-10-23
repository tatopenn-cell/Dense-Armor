"""Tests for benchmarks/neural/check_examples.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "neural" / "check_examples.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("check_examples", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_examples"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_ok_when_class_in_matching_block(tmp_path):
    mod = _load_module()
    page = tmp_path / "index.md"
    page.write_text(
        "## Header\n\n"
        "~~~python\n"
        "class PLACEHOLDER:\n"
        "    pass\n"
        "print(2)\n"
        "~~~\n\n"
        "~~~\n"
        "2\n"
        "~~~\n",
        encoding="utf-8",
    )
    classes, blocks = mod.check_page(page, ["PLACEHOLDER"])
    assert classes == [("PLACEHOLDER", True)]
    assert len(blocks) == 1
    assert blocks[0][1] is True
    assert blocks[0][2] == "2"
    assert blocks[0][3] == "2"


def test_mismatch_when_output_differs(tmp_path):
    mod = _load_module()
    page = tmp_path / "index.md"
    page.write_text(
        "~~~python\nclass PLACEHOLDER:\n    pass\nprint(2)\n~~~\n\n~~~\n3\n~~~\n",
        encoding="utf-8",
    )
    classes, blocks = mod.check_page(page, ["PLACEHOLDER"])
    assert classes == [("PLACEHOLDER", True)]
    assert len(blocks) == 1
    assert blocks[0][1] is False
    assert blocks[0][3] == "2"


def test_missing_when_class_only_in_prose(tmp_path):
    mod = _load_module()
    page = tmp_path / "index.md"
    page.write_text(
        "PLACEHOLDER is mentioned only in prose.\n",
        encoding="utf-8",
    )
    classes, blocks = mod.check_page(page, ["PLACEHOLDER"])
    assert classes == [("PLACEHOLDER", False)]
    assert blocks == []


def test_page_not_found(tmp_path):
    mod = _load_module()
    missing = tmp_path / "nope.md"
    classes, blocks = mod.check_page(missing, ["Sigmoid", "Tanh"])
    assert classes == [("Sigmoid", False), ("Tanh", False)]
    assert blocks == []
