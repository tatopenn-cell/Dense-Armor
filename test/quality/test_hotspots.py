"""Smoke test for benchmarks/hotspots.py."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "hotspots", HERE / "benchmarks" / "hotspots.py"
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["hotspots"] = m
    spec.loader.exec_module(m)
    return m


def test_import_and_help():
    m = _load()
    assert hasattr(m, "main")
    assert hasattr(m, "_profile")
