"""Smoke test for benchmarks/seed_audit.py (does not run the full audit)."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "seed_audit", HERE / "benchmarks" / "seed_audit.py"
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["seed_audit"] = m
    spec.loader.exec_module(m)
    return m


def test_walk_estimators_returns_some():
    m = _load()
    est = m._walk_estimators()
    assert len(est) > 50


def test_seed_info_of_hoeffding_tree_has_no_seed():
    m = _load()
    from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier
    has, default, ann = m._seed_info(HoeffdingTreeClassifier)
    assert has is False
    assert default == "<absent>"


def test_seed_info_of_arf_has_seed_none():
    m = _load()
    from dense_armor.utility.forest.arf import AdaptiveRandomForestClassifier
    has, default, ann = m._seed_info(AdaptiveRandomForestClassifier)
    assert has is True
    assert default is None
