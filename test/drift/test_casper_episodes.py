"""Unit test for count_alarms in casper_hours_episodes (no dataset)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

from dense_armor.utility.drift.detector import CUSUMDriftDetector

HERE = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "casper_hours_episodes",
    HERE / "benchmarks" / "casper_hours_episodes.py",
)
mod = importlib.util.module_from_spec(spec)
sys.modules["casper_hours_episodes"] = mod
spec.loader.exec_module(mod)


def test_count_alarms_one_episode_on_a_step():
    rng = np.random.default_rng(0)
    s = np.concatenate(
        [rng.normal(0.0, 0.01, 500), rng.normal(3.0, 0.01, 500)]
    )
    det = CUSUMDriftDetector(reference="fixed", reset_on_alarm=True)
    flagged, episodes = mod.count_alarms(det, s)
    assert episodes <= 2
    assert flagged == episodes  # reset_on_alarm=True: una flag per episodio


def test_count_alarms_many_flags_without_reset():
    rng = np.random.default_rng(0)
    s = np.concatenate(
        [rng.normal(0.0, 0.01, 500), rng.normal(3.0, 0.01, 500)]
    )
    det = CUSUMDriftDetector(reference="fixed", reset_on_alarm=False)
    flagged, episodes = mod.count_alarms(det, s)
    assert flagged >= episodes
    assert episodes >= 1


def test_count_alarms_zero_on_flat_stream():
    rng = np.random.default_rng(0)
    s = rng.normal(0.0, 0.01, 1000)
    det = CUSUMDriftDetector(reference="fixed", reset_on_alarm=False)
    flagged, episodes = mod.count_alarms(det, s)
    assert flagged == 0
    assert episodes == 0


def test_smoke_without_dataset_is_clean(capsys):
    """The parser accepts --smoke and does not require the dataset to parse."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", default="0-14")
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="x")
    a = ap.parse_args(["--smoke"])
    assert a.smoke is True
