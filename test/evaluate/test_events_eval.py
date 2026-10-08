"""Tests for dense_armor/utility/evaluate/events.py."""
import doctest
import importlib.util
from pathlib import Path

import numpy as np
import pytest

import dense_armor.utility.evaluate.events as ev_mod
from dense_armor.utility.drift.detector import CUSUMDriftDetector
from dense_armor.utility.evaluate import evaluate_events

BENCH = Path(__file__).resolve().parents[2] / "benchmarks" / "casper_benchmark.py"


def _casper():
    pytest.importorskip("pandas")
    spec = importlib.util.spec_from_file_location("casper_benchmark", BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("h", [2.0, 5.0, 20.0])
def test_same_numbers_as_casper_run_cusum(h):
    cb = _casper()
    rng = np.random.default_rng(1)
    n, c = 4000, 3000
    x = np.concatenate([rng.normal(0, 1, c), rng.normal(3, 1, n - c)])
    delay, fa_h = cb.run_cusum(x, c, h=h)
    det = CUSUMDriftDetector(
        reference="fixed", radius=cb.CUSUM_RADIUS, ref_mult=cb.CUSUM_REF_1C,
        k=cb.CUSUM_K_DEFAULT, h=h, two_sided=False,
    )
    fs = cb.FS
    r = evaluate_events(
        [(i / fs, v) for i, v in enumerate(x)], det, [(c / fs, (n - 1) / fs)],
        dt=1 / fs, normal_time_s=c / fs,
    )
    assert r["detection_delays_s"][0] * fs == pytest.approx(delay)
    assert r["false_alarms_per_hour"] == pytest.approx(fa_h)


class _Score:
    score_threshold = 0.5

    def __init__(self):
        self.seen = 0

    def score_one(self, x, t=None):
        return 1.0 if x["v"] > 10 else 0.0

    def learn_one(self, x, t=None):
        self.seen += 1
        return self


def test_anomaly_detector_path():
    det = _Score()
    stream = [(float(i), {"v": float(i)}) for i in range(20)]
    r = evaluate_events(stream, det, [(12.0, 15.0)])
    assert det.seen == 20
    assert r["detection_delays_s"] == [0.0]
    assert r["false_alarms"] == 5


def test_anomaly_detector_needs_threshold():
    class _NoThr(_Score):
        score_threshold = None

    with pytest.raises(ValueError):
        evaluate_events([(0.0, {"v": 0.0})], _NoThr(), [(0.0, 1.0)])


def test_unknown_detector_rejected():
    with pytest.raises(TypeError):
        evaluate_events([(0.0, 0.0)], object(), [(0.0, 1.0)])


def test_doctests():
    assert doctest.testmod(ev_mod).failed == 0
