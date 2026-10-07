
"""Unit tests for dense_armor/utility/river_drift.py."""
import importlib
import sys

import numpy as np
import pytest

pytest.importorskip("river")

from dense_armor.drift.cusum import cusum_detector
from dense_armor.drift.detector import CUSUMDriftDetector


def test_equivalence_with_batch_adaptive():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 1000)
    x[300:] += 2.0
    fb, _ = cusum_detector(x, reference="adaptive")
    det = CUSUMDriftDetector(reference="adaptive")
    fs = [det.update(v).drift_detected for v in x]
    assert np.array_equal(fb, np.array(fs))


def test_equivalence_with_batch_fixed():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 1000)
    x[300:] += 2.0
    fb, _ = cusum_detector(x, reference="fixed")
    det = CUSUMDriftDetector(reference="fixed")
    fs = [det.update(v).drift_detected for v in x]
    assert np.array_equal(fb, np.array(fs))


def test_no_drift_on_stationary_series(capsys):
    rng = np.random.default_rng(123)
    x = rng.normal(0, 1, 5000)
    for mode in ("adaptive", "fixed"):
        det = CUSUMDriftDetector(reference=mode)
        fa = sum(1 for v in x if (det.update(v), det.drift_detected)[1])
        predicted = 5000 / det.expected_false_alarm_run()
        with capsys.disabled():
            print(f"[stationary/{mode}] false alarms={fa}, ARL-predicted~{predicted:.2f}")
        assert fa < 5


def test_detection_delay_matches_arl_fixed(capsys):
    rng = np.random.default_rng(456)
    x = rng.normal(0, 1, 1000)
    x[500:] += 1.0
    det = CUSUMDriftDetector(reference="fixed")
    detected_at = None
    for i, v in enumerate(x):
        det.update(v)
        if det.drift_detected and i >= 500:
            detected_at = i
            break
    assert detected_at is not None, "fixed reference should detect a 1-sigma shift"
    delay = detected_at - 500
    predicted = det.expected_detection_delay(1.0)
    with capsys.disabled():
        print(f"[1-sigma/fixed] delay={delay}, ARL-predicted={predicted:.2f}")
    assert abs(delay - predicted) < 0.5 * predicted


def test_clone_and_pickle_keep_parameters():
    import pickle

    det = CUSUMDriftDetector(k=0.7, h=12.0, reference="fixed")
    for v in np.random.default_rng(0).normal(0, 1, 50):
        det.update(v)
    clone = det.clone()
    assert (clone.k, clone.h, clone.reference) == (0.7, 12.0, "fixed")
    assert pickle.loads(pickle.dumps(det)).h == 12.0


def test_missing_river_raises_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "river", None)
    monkeypatch.delitem(sys.modules, "dense_armor.drift.detector", raising=False)
    with pytest.raises(ModuleNotFoundError, match=r"dense-armor\[river\]"):
        importlib.import_module("dense_armor.drift.detector")
