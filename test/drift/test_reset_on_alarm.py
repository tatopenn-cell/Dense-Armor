"""Tests for reset_on_alarm and n_alarms_ in CUSUMDriftDetector."""

import numpy as np

from dense_armor.utility.drift.detector import CUSUMDriftDetector


def _step_stream(n=500, shift_at=250, delta=3.0, seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [rng.normal(0.0, 1.0, shift_at), rng.normal(delta, 1.0, n - shift_at)]
    )


def test_default_behaviour_is_unchanged():
    """With reset_on_alarm=False the output matches a run with the flag off."""
    x = _step_stream()
    a = CUSUMDriftDetector(reference="fixed", reset_on_alarm=False)
    b = CUSUMDriftDetector(reference="fixed")  # default
    flags_a = []
    flags_b = []
    for xi in x:
        a.update(float(xi))
        b.update(float(xi))
        flags_a.append(a.drift_detected)
        flags_b.append(b.drift_detected)
    assert flags_a == flags_b
    assert a.n_alarms_ == b.n_alarms_


def test_one_alarm_on_a_step_with_reset():
    """A single step of the mean gives one alarm, not one per sample."""
    x = _step_stream(n=500, shift_at=250, delta=3.0)
    det = CUSUMDriftDetector(reference="fixed", reset_on_alarm=True)
    for xi in x:
        det.update(float(xi))
    # UCB del test: un solo allarme, o al massimo due (edge + rientro)
    assert det.n_alarms_ <= 2, f"troppi allarmi: {det.n_alarms_}"


def test_many_alarms_without_reset():
    """Without reset the same stream produces more alarms."""
    x = _step_stream(n=500, shift_at=250, delta=3.0)
    det_off = CUSUMDriftDetector(reference="fixed", reset_on_alarm=False)
    det_on = CUSUMDriftDetector(reference="fixed", reset_on_alarm=True)
    for xi in x:
        det_off.update(float(xi))
        det_on.update(float(xi))
    assert det_off.n_alarms_ > det_on.n_alarms_


def test_adaptive_reference_with_reset_clears_window():
    """With adaptive reference the reset clears the window."""
    det = CUSUMDriftDetector(reference="adaptive", reset_on_alarm=True)
    rng = np.random.default_rng(1)
    for _ in range(30):
        det.update(float(rng.normal(0.0, 0.01)))
    assert len(det._buffer) > 0
    det._on_reset()
    assert len(det._buffer) == 0


def test_n_alarms_is_zero_before_any_alarm():
    det = CUSUMDriftDetector()
    assert det.n_alarms_ == 0
    for v in [0.0, 0.1, -0.1, 0.05]:
        det.update(v)
    assert det.n_alarms_ == 0


def test_reset_restores_n_alarms():
    det = CUSUMDriftDetector()
    det.n_alarms_ = 5
    det._reset()
    assert det.n_alarms_ == 0


def test_n_missing_property_still_works():
    det = CUSUMDriftDetector()
    det.update(float("nan"))
    assert det.n_missing == 1
    assert det.n_missing_ == 1


def test_nan_does_not_count_as_alarm():
    det = CUSUMDriftDetector(reference="fixed")
    for _ in range(5):
        det.update(float("nan"))
    assert det.n_alarms_ == 0
    assert det.n_missing_ == 5
