import numpy as np

from dense_armor.utility.drift.adwin import ADWIN


def _alarms(x, **kw):
    det = ADWIN(**kw)
    out = []
    for i, v in enumerate(x):
        det.update(float(v))
        if det.drift_detected:
            out.append(i)
    return out


def test_no_alarm_on_a_stationary_stream():
    rng = np.random.default_rng(3)
    assert _alarms(rng.normal(0, 1, 2000)) == []


def test_a_step_is_flagged_soon_and_the_burst_is_short():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, 500), rng.normal(3, 1, 500)])
    hits = _alarms(x)
    assert hits and hits[0] >= 500
    assert hits[0] - 500 <= 100
    assert hits[-1] - hits[0] <= 100


def test_a_new_detector_after_the_first_flag_gives_one_event_per_change():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, 500), rng.normal(3, 1, 500)])
    det = ADWIN()
    events = []
    for i, v in enumerate(x):
        det.update(float(v))
        if det.drift_detected:
            events.append(i)
            det = ADWIN()
    assert len(events) == 1 and 500 <= events[0] <= 600
