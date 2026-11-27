"""The arbiter with a detector of the library: a point detector decides which samples deviate, an event detector confirms a regime."""

import numpy as np
import pytest

from dense_armor.core.online.anomaly.filters import HampelScorer, TukeyScorer
from dense_armor.core.online.drift import ADWIN, BayesianOnlineChangepoint
from dense_armor.core.online.protect.arbiter import classify_segments, route_and_correct
from dense_armor.core.online.protect.streaming_arbiter import StreamingArbiter


def _series():
    r = np.random.default_rng(0)
    x = r.normal(0, 1, 300)
    x[60] += 15.0
    x[150:200] += 8.0
    return x


class _Event:
    """An event detector that signals at the indexes it is given."""

    drift_detected = False

    def __init__(self, at):
        self.at, self.i = set(at), -1

    def update(self, v, t=None):
        self.i += 1
        self.drift_detected = self.i in self.at
        return self


def test_no_detector_is_the_same_as_before():
    x = _series()
    a, b = classify_segments(x), classify_segments(x, detector=None, threshold=None)
    assert all(np.array_equal(u, v) for u, v in zip(a, b))
    assert all(np.array_equal(u, v) for u, v in zip(route_and_correct(x), route_and_correct(x, detector=None)))


def test_an_object_that_is_not_a_detector_is_not_used():
    x = _series()
    a = classify_segments(x)
    b = classify_segments(x, detector=object())
    assert all(np.array_equal(u, v) for u, v in zip(a, b))


@pytest.mark.parametrize("det", [HampelScorer(), TukeyScorer()], ids=lambda d: type(d).__name__)
def test_a_point_detector_finds_the_spike_and_the_regime(det):
    labels, dev, unc = classify_segments(_series(), detector=det, threshold=3.0)
    assert labels[60] == "spike"
    assert all(labels[i] == "regime" for i in range(150, 154))
    assert dev[60] > 3.0 and 0.0 <= unc.min() and unc.max() <= 1.0


def test_the_threshold_is_the_one_given():
    x = _series()
    labels, _, _ = classify_segments(x, detector=HampelScorer(), threshold=1e9)
    assert set(labels) == {"clean"}
    assert classify_segments(x, detector=HampelScorer())[0][60] == "spike"


def test_the_detector_given_is_not_changed():
    det = HampelScorer()
    before = det.score_one({"x": 100.0})
    classify_segments(_series(), detector=det)
    assert det.score_one({"x": 100.0}) == before


def test_non_finite_samples_are_clean_and_are_not_given_to_the_detector():
    x = _series()
    x[30] = np.nan
    labels, _, _ = classify_segments(x, detector=HampelScorer())
    assert labels[30] == "clean" and labels[60] == "spike"


def test_an_event_detector_that_signals_in_a_run_makes_it_a_regime():
    x = _series()
    assert classify_segments(x)[0][60] == "spike"
    assert classify_segments(x, detector=_Event(at=[60]))[0][60] == "regime"
    assert classify_segments(x, detector=_Event(at=[30]))[0][60] == "spike"


def test_a_real_event_detector_that_does_not_fire_changes_nothing():
    x = _series()
    assert all(np.array_equal(u, v) for u, v in zip(classify_segments(x), classify_segments(x, detector=ADWIN())))


def test_bayesian_online_changepoint_signals_on_the_spike_and_turns_it_into_a_regime():
    assert classify_segments(_series(), detector=BayesianOnlineChangepoint())[0][60] == "regime"


def _stream(sa, x):
    out = []
    for v in x:
        out.extend(sa.update(float(v)))
    out.extend(sa.flush())
    return [lab for _, lab, _ in sorted(out)]


def test_the_streaming_arbiter_with_a_detector_equals_the_batch_one():
    x = _series()
    batch = list(classify_segments(x, detector=HampelScorer(), threshold=3.0)[0])
    assert _stream(StreamingArbiter(max_delay=400, detector=HampelScorer(), threshold=3.0), x) == batch


def test_the_streaming_arbiter_without_a_detector_is_as_before():
    x = _series()
    assert _stream(StreamingArbiter(max_delay=400), x) == list(classify_segments(x)[0])


def test_the_streaming_arbiter_with_an_event_detector():
    x = _series()
    batch = list(classify_segments(x, detector=_Event(at=[60]))[0])
    assert _stream(StreamingArbiter(max_delay=400, detector=_Event(at=[60])), x) == batch
