"""Regression tests for the fixes of the review: tracker, EventFrame, FrameQuality."""

import numpy as np

from dense_armor.core.signals.vision.quality import EventFrame, FrameQuality
from dense_armor.core.signals.vision.tracking import _default_kf, _Track


def test_new_track_predicts_its_own_box():
    box = (500.0, 300.0, 520.0, 320.0)
    track = _Track(1, box, 1.0, _default_kf())
    np.testing.assert_allclose(track.predict(), box, atol=1e-6)


def test_missing_track_is_extrapolated_with_its_velocity():
    track = _Track(1, (10.0, 10.0, 20.0, 20.0), 1.0, _default_kf())
    for k in range(1, 6):
        track.update((10.0 + 2 * k, 10.0, 20.0 + 2 * k, 20.0), 1.0)
    seen = track.predict()
    track.missing = 3
    extrapolated = track.predict()
    assert extrapolated[0] > seen[0]


def test_event_frame_with_decay_one_keeps_the_events():
    frame = EventFrame(shape=(8, 8), decay=1.0)
    for _ in range(3):
        frame.learn_one({"x": 4, "y": 2, "polarity": 1, "t": 0.0})
    first = frame.transform_one(None)
    second = frame.transform_one(None)
    assert int(first["frame"][2, 4]) == 3
    assert int(second["frame"][2, 4]) == 3
    assert first["n_events"] == 3
    assert second["n_events"] == 0


def test_n_events_counts_events_of_opposite_polarity():
    frame = EventFrame(shape=(8, 8), decay=0.5)
    frame.learn_one({"x": 1, "y": 1, "polarity": 1, "t": 0.0})
    frame.learn_one({"x": 1, "y": 1, "polarity": -1, "t": 0.0})
    out = frame.transform_one(None)
    assert out["n_events"] == 2
    assert float(out["frame"][1, 1]) == 0.0


def test_sharpness_of_a_smooth_ramp_is_almost_zero():
    ramp = np.tile(np.linspace(0.0, 1.0, 16), (16, 1)).astype(np.float32)
    assert FrameQuality().transform_one({"array": ramp})["sharpness"] < 0.01
