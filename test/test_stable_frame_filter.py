# -*- coding: utf-8 -*-
"""
Unit tests for dense_armor/utility/stable_frame_filter.py -- promoted
from Dense-Evolution-Discovery after validation on two independent real
physical domains (a teleoperated robot arm's leader-position channel,
and human IMU gyroscope/accelerometer data). Previously only exercised
indirectly through that real experiment data.
"""
import numpy as np
import pytest

from dense_armor.utility.stable_frame_filter import velocity_gated_stable_mask


def test_1d_flags_slow_changing_region_stable():
    x = np.array([0.0, 0.0, 0.0, 10.0, 20.0, 20.0, 20.0])
    mask = velocity_gated_stable_mask(x, vel_threshold=1.0)
    assert mask.tolist() == [True, True, True, False, False, True, True]


def test_2d_requires_every_channel_stable():
    x = np.array([
        [0.0, 0.0],
        [5.0, 0.1],
        [5.1, 0.2],
    ])
    mask = velocity_gated_stable_mask(x, vel_threshold=1.0)
    assert mask.tolist() == [True, False, True]


def test_first_frame_is_always_stable_by_construction():
    x = np.array([100.0, 100.0, 100.0])
    mask = velocity_gated_stable_mask(x, vel_threshold=0.5)
    assert mask[0]


def test_raises_on_invalid_dimensionality():
    with pytest.raises(ValueError):
        velocity_gated_stable_mask(np.zeros((2, 2, 2)), vel_threshold=1.0)


def test_already_rate_thresholds_directly_not_via_diff():
    """already_rate=True: reference IS already a rate (e.g. angular
    velocity magnitude) -- thresholded on its own value, not on its
    diff. A constant-but-large reference should be entirely UNstable
    under already_rate=True (every value exceeds the threshold), the
    opposite of already_rate=False's behavior on the same array (a
    constant reference has zero velocity, so it would be entirely
    stable)."""
    x = np.array([5.0, 5.0, 5.0, 5.0])
    mask_rate = velocity_gated_stable_mask(x, vel_threshold=1.0, already_rate=True)
    assert not mask_rate.any(), "a constant reference of 5.0 must be UNstable when thresholded directly at 1.0"
    mask_diff = velocity_gated_stable_mask(x, vel_threshold=1.0, already_rate=False)
    assert mask_diff.all(), "a constant reference has zero velocity, so it must be stable under the diff-based default"


def test_already_rate_2d_requires_every_channel_below_threshold():
    x = np.array([
        [0.1, 0.1],
        [2.0, 0.1],
        [0.1, 0.1],
    ])
    mask = velocity_gated_stable_mask(x, vel_threshold=1.0, already_rate=True)
    assert mask.tolist() == [True, False, True]


def test_real_detector_composition_smoke():
    """Composes with an actual detector (classify_segments), not just a
    synthetic flags array -- the real intended usage pattern: gate an
    analyzed signal's anomaly labels by whether a companion reference
    was stable at each point."""
    from dense_armor.utility.arbiter import classify_segments

    rng = np.random.default_rng(1)
    reference = rng.normal(0.0, 0.1, 200)  # e.g. a real command/position channel
    analyzed = rng.normal(2.0, 0.3, 200)
    reference[100] = 20.0  # reference moving fast at this frame -- a real confound
    labels, _, _ = classify_segments(analyzed, radius=10, ref_mult=3)
    flags = labels != "clean"

    mask = velocity_gated_stable_mask(reference, vel_threshold=1.0)
    assert not mask[100], "the frame where the reference moved fast must be excluded"
    trusted_flags = flags & mask
    assert not trusted_flags[100], "an anomaly flag at an unstable reference frame must be discounted"
