"""Tests for pipelines/drones/visual_landing.py (task D1, item B6)."""

import numpy as np
import pytest

from dense_armor.utility.vision.motion import GlobalMotion, MotionEnergy
from pipelines.drones.visual_landing import (
    FRAME_H,
    FRAME_W,
    VisualLandingResult,
    run,
    synthetic_frames,
)


def test_synthetic_frames_count_and_truth():
    frames, truth = synthetic_frames(n=5, shift=(4, -2), seed=0)
    assert len(frames) == 5
    assert len(truth) == 5
    assert truth[0] == (0, 0)
    assert truth[1] == (4, -2)
    assert truth[4] == (16, -8)


def test_synthetic_frames_shape():
    frames, _ = synthetic_frames(n=2, shift=(1, 0), seed=0)
    assert frames[0]["array"].shape == (FRAME_H, FRAME_W)
    assert frames[0]["array"].dtype == np.uint8


def test_synthetic_frames_zero_shift():
    frames, truth = synthetic_frames(n=3, shift=(0, 0), seed=0)
    assert all(t == (0, 0) for t in truth)
    assert np.array_equal(frames[0]["array"], frames[2]["array"])


def test_synthetic_frames_n_zero_raises():
    with pytest.raises(ValueError):
        synthetic_frames(n=0, shift=(1, 0), seed=0)


def test_synthetic_frames_negative_raises():
    with pytest.raises(ValueError):
        synthetic_frames(n=-1, shift=(1, 0), seed=0)


def test_pad_visible_in_first_frame():
    frames, _ = synthetic_frames(n=1, shift=(0, 0), seed=0)
    assert (frames[0]["array"] > 0).sum() > 4000


def test_run_zero_error_for_pure_translation():
    for shift in [(4, -2), (1, 0), (10, 5)]:
        frames, truth = synthetic_frames(n=8, shift=shift, seed=0)
        r = run(frames, truth)
        assert isinstance(r, VisualLandingResult)
        assert r.true_dx == shift[0]
        assert r.true_dy == shift[1]
        assert r.mean_error_px < 0.5
        assert r.max_error_px < 0.5


def test_run_zero_shift():
    frames, truth = synthetic_frames(n=5, shift=(0, 0), seed=0)
    r = run(frames, truth)
    assert r.mean_error_px < 0.5


def test_run_no_truth():
    frames, _ = synthetic_frames(n=3, shift=(4, -2), seed=0)
    r = run(frames, None)
    assert r.n_frames == 3
    assert np.all(np.isfinite(r.est_dx[1:]))
    assert np.all(np.isnan(r.error_px))


def test_run_motion_sum_zero_for_static():
    frames, truth = synthetic_frames(n=5, shift=(0, 0), seed=0)
    r = run(frames, truth)
    assert np.nanmax(r.motion_sum[1:]) < 1.0


def test_run_motion_sum_grows_with_shift():
    small_frames, small_truth = synthetic_frames(n=4, shift=(1, 0), seed=0)
    large_frames, large_truth = synthetic_frames(n=4, shift=(10, 5), seed=0)
    r_small = run(small_frames, small_truth)
    r_large = run(large_frames, large_truth)
    assert np.nanmedian(r_large.motion_sum) > np.nanmedian(r_small.motion_sum)


def test_run_estimates_close_to_true():
    frames, truth = synthetic_frames(n=6, shift=(4, -2), seed=0)
    r = run(frames, truth)
    np.testing.assert_allclose(r.est_dx[1:], 4.0, atol=0.5)
    np.testing.assert_allclose(r.est_dy[1:], -2.0, atol=0.5)


def test_run_time_measured():
    frames, truth = synthetic_frames(n=3, shift=(4, -2), seed=0)
    r = run(frames, truth)
    assert r.time_per_frame_ms > 0
    assert r.time_per_frame_ms < 1000


def test_globalmotion_and_motionenergy_used():
    gm = GlobalMotion()
    assert hasattr(gm, "transform_one")
    me = MotionEnergy()
    assert hasattr(me, "transform_one")
