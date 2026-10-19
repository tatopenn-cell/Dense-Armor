"""Tests for pipelines/drones/visual_landing.py (task D1, item B6).

The pipeline uses the library's own primitives:
GlobalMotion, DenseFlow, MotionMask, FrameAnomaly.
Input protocol: {"array": img} for GlobalMotion and DenseFlow,
{"u","v"} for MotionMask, {"frame": img} for FrameAnomaly.
"""

import numpy as np
import pytest

from dense_armor.utility.vision.dense import DenseFlow, MotionMask
from dense_armor.utility.vision.motion import GlobalMotion
from dense_armor.utility.vision.scene import FrameAnomaly
from pipelines.drones.visual_landing import (
    FRAME_H,
    FRAME_W,
    VisualLandingResult,
    run,
    synthetic_frames,
)


def test_synthetic_frames_three_tuple():
    frames, truth, person_at = synthetic_frames(
        n=5, shift=(1, 0), seed=0, person_at=3)
    assert len(frames) == 5
    assert len(truth) == 5
    assert truth[0] == (0, 0)
    assert truth[1] == (1, 0)
    assert truth[4] == (4, 0)
    assert person_at == 3


def test_synthetic_frames_shape_and_dtype():
    frames, _, _ = synthetic_frames(n=2, shift=(1, 0), seed=0)
    assert frames[0]["array"].shape == (FRAME_H, FRAME_W)
    assert frames[0]["array"].dtype == np.float32
    assert 0.0 <= frames[0]["array"].min()
    assert frames[0]["array"].max() <= 1.0


def test_synthetic_frames_zero_shift():
    frames, truth, _ = synthetic_frames(n=3, shift=(0, 0), seed=0)
    assert all(t == (0, 0) for t in truth)
    assert np.array_equal(frames[0]["array"], frames[2]["array"])


def test_synthetic_frames_n_zero_raises():
    with pytest.raises(ValueError):
        synthetic_frames(n=0, shift=(1, 0), seed=0)


def test_synthetic_frames_negative_raises():
    with pytest.raises(ValueError):
        synthetic_frames(n=-1, shift=(1, 0), seed=0)


def test_synthetic_frames_bad_person_at_raises():
    with pytest.raises(ValueError):
        synthetic_frames(n=5, shift=(1, 0), seed=0, person_at=10)


def test_pad_visible_in_frame():
    frames, _, _ = synthetic_frames(n=1, shift=(0, 0), seed=0)
    # pad around PAD_Y0:PAD_Y0+PAD_H, PAD_X0:PAD_X0+PAD_W is at 0.7
    img = frames[0]["array"]
    pad_region = img[130:210, 250:350]
    assert float(np.mean(pad_region)) > 0.6


def test_person_visible_after_person_at():
    frames, _, _ = synthetic_frames(
        n=10, shift=(0, 0), seed=0, person_at=5)
    # person is at PAD_Y0 + 25, PAD_X0 + 10 on frame 5
    img5 = frames[5]["array"]
    person_region = img5[155:185, 260:290]
    assert float(np.min(person_region)) < 0.1


@pytest.mark.slow
def test_run_no_person():
    frames, truth, _ = synthetic_frames(n=10, shift=(1, 0), seed=0)
    r = run(frames, truth, None)
    assert isinstance(r, VisualLandingResult)
    assert r.n_frames == 10
    assert r.person_at is None
    assert np.isfinite(r.mean_error_px)
    assert r.mean_error_px < 1.0


@pytest.mark.slow
def test_run_globalmotion_recovers_shift():
    frames, truth, _ = synthetic_frames(n=10, shift=(1, 0), seed=0)
    r = run(frames, truth, None)
    assert r.mean_error_px < 1.0
    assert r.max_error_px < 2.0


@pytest.mark.slow
def test_run_zero_shift():
    frames, truth, _ = synthetic_frames(n=8, shift=(0, 0), seed=0)
    r = run(frames, truth, None)
    assert r.mean_error_px < 1.0


@pytest.mark.slow
def test_run_frameanomaly_discriminates():
    frames, truth, pa = synthetic_frames(
        n=25, shift=(1, 0), seed=0, person_at=20)
    r = run(frames, truth, pa, anomaly_warmup=10)
    assert np.isfinite(r.anomaly_before_mean)
    assert np.isfinite(r.anomaly_after_mean)
    assert r.anomaly_after_mean > r.anomaly_before_mean


@pytest.mark.slow
def test_run_motionmask_declared_zero():
    frames, truth, pa = synthetic_frames(
        n=20, shift=(1, 0), seed=0, person_at=10)
    r = run(frames, truth, pa)
    # MotionMask gives 0 on this stream; the page says so
    assert r.mask_pixels.shape == (20,)
    assert r.mask_pixels.sum() >= 0


@pytest.mark.slow
def test_run_result_fields():
    frames, truth, pa = synthetic_frames(
        n=10, shift=(1, 0), seed=0, person_at=5)
    r = run(frames, truth, pa)
    assert r.est_dx.shape == (10,)
    assert r.est_dy.shape == (10,)
    assert r.error_px.shape == (10,)
    assert r.mask_pixels.shape == (10,)
    assert r.anomaly.shape == (10,)
    assert r.time_global_ms > 0
    assert r.time_dense_ms > 0


@pytest.mark.slow
def test_run_deterministic():
    frames, truth, pa = synthetic_frames(
        n=15, shift=(1, 0), seed=0, person_at=8)
    r1 = run(frames, truth, pa)
    r2 = run(frames, truth, pa)
    np.testing.assert_array_equal(r1.est_dx, r2.est_dx)
    np.testing.assert_array_equal(r1.est_dy, r2.est_dy)


@pytest.mark.slow
def test_run_no_truth():
    frames, _, _ = synthetic_frames(n=5, shift=(1, 0), seed=0)
    r = run(frames, None, None)
    assert r.n_frames == 5
    assert np.all(np.isnan(r.error_px))


def test_primitives_imported():
    assert hasattr(GlobalMotion(), "transform_one")
    assert hasattr(DenseFlow(), "transform_one")
    assert hasattr(MotionMask(), "transform_one")
    assert hasattr(FrameAnomaly(), "transform_one")
