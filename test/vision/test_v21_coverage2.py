"""Remaining branches of the video modules: argument checks of the streams, fallbacks, tiny and flat inputs."""

import numpy as np
import pytest

from dense_armor.utility.vision import motion, streams
from dense_armor.utility.vision.adapters import FeatureToSignal
from dense_armor.utility.vision.dense import DenseFlow
from dense_armor.utility.vision.fusion import FrameJointFusion
from dense_armor.utility.vision.landmarks import Kinematics
from dense_armor.utility.vision.motion import GlobalMotion, MotionEnergy
from dense_armor.utility.vision.quality import FrameQuality, Sync
from dense_armor.utility.vision.registration import DifferenceImage, Registration
from dense_armor.utility.vision.robustness import FrameNormalizer
from dense_armor.utility.vision.scene import FrameAnomaly
from dense_armor.utility.vision.streams import ArrayStream, CubeStream, Frame
from dense_armor.utility.vision.tracking import MovingObjectTracker
from dense_armor.utility.vision.visualize import Overlay
from dense_armor.utility.vision.window import FrameWindow


def _frames(n=2, shape=(24, 24)):
    rng = np.random.default_rng(0)
    return [Frame(array=rng.normal(0.4, 0.02, shape).astype(np.float32), t=0.1 * i) for i in range(n)]


def test_stream_argument_checks_and_the_interface():
    with pytest.raises(ValueError):
        ArrayStream(np.zeros((2, 4, 4)), dtype="int64")
    for kw in ({"fps": 0.0}, {"stride": 0}, {"start": -1}, {"start": 5, "stop": 2}):
        with pytest.raises(ValueError):
            streams.VideoStream(**kw)
    with pytest.raises(NotImplementedError):
        iter(streams.VideoStream())
    cube = np.random.default_rng(0).random((3, 6, 6, 3)).astype(np.float32)
    assert list(CubeStream(cube, size=(3, 3)))[0].array.shape[:2] == (3, 3)


def test_fuse_frames_joints_merges_in_time_order():
    from dense_armor.roles import Signal
    from dense_armor.utility.vision.fusion import fuse_frames_joints

    def a():
        for t in (0.0, 0.2):
            yield Signal(np.array([1.0]), ["frame"], [""], t=t), None

    def b():
        for t in (0.1, 0.3):
            yield Signal(np.array([2.0]), ["q0"], ["rad"], t=t), None

    assert [round(s.t, 2) for s, _ in fuse_frames_joints(a(), b())] == [0.0, 0.1, 0.2, 0.3]


def test_fusion_estimate_with_flat_or_stepped_signals():
    f = FrameJointFusion(max_lag_s=0.5, window_s=10.0, frame_rate_hz=10.0)
    for i in range(12):
        f.add_video_velocity(1.0, i * 0.1)
        f.add_joint_speed(float(i), i * 0.1)
    assert f.estimate()["correlation"] == 0.0
    g = FrameJointFusion(max_lag_s=1.0, window_s=10.0, frame_rate_hz=10.0)
    rng = np.random.default_rng(0)
    for i in range(12):
        g.add_video_velocity(0.0 if i < 9 else float(rng.normal()) + i, i * 0.1)
        g.add_joint_speed(float(rng.normal()), i * 0.1)
    assert "lag_s" in g.estimate()


def test_sync_skips_flat_segments():
    s = Sync(window=24, max_offset_s=0.5)
    rng = np.random.default_rng(1)
    for i in range(24):
        s.add_a(0.0 if i < 16 else float(rng.normal()) + 3.0, i * 0.01)
        s.add_b(float(rng.normal()), i * 0.01)
    assert "offset_s" in s.estimate()


def test_registration_without_refinement_and_flat_scenes():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, (32, 32)).astype(np.float32)
    mv = np.roll(ref, (2, 3), axis=(0, 1))
    r = Registration(subpixel=False)
    r.set_reference(ref)
    out = r.transform_one(mv)
    assert (round(out["dx"]), round(out["dy"])) == (3, 2)
    g = GlobalMotion(refine=False)
    g.learn_one(ref)
    assert round(g.transform_one(mv)["dx"]) == 3
    z = GlobalMotion()
    z.learn_one(np.zeros((16, 16), dtype=np.float32))
    assert z.transform_one(np.zeros((16, 16), dtype=np.float32))["mag"] == 0.0
    d = DifferenceImage()
    assert d.transform_one(ref)["max"] == 0.0
    d.learn_one(ref)
    assert d.transform_one(mv)["diff"].shape == ref.shape


def test_motion_helpers_accept_dicts_and_zero_radius():
    img = np.random.default_rng(0).random((6, 6))
    assert motion._box_blur(img, 0) is img
    assert MotionEnergy().transform_one({"array": img.astype(np.float32)}) is not None


def test_dense_flow_falls_back_to_numpy_when_jax_fails(monkeypatch):
    def boom(self, g1, g2):
        raise RuntimeError("no jax")

    monkeypatch.setattr(DenseFlow, "_jax_solve", boom)
    fr = _frames(2)
    f = DenseFlow(n_iter=3)
    f.learn_one(fr[0])
    assert f.transform_one(fr[1])["u"].shape == fr[0].array.shape


def test_normalizer_log_scale_flat_image_and_plain_arrays():
    flat = np.full((8, 8), 0.3, dtype=np.float32)
    n = FrameNormalizer(log_scale=True)
    assert n.transform_one(flat, t=1.0) is not None
    n.learn_one(flat, t=1.0)
    ramp = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    assert FrameNormalizer(log_scale=True).transform_one(ramp) is not None


def test_quality_on_a_tiny_image_and_window_on_one_pixel_frames():
    assert FrameQuality().transform_one(np.zeros((2, 2), dtype=np.float32)) is not None
    w = FrameWindow(n=3)
    for i in range(3):
        w.learn_one(Frame(array=np.full((1, 1), 0.1 * i, dtype=np.float32), t=float(i)))
    assert w.transform_one(None)["n_frames"] == 3


def test_frame_anomaly_resets_its_parts_and_survives_a_failing_memory():
    from dense_armor.utility.vision.patches import PatchFeatures, PatchMemory

    class P(PatchFeatures):
        def _reset(self):
            self.was_reset = True

    class M(PatchMemory):
        def _reset(self):
            self.was_reset = True

        def score_one(self, d):
            raise RuntimeError("boom")

        def learn_one(self, d):
            raise RuntimeError("boom")

    fa = FrameAnomaly(patch_features=P(), memory=M())
    fr = _frames(2)[0]
    assert fa.transform_one(fr)["anomaly"] == 0.0
    fa.learn_one(fr)
    fa._reset()
    assert fa.memory.was_reset and fa.patch_features.was_reset


def test_tracker_on_color_and_static_scenes_and_competing_boxes():
    rng = np.random.default_rng(0)
    tr = MovingObjectTracker(history=2, min_area=4)
    still = rng.normal(0.4, 0.01, (20, 20, 3)).astype(np.float32)
    for _ in range(4):
        tr.transform_one(Frame(array=still, t=0.0))
    two = MovingObjectTracker(history=2, min_area=4)
    two.transform_one(Frame(array=still[..., :2], t=0.0))
    t = MovingObjectTracker(max_missing=3)
    t.update([(2, 2, 12, 12, 0.9)])
    t.update([(2, 2, 12, 12, 0.2)])
    assert t.update([(2, 2, 12, 12, 0.9), (3, 3, 12, 12, 0.9)]) is not None


def test_landmark_angle_with_coincident_points_and_feature_adapter_learn():
    k = Kinematics(triples=[(0, 1, 2)])
    assert k.transform_one(np.zeros((3, 2)), t=0.0) is not None
    assert FeatureToSignal().learn_one({"a": 1.0}) is not None


def test_overlay_accepts_a_single_channel_array():
    assert Overlay().transform_one(np.zeros((8, 8, 1), dtype=np.float32)) is not None
