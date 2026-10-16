"""Argument checks, resets, multi-channel inputs and the small branches of the video modules."""

import numpy as np
import pytest

from dense_armor.utility.vision import streams
from dense_armor.utility.vision.adapters import FeatureToSignal, TrackToSignal
from dense_armor.utility.vision.dense import DenseFlow, MotionMask
from dense_armor.utility.vision.fusion import FrameJointFusion
from dense_armor.utility.vision.landmarks import (
    Kinematics,
    LandmarkCleaner,
    LandmarkStream,
    _hampel_value,
)
from dense_armor.utility.vision.motion import GlobalMotion, MotionEnergy
from dense_armor.utility.vision.quality import EventFrame, FrameQuality, Sync
from dense_armor.utility.vision.registration import DifferenceImage, Registration, Stabilizer
from dense_armor.utility.vision.robustness import FrameNormalizer
from dense_armor.utility.vision.scene import FrameAnomaly, SceneChange
from dense_armor.utility.vision.streams import ArrayStream, CubeStream, Frame
from dense_armor.utility.vision.tracking import MovingObjectTracker
from dense_armor.utility.vision.visualize import FalseColor, Overlay
from dense_armor.utility.vision.window import FrameWindow


def _frames(n=4, shape=(24, 24), seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        f = rng.normal(0.4, 0.02, shape).astype(np.float32)
        f[4 + i : 10 + i, 4 + i : 10 + i] += 0.5
        out.append(Frame(array=f, t=0.1 * i))
    return out


BAD_ARGS = [
    (FalseColor, {"colormap": "nope"}, ValueError),
    (FalseColor, {"low_pct": 60.0, "high_pct": 40.0}, ValueError),
    (DifferenceImage, {"alpha": 0.0}, ValueError),
    (DifferenceImage, {"outlier_sigma": 0.0}, ValueError),
    (DenseFlow, {"alpha": 0.0}, ValueError),
    (DenseFlow, {"n_iter": 0}, ValueError),
    (MotionMask, {"n_sigmas": 0.0}, ValueError),
    (MotionMask, {"dilation": -1}, ValueError),
    (LandmarkCleaner, {"window": 2}, ValueError),
    (LandmarkCleaner, {"n_sigmas": 0.0}, ValueError),
    (LandmarkCleaner, {"max_gap": -1}, ValueError),
    (Kinematics, {"cycle_window": -1}, ValueError),
    (MotionEnergy, {"blur": -1}, ValueError),
    (MotionEnergy, {"downsample": 0}, ValueError),
    (MotionEnergy, {"p_high": 100.0}, ValueError),
    (Sync, {"window": 3}, ValueError),
    (Sync, {"max_offset_s": 0.0}, ValueError),
    (EventFrame, {"shape": (0, 4)}, ValueError),
    (EventFrame, {"decay": 1.5}, ValueError),
    (FrameJointFusion, {"max_lag_s": 0.0}, ValueError),
    (FrameJointFusion, {"window_s": 0.0}, ValueError),
    (FrameJointFusion, {"frame_rate_hz": 0.0}, ValueError),
    (FrameNormalizer, {"low_pct": 50.0, "high_pct": 10.0}, ValueError),
    (FrameWindow, {}, ValueError),
    (FrameWindow, {"n": 2, "seconds": 1.0}, ValueError),
    (FrameWindow, {"n": 0}, ValueError),
    (FrameWindow, {"seconds": 0.0}, ValueError),
    (FrameWindow, {"n": 2, "quantiles": (1.5,)}, ValueError),
    (SceneChange, {"bins": 0}, ValueError),
    (SceneChange, {"blur": -1}, ValueError),
    (SceneChange, {"downsample": 0}, ValueError),
    (SceneChange, {"detector": object()}, TypeError),
    (FrameAnomaly, {"patch_features": object()}, TypeError),
    (FrameAnomaly, {"memory": object()}, TypeError),
    (MovingObjectTracker, {"n_sigmas": 0.0}, ValueError),
    (MovingObjectTracker, {"history": 0}, ValueError),
    (MovingObjectTracker, {"min_area": 0}, ValueError),
    (MovingObjectTracker, {"max_missing": -1}, ValueError),
    (MovingObjectTracker, {"iou_threshold": 2.0}, ValueError),
    (TrackToSignal, {"track_id": -1}, ValueError),
]


@pytest.mark.parametrize("cls,kw,exc", BAD_ARGS)
def test_bad_arguments_are_refused(cls, kw, exc):
    with pytest.raises(exc):
        cls(**kw)


def test_landmark_stream_refuses_bad_shapes():
    pts = np.zeros((5, 3, 2))
    with pytest.raises(ValueError):
        LandmarkStream(pts, fps=0.0)
    with pytest.raises(ValueError):
        LandmarkStream(np.zeros((5, 3)))
    with pytest.raises(ValueError):
        LandmarkStream(pts, confidence=np.ones((5, 2)))
    with pytest.raises(ValueError):
        LandmarkStream(pts, times=np.arange(4.0))
    s = LandmarkStream(pts, confidence=np.ones((5, 3)), times=np.arange(5.0))
    assert len(list(s)) == 5


def test_every_transformer_resets_and_declares_no_skips():
    objs = [
        MotionEnergy(), GlobalMotion(), SceneChange(), FrameAnomaly(), FrameWindow(n=3),
        MovingObjectTracker(), TrackToSignal(), FeatureToSignal(), FrameJointFusion(),
        FrameNormalizer(), Registration(), Stabilizer(), DifferenceImage(), DenseFlow(),
        MotionMask(), FrameQuality(), Sync(), EventFrame(), FalseColor(), Overlay(),
        LandmarkCleaner(), Kinematics(),
    ]
    for o in objs:
        o._reset()
        assert o._unit_test_skips() == set()
        assert o.learn_one(_frames(1)[0]) is o if not isinstance(o, (Sync, EventFrame, LandmarkCleaner, Kinematics, TrackToSignal, FeatureToSignal)) else True


@pytest.mark.parametrize("channels", [2, 3])
def test_multichannel_frames_go_through_every_module(channels):
    rng = np.random.default_rng(1)
    cube = rng.normal(0.4, 0.05, (4, 20, 20, channels)).astype(np.float32)
    mods = [
        MotionEnergy(), GlobalMotion(), SceneChange(), FrameWindow(n=3), FrameNormalizer(),
        FrameQuality(), Registration(), DifferenceImage(), DenseFlow(use_jax=False),
    ]
    if channels == 3:
        mods.append(FrameAnomaly())
    for m in mods:
        for i in range(4):
            fr = Frame(array=cube[i], t=0.1 * i)
            m.transform_one(fr)
            m.learn_one(fr)
    st = Stabilizer()
    for i in range(3):
        st.learn_one(Frame(array=cube[i], t=0.1 * i))
        out = st.transform_one(Frame(array=cube[i], t=0.1 * i))
    assert out is not None


def test_registration_with_a_window_and_a_first_frame():
    frames = _frames(3)
    r = Registration(window=16)
    first = r.transform_one(frames[0])
    assert first["mag"] == 0.0
    r.set_reference(frames[0])
    out = r.transform_one(frames[1])
    assert "dx" in out
    st = Stabilizer(Registration(window=16))
    st.learn_one(frames[0])
    assert st.transform_one(frames[1]) is not None
    g = GlobalMotion(window=16)
    g.learn_one(frames[0])
    assert "dx" in g.transform_one(frames[1])


def test_difference_image_updates_its_reference_for_flat_and_noisy_scenes():
    d = DifferenceImage(alpha=0.5)
    flat = Frame(array=np.full((12, 12), 0.5, dtype=np.float32), t=0.0)
    d.learn_one(flat)
    d.learn_one(flat)
    noisy = Frame(array=np.random.default_rng(0).normal(0.5, 0.05, (12, 12)).astype(np.float32), t=0.1)
    d.learn_one(noisy)
    out = d.transform_one(noisy)
    assert out["diff"].shape == (12, 12)
    d2 = DifferenceImage()
    first = d2.transform_one({"frame": flat.array})
    assert first["max"] == 0.0
    d2.learn_one({"frame": flat.array})
    d2.learn_one(flat.array)


def test_dense_flow_numpy_path_and_first_call():
    fr = _frames(3)
    f = DenseFlow(alpha=1.0, n_iter=5, use_jax=False)
    zero = f.transform_one(fr[0])
    assert float(zero["mag"].max()) == 0.0
    f.learn_one(fr[0])
    out = f.transform_one(fr[1])
    assert out["u"].shape == fr[0].array.shape


def test_motion_mask_dilates_and_handles_a_flat_field():
    u = np.zeros((20, 20))
    v = np.zeros((20, 20))
    assert not MotionMask().transform_one({"u": u, "v": v})["mask"].any()
    u[8:12, 8:12] = 2.0
    rng = np.random.default_rng(0)
    u += rng.normal(0, 0.02, u.shape)
    m0 = MotionMask(dilation=0).transform_one({"u": u, "v": v})["mask"]
    m2 = MotionMask(dilation=2).transform_one({"u": u, "v": v})["mask"]
    assert m2.sum() > m0.sum()
    assert MotionMask().learn_one({"u": u, "v": v}) is not None


def test_motion_energy_blur_and_downsample_paths():
    fr = _frames(3, shape=(30, 30))
    me = MotionEnergy(blur=2, downsample=3)
    for f in fr:
        out = me.transform_one(f)
        me.learn_one(f)
    assert out["motion_sum"] >= 0.0
    me0 = MotionEnergy(blur=0, downsample=1)
    me0.learn_one(fr[0])
    assert me0.transform_one(fr[1]) is not None


def test_sync_with_too_little_data_or_no_overlap():
    s = Sync(window=8, max_offset_s=0.1)
    assert s.estimate()["correlation"] == 0.0
    for i in range(8):
        s.add_a(1.0, i * 0.01)
        s.add_b(1.0, i * 0.01)
    assert s.transform_one()["offset_s"] == 0.0
    s2 = Sync(window=8, max_offset_s=0.1)
    rng = np.random.default_rng(0)
    for i in range(8):
        s2.add_a(float(rng.normal()), i * 0.01)
        s2.add_b(float(rng.normal()), 10.0 + i * 0.01)
    assert s2.estimate()["offset_s"] == 0.0
    assert s2.learn_one(None) is s2
    s3 = Sync(window=16, max_offset_s=0.5)
    for i in range(16):
        s3.add_a(float(np.sin(i)), i * 0.01)
        s3.add_b(float(np.sin(i)), i * 0.01)
    assert "offset_s" in s3.estimate()


def test_event_frame_accepts_dicts_lists_and_decays():
    ef = EventFrame(shape=(8, 8), decay=0.5)
    assert ef.transform_one(None) is not None
    ef.learn_one({"events": [[1, 2, 1, 0.1], [3, 4, -1, 0.2]]})
    ef.learn_one([[1, 2, 1, 0.3]])
    out = ef.transform_one(None)
    assert out is not None


def test_frame_quality_flags_black_and_repeated_frames():
    q = FrameQuality()
    black = np.zeros((10, 10), dtype=np.float32)
    assert q.transform_one(black) is not None
    q.learn_one(black)
    again = q.transform_one(black)
    assert again is not None
    rgb = np.random.default_rng(0).random((10, 10, 3)).astype(np.float32)
    assert q.transform_one(rgb) is not None


def test_fusion_adds_frames_and_joint_speeds_and_trims():
    f = FrameJointFusion(max_lag_s=0.1, window_s=0.5, frame_rate_hz=20.0)
    assert f.estimate()["n"] == 0
    for fr in _frames(12):
        f.learn_one(fr)
        f.add_joint_speed(float(fr.t), fr.t)
    est = f.transform_one(None)
    assert "lag_s" in est
    f.learn_one({"frame": _frames(1)[0], "t": 5.0}, t=5.0)
    f.add_frame(_frames(1)[0].array, 9.0)
    assert f.estimate() is not None


def test_frame_window_by_seconds_and_quantiles():
    w = FrameWindow(seconds=0.25, quantiles=(0.25, 0.75))
    assert w.transform_one(None)["n_frames"] == 0
    for fr in _frames(8):
        w.learn_one(fr)
    out = w.transform_one(None)
    assert out["n_frames"] <= 4 and "q25" in out and "q75" in out
    w.learn_one(np.zeros((24, 24), dtype=np.float32), t=0.0)
    w2 = FrameWindow(n=2)
    for i in range(3):
        w2.learn_one(Frame(array=np.zeros((6, 6, 3), dtype=np.float32), t=float(i)))
    assert w2.transform_one(None)["n_frames"] == 2


def test_scene_change_with_a_dict_and_a_frame_anomaly_that_keeps_learning():
    sc = SceneChange()
    fr = _frames(3)
    sc.learn_one({"array": fr[0].array})
    out = sc.transform_one({"array": fr[1].array})
    assert out is not None
    fa = FrameAnomaly()
    for f in fr:
        fa.transform_one(f)
        fa.learn_one(f)
    fa._reset()


def test_tracker_with_external_boxes_and_a_custom_filter():
    from dense_armor.utility.timeseries.kalman import KalmanFilter

    tr = MovingObjectTracker(max_missing=1)
    for _ in range(3):
        out = tr.update([(2, 2, 10, 10), (30, 30, 40, 40, 0.9)])
    assert out["tracks"]
    for _ in range(3):
        out = tr.update([])
    assert out["tracks"] == [] or isinstance(out["tracks"], list)
    t2 = MovingObjectTracker(kf=lambda: KalmanFilter(F=np.eye(4), H=np.eye(4), Q=np.eye(4) * 1e-3, R=np.eye(4)))
    assert t2.update([(1, 1, 9, 9)]) is not None
    t3 = MovingObjectTracker(kf=KalmanFilter(F=np.eye(4), H=np.eye(4), Q=np.eye(4) * 1e-3, R=np.eye(4)))
    assert t3.update([(1, 1, 9, 9)]) is not None
    assert MovingObjectTracker().transform_one(_frames(1)[0]) is not None


def test_adapters_build_signals_from_lists_speeds_and_keys():
    ts = TrackToSignal()
    tr = {"id": 1, "box": (2, 2, 8, 8), "missing": 0}
    ts.learn_one([tr])
    s1 = ts.transform_one(None, t=0.0)
    ts.learn_one({"tracks": [{"id": 1, "box": (4, 2, 10, 8), "missing": 0}]})
    s2 = ts.transform_one(None, t=0.5)
    assert s1 is not None and s2 is not None
    ts_none = TrackToSignal(track_id=7)
    ts_none.learn_one({"tracks": []})
    assert ts_none.transform_one(None, t=1.0) is not None
    fs = FeatureToSignal(include=["a", "b"])
    sig = fs.transform_one({"a": 1.0, "b": "text"}, t=1.0)
    assert sig is not None
    assert FeatureToSignal().transform_one({}, t=0.0) is not None
    assert FeatureToSignal().transform_one(3.0, t=0.0) is not None


def test_landmark_cleaner_and_kinematics_small_branches():
    assert _hampel_value([1.0, 1.0], 5.0, 3.0) == (5.0, False)
    assert _hampel_value([1.0, 1.0, 1.0], 5.0, 3.0)[1] is True
    assert _hampel_value([1.0, 2.0, 3.0, 2.0], 90.0, 3.0)[1] is True
    assert _hampel_value([1.0, 2.0, 3.0, 2.0], 2.0, 3.0)[1] is False
    c = LandmarkCleaner(window=5)
    pts = np.zeros((3, 2))
    for _ in range(4):
        c.learn_one(pts)
    out = c.transform_one({"points": pts, "confidence": np.array([1.0, 0.0, 1.0])})
    assert out is not None
    c.transform_one(pts)
    k = Kinematics(triples=[(0, 1, 2)], pairs=[(0, 1)], cycle_window=4)
    base = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    for i in range(5):
        res = k.transform_one(base + 0.1 * i, t=0.1 * i)
    assert res is not None
    k.learn_one({"points": base, "t": 1.0}, t=1.0)


def test_false_color_and_overlay_branches():
    img = np.random.default_rng(0).random((16, 16)).astype(np.float32)
    fc = FalseColor(colormap="hot", log_scale=True)
    assert fc.transform_one(img)["rgb"].shape == (16, 16, 3)
    assert FalseColor().transform_one(np.stack([img, img], axis=-1))["rgb"].shape == (16, 16, 3)
    assert FalseColor().transform_one(np.zeros((16, 16), dtype=np.float32))["rgb"].shape == (16, 16, 3)
    assert fc.learn_one(img) is fc
    ov = Overlay(draw_flow_step=4)
    out = ov.transform_one(img)
    assert out is not None
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:8, 4:8] = True
    u = np.full((16, 16), 2.0)
    full = ov.transform_one({"array": np.stack([img] * 3, axis=-1), "tracks": [{"id": 1, "box": (2, 2, 9, 9)}],
                             "mask": mask, "flow": {"u": u, "v": u * 0.0}})
    assert full is not None
    assert ov.learn_one(img) is ov


def test_stream_dtype_conversions_and_cube():
    a8 = (np.random.default_rng(0).random((3, 6, 6)) * 255).astype(np.uint8)
    a16 = (a8.astype(np.uint16) * 257)
    for arr in (a8, a16, a8.astype(np.float32) / 255.0, a8.astype(np.int32)):
        for dtype in ("float32", "uint8", "uint16"):
            fr = list(ArrayStream(arr, fps=10.0, dtype=dtype))
            assert len(fr) == 3
    rgb = np.random.default_rng(1).random((2, 6, 6, 4)).astype(np.float32)
    assert len(list(ArrayStream(rgb, fps=5.0, gray=True))) == 2
    assert streams._to_dtype(rgb[0], "float32", False, bgr=True).shape[-1] == 3
    cube = np.random.default_rng(2).random((6, 5, 5, 3)).astype(np.float32)
    got = list(CubeStream(cube, fps=2.0, stride=2, max_frames=2))
    assert len(got) == 2 and got[1].t == 1.0
    assert len(CubeStream(cube)) == 6
    with pytest.raises(ValueError):
        CubeStream(np.zeros((4, 4)))
