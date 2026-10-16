"""V2-V7: motion, scene, window, tracking, adapters, fusion.

Every test generates its own synthetic frames from NumPy, no real video
in the repository. The heavy path (the file reader) is in
``test_video_file_stream.py`` and skips without OpenCV; the tests here
run everywhere.
"""

import numpy as np
import pytest

from dense_armor.utility.vision.adapters import (
    FeatureToSignal,
    TrackToSignal,
)
from dense_armor.utility.vision.fusion import FrameJointFusion
from dense_armor.utility.vision.motion import GlobalMotion, MotionEnergy
from dense_armor.utility.vision.scene import FrameAnomaly, SceneChange
from dense_armor.utility.vision.streams import ArrayStream
from dense_armor.utility.vision.tracking import MovingObjectTracker
from dense_armor.utility.vision.window import FrameWindow


def test_motion_energy_zero_on_first_frame():
    frames = np.zeros((3, 8, 8), dtype=np.float32)
    frames[1, 2:6, 2:6] = 1.0
    me = MotionEnergy(blur=0, downsample=1)
    out = []
    for fr in ArrayStream(frames, fps=10.0):
        out.append(me.transform_one(fr))
        _ = me.learn_one(fr)
    assert out[0]["motion_sum"] == 0.0
    assert out[1]["motion_sum"] > 0.0
    assert out[2]["motion_sum"] > 0.0


def test_global_motion_exact_on_pure_translation():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, (64, 64)).astype(np.float32)
    for seed in range(20):
        rng = np.random.default_rng(seed)
        base = rng.normal(0, 1, (64, 64)).astype(np.float32)
        dx = float(rng.integers(-5, 6))
        dy = float(rng.integers(-5, 6))
        shifted = np.roll(base, (int(dy), int(dx)), axis=(0, 1))
        gm = GlobalMotion()
        out = []
        for fr in ArrayStream(np.stack([base, shifted]), fps=10.0, gray=False):
            out.append(gm.transform_one(fr))
            _ = gm.learn_one(fr)
        assert abs(out[1]["dx"] - dx) < 0.2
        assert abs(out[1]["dy"] - dy) < 0.2


def test_scene_change_distance_peaks_at_cut():
    rng = np.random.default_rng(0)
    frames = [rng.normal(0.5, 0.05, (16, 16)).astype(np.float32)
              for _ in range(30)]
    frames += [np.full((16, 16), 0.95, dtype=np.float32) for _ in range(30)]
    sc = SceneChange()
    dists = []
    for fr in ArrayStream(np.stack(frames), fps=10.0):
        out = sc.transform_one(fr)
        _ = sc.learn_one(fr)
        dists.append(out["distance"])
    assert dists[30] > 0.5
    assert max(dists[31:]) < 0.05


def test_frame_window_matches_batch_numpy():
    rng = np.random.default_rng(0)
    frames = rng.uniform(0, 1, (30, 16, 16)).astype(np.float32)
    w = FrameWindow(n=10, quantiles=())
    for fr in ArrayStream(frames, fps=10.0):
        _ = w.learn_one(fr)
    out = w.transform_one(next(iter(ArrayStream(frames[-1:], fps=10.0))))
    batch = frames[-10:]
    assert abs(out["mean"] - batch.mean()) < 1e-9
    assert abs(out["std"] - batch.std()) < 1e-9


def test_frame_window_memory_does_not_grow():
    w = FrameWindow(n=10)
    for i in range(1000):
        f = np.zeros((4, 4), dtype=np.float32)
        _ = w.learn_one({"array": f, "t": i / 30.0})
    assert len(w._arrays) == 10


def test_tracker_keeps_one_id_on_a_moving_disk():
    rng = np.random.default_rng(0)
    frames = []
    pos = np.array([10.0, 10.0])
    vel = np.array([1.0, 0.5])
    for _ in range(30):
        f = rng.normal(0.1, 0.005, (64, 64)).astype(np.float32)
        y, x = int(round(pos[0])), int(round(pos[1]))
        f[y:y+8, x:x+8] = 1.0
        frames.append(f)
        pos = pos + vel
    t = MovingObjectTracker(history=10, min_area=20, max_missing=5,
                            iou_threshold=0.2, n_sigmas=5.0)
    ids = []
    for fr in ArrayStream(np.stack(frames), fps=10.0):
        out = t.transform_one(fr)
        _ = t.learn_one(fr)
        if out["tracks"]:
            ids.append(out["tracks"][0]["id"])
    assert ids, "no track was formed"
    assert len(set(ids)) == 1


def test_track_to_signal_shape():
    a = TrackToSignal(track_id=7, units={"cx": "px"})
    _ = a.learn_one({"tracks": [{"id": 7, "box": (10, 20, 14, 26), "missing": 0}]})
    s = a.transform_one(None, t=0.5)
    assert s["cx"] == 12.0
    assert s["cy"] == 23.0
    assert s.units[0] == "px"
    assert abs(s.t - 0.5) < 1e-9


def test_feature_to_signal_shape():
    f = FeatureToSignal(units={"mean": "1"})
    s = f.transform_one({"mean": 0.5, "std": 0.1})
    assert abs(s["mean"] - 0.5) < 1e-6
    assert abs(s["std"] - 0.1) < 1e-5
    assert s.units[0] == "1"


def test_fusion_estimates_known_latency():
    rng = np.random.default_rng(0)
    latency = 0.03
    fps = 60.0
    n = 200
    dt = 1.0 / fps
    vel = rng.normal(0, 1, n)
    shift = int(round(latency * fps))
    vel_lag = np.zeros(n)
    for i in range(n):
        j = i - shift
        vel_lag[i] = vel[j] if j >= 0 else 0.0
    fus = FrameJointFusion(max_lag_s=0.1, window_s=5.0, frame_rate_hz=fps)
    t = np.arange(n) * dt
    for i in range(n):
        fus.add_video_velocity(float(vel_lag[i]), t[i])
        fus.add_joint_speed(float(vel[i]), t[i])
    out = fus.estimate()
    assert abs(out["lag_s"] - latency) < 0.005


def test_frame_anomaly_higher_after_change():
    rng = np.random.default_rng(0)
    normal = [rng.normal(0.5, 0.02, (16, 16)).astype(np.float32)
              for _ in range(40)]
    fa = FrameAnomaly()
    for fr in ArrayStream(np.stack(normal), fps=10.0):
        _ = fa.learn_one(fr)
    normal_scores = [fa.transform_one(fr)["anomaly"]
                     for fr in ArrayStream(np.stack(normal), fps=10.0)]
    anomalous = []
    for _ in range(10):
        f = np.zeros((16, 16), dtype=np.float32)
        f[::2, ::2] = 1.0
        f[1::2, 1::2] = 1.0
        anomalous.append(f)
    anomalous_scores = [fa.transform_one(fr)["anomaly"]
                        for fr in ArrayStream(np.stack(anomalous), fps=10.0)]
    assert float(np.mean(normal_scores)) < float(np.mean(anomalous_scores))
