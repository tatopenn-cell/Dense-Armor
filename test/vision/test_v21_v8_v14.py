"""Tests for V8-V14: robustness, landmarks, registration, dense, quality, viz."""

import numpy as np

from dense_armor.utility.vision.dense import DenseFlow, MotionMask
from dense_armor.utility.vision.landmarks import (
    Kinematics,
    LandmarkCleaner,
    LandmarkStream,
)
from dense_armor.utility.vision.quality import EventFrame, FrameQuality, Sync
from dense_armor.utility.vision.registration import (
    DifferenceImage,
    Registration,
    Stabilizer,
)
from dense_armor.utility.vision.robustness import FrameNormalizer
from dense_armor.utility.vision.visualize import FalseColor, Overlay


def _shift_subpixel(img, dy, dx):
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    y2, x2 = yy - dy, xx - dx
    y0 = np.floor(y2).astype(int); x0 = np.floor(x2).astype(int)
    wy = y2 - y0; wx = x2 - x0
    def g(y, x):
        return img[np.clip(y, 0, h-1), np.clip(x, 0, w-1)]
    return (g(y0,x0)*(1-wy)*(1-wx) + g(y0+1,x0)*wy*(1-wx)
            + g(y0,x0+1)*(1-wy)*wx + g(y0+1,x0+1)*wy*wx).astype(np.float32)


def test_normalizer_black_and_time_back():
    n = FrameNormalizer()
    f = np.zeros((8, 8), dtype=np.float32)
    _ = n.transform_one({"array": f, "t": 1.0})
    out = n.transform_one({"array": f + 0.5, "t": 0.5})
    assert out["time_back"]
    black = np.zeros((8, 8), dtype=np.float32)
    assert n.transform_one({"array": black, "t": 2.0})["black"]


def test_registration_subpixel_under_0_2():
    rng = np.random.default_rng(0)
    for seed in range(10):
        rng = np.random.default_rng(seed)
        base = rng.normal(0, 1, (128, 128)).astype(np.float32)
        dx = float(rng.uniform(-2, 2)); dy = float(rng.uniform(-2, 2))
        b = _shift_subpixel(base, dy, dx)
        r = Registration()
        r.set_reference({"array": base})
        out = r.transform_one({"array": b})
        assert abs(out["dx"] - dx) < 0.2
        assert abs(out["dy"] - dy) < 0.2


def test_stabilizer_and_difference():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, (32, 32)).astype(np.float32)
    moved = np.roll(base, (2, -3), axis=(0, 1))
    s = Stabilizer()
    _ = s.learn_one({"array": base})
    out = s.transform_one({"array": moved})
    assert out["frame"].shape == (32, 32)
    d = DifferenceImage(alpha=0.5)
    _ = d.learn_one({"array": base})
    out = d.transform_one({"array": base})
    assert float(out["diff"].max()) < 1e-6


def test_dense_flow_and_mask():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (32, 32)).astype(np.float32)
    b = np.roll(a, 1, axis=1)
    f = DenseFlow(alpha=0.5, n_iter=10)
    _ = f.learn_one({"array": a})
    out = f.transform_one({"array": b})
    assert out["u"].shape == (32, 32)
    m = MotionMask(n_sigmas=3.0, dilation=0)
    out2 = m.transform_one({"u": out["u"], "v": out["v"]})
    assert "mask" in out2
    assert "iou_dummy" not in out2


def test_quality_and_sync():
    q = FrameQuality()
    f = np.zeros((8, 8), dtype=np.float32)
    out = q.transform_one({"array": f})
    assert out["black"]
    s = Sync(window=200, max_offset_s=0.2)
    rng = np.random.default_rng(0)
    sig = rng.normal(0, 1, 200)
    for i in range(200):
        s.add_a(float(sig[i]), i * 0.01)
        s.add_b(float(sig[i]), i * 0.01 + 0.05)
    out = s.estimate()
    assert abs(out["offset_s"] - 0.05) < 0.02


def test_event_frame_accumulates():
    e = EventFrame(shape=(8, 8), decay=1.0)
    for _ in range(3):
        _ = e.learn_one({"x": 4, "y": 2, "polarity": 1, "t": 0.0})
    out = e.transform_one(None)
    assert int(out["frame"][2, 4]) == 3


def test_falsecolor_and_overlay():
    f = FalseColor(colormap="hot")
    a = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    out = f.transform_one({"array": a})
    assert out["rgb"].shape == (8, 8, 3)
    o = Overlay()
    fr = np.zeros((32, 32, 3), dtype=np.float32)
    tracks = [{"id": 1, "box": (4, 4, 12, 16), "missing": 0}]
    out = o.transform_one({"array": fr, "tracks": tracks})
    assert out["rgb"][4, 4].tolist() == [0.0, 1.0, 0.0]


def test_landmark_cleaner_and_kinematics():
    c = LandmarkCleaner(window=7, n_sigmas=3.0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        v = float(rng.normal(0, 0.01))
        _ = c.learn_one({"points": np.array([[v, 0.0]])})
    out = c.transform_one({"points": np.array([[100.0, 0.0]])})
    assert int(out["cleaned"][0]) == 1
    assert abs(out["points"][0, 0]) < 0.1
    k = Kinematics(triples=[(0, 1, 2)], pairs=[(0, 2)])
    pts = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    out = k.transform_one({"points": pts, "t": 0.0})
    assert abs(out["angle_0_1_2"] - 1.5708) < 0.01
    assert abs(out["sym_0_2"] - 1.41421) < 0.001


def test_landmark_stream_iterates():
    pts = np.zeros((3, 4, 2), dtype=np.float32)
    s = LandmarkStream(pts, fps=10.0)
    assert len(list(s)) == 3
