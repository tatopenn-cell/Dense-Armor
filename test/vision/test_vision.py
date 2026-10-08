"""Tests for the native online vision package."""

import numpy as np
import pytest

from dense_armor.checks import check_estimator
from dense_armor.utility.anomaly.mahalanobis import OnlineRobustMahalanobis
from dense_armor.utility.vision.features import FrameFeatures
from dense_armor.utility.vision.reduce import (
    IncrementalPCA,
    RandomProjection,
)
from dense_armor.utility.vision.streams import (
    ArrayStream,
    ImageFolderStream,
)


def _square_moving(n=80, size=32):
    frames = []
    for i in range(n):
        f = np.zeros((size, size), dtype=np.float32)
        x = min(size - 8, 4 + i // 4)
        f[8:16, x : x + 8] = 1.0
        frames.append(f)
    return np.stack(frames)


def test_array_stream_shapes_and_times():
    frames = _square_moving(n=5)
    s = ArrayStream(frames, fps=10.0)
    assert len(s) == 5
    fs = list(s)
    assert fs[0].array.shape == (32, 32)
    assert fs[0].t == 0.0
    assert abs(fs[4].t - 0.4) < 1e-9


def test_array_stream_rejects_bad_fps():
    with pytest.raises(ValueError):
        ArrayStream(_square_moving(2), fps=0.0)


def test_array_stream_resizes():
    frames = _square_moving(n=2, size=32)
    s = ArrayStream(frames, fps=10.0, size=(16, 16))
    f = next(iter(s))
    assert f.array.shape == (16, 16)


def test_image_folder_stream_roundtrip(tmp_path):
    from PIL import Image

    frames = _square_moving(n=3)
    for i, f in enumerate(frames):
        Image.fromarray((f * 255).astype(np.uint8)).save(tmp_path / f"{i:03d}.png")
    s = ImageFolderStream(tmp_path, fps=5.0)
    assert len(s) == 3
    fs = list(s)
    assert abs(fs[0].t - 0.0) < 1e-9
    assert abs(fs[2].t - 0.4) < 1e-9


def test_image_folder_stream_empty(tmp_path):
    with pytest.raises(ValueError):
        ImageFolderStream(tmp_path)


def test_features_constant_frame_zero_contrast():
    ff = FrameFeatures()
    out = ff.transform_one(np.zeros((16, 16), dtype=np.float32))
    assert out["mean"] == 0.0
    assert out["contrast"] == 0.0
    assert out["hog_0_0_0"] == 0.0


def test_features_horizontal_edge_lands_in_one_bin():
    img = np.zeros((16, 16), dtype=np.float32)
    img[8:, :] = 1.0
    ff = FrameFeatures(n_bins=4, cells=(1, 1))
    out = ff.transform_one(img)
    bins = [out[f"hog_0_0_{j}"] for j in range(4)]
    top = int(np.argmax(bins))
    assert bins[top] > 0.9
    assert sum(bins) - bins[top] < 0.1


def test_features_flow_one_pixel_shift():
    a = np.zeros((32, 32), dtype=np.float32)
    a[12:20, 12:20] = 1.0
    b = np.roll(a, 1, axis=1)
    ff = FrameFeatures(flow_cells=(1, 1))
    ff.learn_one(a)
    out = ff.transform_one(b)
    assert abs(out["flow_u_0_0"] - 1.0) < 0.1
    assert abs(out["flow_v_0_0"]) < 0.1


def test_features_flow_one_pixel_vertical():
    a = np.zeros((32, 32), dtype=np.float32)
    a[12:20, 12:20] = 1.0
    b = np.roll(a, 1, axis=0)
    ff = FrameFeatures(flow_cells=(1, 1))
    ff.learn_one(a)
    out = ff.transform_one(b)
    assert abs(out["flow_v_0_0"] - 1.0) < 0.1
    assert abs(out["flow_u_0_0"]) < 0.1


def test_features_flow_pyramid_recovers_six_pixels():
    c = np.zeros((64, 64), dtype=np.float32)
    c[24:40, 24:40] = 1.0
    d = np.roll(c, 6, axis=1)
    flat = FrameFeatures(flow_cells=(1, 1), flow_levels=1)
    flat.learn_one(c)
    u_flat = flat.transform_one(d)["flow_u_0_0"]
    pyr = FrameFeatures(flow_cells=(1, 1), flow_levels=3)
    pyr.learn_one(c)
    u_pyr = pyr.transform_one(d)["flow_u_0_0"]
    assert abs(u_flat - 6.0) > 1.5
    assert abs(u_pyr - 6.0) < 0.5


def test_features_flow_levels_one_unchanged():
    a = np.zeros((32, 32), dtype=np.float32)
    a[12:20, 12:20] = 1.0
    b = np.roll(a, 1, axis=1)
    ff = FrameFeatures(flow_cells=(1, 1), flow_levels=1)
    ff.learn_one(a)
    out = ff.transform_one(b)
    assert abs(out["flow_u_0_0"] - 1.0) < 0.1


def test_features_color_moments():
    rgb = np.zeros((16, 16, 3), dtype=np.float32)
    rgb[..., 0] = 0.8
    rgb[..., 1] = 0.4
    rgb[..., 2] = 0.1
    ff = FrameFeatures(gray=False)
    out = ff.transform_one(rgb)
    assert abs(out["mean_R"] - 0.8) < 1e-6
    assert abs(out["mean_G"] - 0.4) < 1e-6
    assert abs(out["mean_B"] - 0.1) < 1e-6
    assert out["contrast_R"] < 1e-6
    assert out["contrast_G"] < 1e-6
    assert out["contrast_B"] < 1e-6


def test_features_gray_has_no_color_keys():
    a = np.zeros((16, 16), dtype=np.float32)
    out = FrameFeatures(gray=True).transform_one(a)
    for name in ("mean_R", "mean_G", "mean_B"):
        assert name not in out


def test_features_rejects_bad_flow_levels():
    with pytest.raises(ValueError):
        FrameFeatures(flow_levels=0)


def test_features_nan_frame_counted_missing():
    ff = FrameFeatures()
    bad = np.full((16, 16), np.nan, dtype=np.float32)
    _ = ff.transform_one(bad)
    assert ff.n_missing >= 1


def test_random_projection_preserves_distances():
    rng = np.random.default_rng(0)
    d, n, k = 256, 100, 64
    X = rng.standard_normal((n, d))
    rp = RandomProjection(k=k, seed=0)
    keys = [f"f{i:03d}" for i in range(d)]
    Y = np.array([list(rp.transform_one(dict(zip(keys, row))).values()) for row in X])
    ratios = []
    for i in range(n):
        for j in range(i + 1, n):
            num = float(np.sum((Y[i] - Y[j]) ** 2))
            den = float(np.sum((X[i] - X[j]) ** 2))
            if den > 0:
                ratios.append(num / den)
    ratios = np.array(ratios)
    median = float(np.median(ratios))
    frac_out = float(np.mean((ratios < 0.5) | (ratios > 1.5)))
    assert 0.9 < median < 1.1, f"median={median:.3f}"
    assert frac_out < 0.05, f"frac_out={frac_out:.4f}"


def test_incremental_pca_recovers_top_direction():
    rng = np.random.default_rng(0)
    pca = IncrementalPCA(k=1)
    for _ in range(400):
        a = rng.normal(0.0, 0.1)
        b = rng.normal(0.0, 1.0)
        _ = pca.learn_one({"a": float(a), "b": float(b)})
    v = pca.components_[:, 0]
    v = v / (np.linalg.norm(v) + 1e-12)
    angle = float(np.degrees(np.arccos(np.clip(abs(v[1]), 0.0, 1.0))))
    assert angle < 10.0


def test_incremental_pca_rejects_bad_k():
    with pytest.raises(ValueError):
        IncrementalPCA(k=0)


def test_pipeline_score_rises_after_change():
    frames = _square_moving(n=80)
    change = []
    for _ in range(20):
        f = np.zeros((32, 32), dtype=np.float32)
        f[8:20, 8:20] = 1.0
        f[24:30, 24:30] = 0.8
        change.append(f)
    all_frames = np.concatenate([frames, np.stack(change)])
    stream = ArrayStream(all_frames, fps=10.0)
    pipe = (
        FrameFeatures()
        | IncrementalPCA(k=4)
        | OnlineRobustMahalanobis(feature_keys=[f"pc{j}" for j in range(4)])
    )
    scores = []
    for fr in stream:
        s = pipe.steps[-1][1].score_one(
            pipe.steps[1][1].transform_one(pipe.steps[0][1].transform_one(fr))
        )
        scores.append(s)
        pipe.learn_one(fr, t=fr.t)
    before = float(np.mean(scores[20:80]))
    after = float(np.mean(scores[80:]))
    assert after > before


def test_pipeline_trains_intermediate_steps():
    frames = _square_moving(n=20)
    stream = ArrayStream(frames, fps=10.0)
    ff = FrameFeatures()
    pca = IncrementalPCA(k=2)
    pipe = ff | pca | OnlineRobustMahalanobis(feature_keys=[f"pc{j}" for j in range(2)])
    for fr in stream:
        pipe.learn_one(fr, t=fr.t)
    assert ff._prev is not None
    assert pca.components_ is not None
    assert ff.n_missing == 0


@pytest.mark.parametrize(
    "cls, kwargs",
    [
        (FrameFeatures, {}),
        (RandomProjection, {"k": 4}),
        (IncrementalPCA, {"k": 4}),
    ],
)
def test_check_estimator(cls, kwargs):
    check_estimator(cls(**kwargs))


def test_random_projection_matches_jl_bound():
    rng = np.random.default_rng(0)
    d, n, k, eps = 256, 100, 64, 0.5
    X = rng.standard_normal((n, d))
    rp = RandomProjection(k=k, seed=0)
    keys = [f"f{i:03d}" for i in range(d)]
    Y = np.array([
        [rp.transform_one({keys[i]: float(X[j, i]) for i in range(d)})[f"p{t}"]
         for t in range(k)]
        for j in range(n)
    ])
    ratios = np.array([
        np.sum((Y[i] - Y[j]) ** 2) / np.sum((X[i] - X[j]) ** 2)
        for i in range(n) for j in range(i + 1, n)
    ])
    delta = 2.0 * np.exp(-(eps**2 - eps**3) * k / 4.0)
    outside = float(np.mean((ratios < 1 - eps) | (ratios > 1 + eps)))
    assert outside <= delta
    assert 0.8 < float(np.median(ratios)) < 1.2


def test_pipeline_operator_trains_every_step():
    frames = _square_moving(n=40)
    pipe = FrameFeatures() | IncrementalPCA(k=4) | OnlineRobustMahalanobis(
        feature_keys=[f"pc{j}" for j in range(4)]
    )
    for frame in ArrayStream(frames, fps=10.0):
        pipe.learn_one(frame, t=frame.t)
    ff, pca = pipe.steps[0][1], pipe.steps[1][1]
    assert pca._n == 40
    out = ff.transform_one(np.roll(frames[-1], 1, axis=1))
    assert any(abs(v) > 0 for k, v in out.items() if k.startswith("flow_u"))


class _FakeCap:
    def __init__(self, device, n=3, opened=True):
        self.n, self.opened, self.released = n, opened, False

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.fps = value

    def read(self):
        if self.n == 0:
            return False, None
        self.n -= 1
        img = np.zeros((4, 6, 3), dtype=np.uint8)
        img[..., 2] = 255
        return True, img

    def release(self):
        self.released = True


def _fake_cv2(opened=True):
    import types

    m = types.ModuleType("cv2")
    m.CAP_PROP_FPS = 5
    m.VideoCapture = lambda device: _FakeCap(device, opened=opened)
    return m


def test_camera_stream_with_fake_driver(monkeypatch):
    import sys

    from dense_armor.utility.vision.streams import CameraStream

    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2())
    frames = list(CameraStream(fps=10.0, size=(3, 2), max_frames=2))
    assert len(frames) == 2
    assert frames[0].array.shape == (2, 3)
    assert abs(float(frames[0].array.mean()) - 0.299) < 1e-3
    assert len(list(CameraStream(size=None))) == 3
    with pytest.raises(ValueError):
        CameraStream(fps=0.0)
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2(opened=False))
    with pytest.raises(RuntimeError):
        list(CameraStream())
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(ImportError):
        CameraStream()


def test_stream_conversions_and_paths(tmp_path):
    from PIL import Image

    from dense_armor.utility.vision.streams import Frame, _resize_nearest, _to_float01

    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    assert abs(float(_to_float01(rgba, gray=True).max()) - 0.299) < 1e-3
    assert _to_float01(rgba, gray=False).shape == (2, 2, 3)
    a = np.ones((4, 4), dtype=np.float32)
    assert _resize_nearest(a, (4, 4)) is a
    assert Frame(array=a, t=1.0).to_dict()["t"] == 1.0
    p = tmp_path / "x.png"
    Image.fromarray(np.full((8, 8), 255, dtype=np.uint8)).save(p)
    fr = next(iter(ImageFolderStream([p], size=(4, 4))))
    assert fr.array.shape == (4, 4) and float(fr.array.max()) == 1.0


def test_features_inputs_and_arguments():
    ff = FrameFeatures()
    img = np.zeros((8, 8), dtype=np.float32)
    assert ff.transform_one({"frame": img})["mean"] == 0.0
    ff.transform_one({"other": 1})
    ff.transform_one(object())
    assert ff.n_missing == 2
    ff.learn_one({"other": 1})
    ff._reset()
    assert ff.n_missing == 0
    for kw in ({"n_bins": 0}, {"cells": (0, 1)}, {"flow_cells": (1, 0)}):
        with pytest.raises(ValueError):
            FrameFeatures(**kw)


def test_reduce_missing_reset_and_arguments():
    with pytest.raises(ValueError):
        RandomProjection(k=0)
    with pytest.raises(ValueError):
        IncrementalPCA(k=1, lr=0.0)
    rp = RandomProjection(k=2, seed=0)
    rp.learn_one({"a": 1.0, "b": 2.0})
    rp.learn_one({"a": 1.0, "b": 2.0})
    assert rp.transform_one({"a": 1.0}) == {"p0": 0.0, "p1": 0.0}
    assert rp.transform_one({"a": float("nan"), "b": 1.0}) == {"p0": 0.0, "p1": 0.0}
    assert rp.n_missing == 2
    rp._reset()
    assert rp.n_missing == 0
    pca = IncrementalPCA(k=1, lr=0.5)
    pca.learn_one({"a": 1.0, "b": 2.0})
    pca.learn_one({"a": 1.0})
    pca.learn_one({"a": float("nan"), "b": 1.0})
    assert pca.transform_one({"a": 1.0}) == {"pc0": 0.0}
    assert pca.transform_one({"a": float("inf"), "b": 1.0}) == {"pc0": 0.0}
    assert pca.n_missing == 4
    pca._reset()
    assert pca.n_missing == 0 and pca.components_ is None


def test_vision_small_edge_branches(monkeypatch):
    import dense_armor.utility.vision.features as fmod

    odd = np.zeros((4, 4, 2), dtype=np.float32)
    assert fmod._to_gray(odd) is odd
    tiny = np.zeros((3, 3), dtype=np.float32)
    assert fmod._downsample(tiny) is tiny
    out = FrameFeatures(gray=False).transform_one(np.full((8, 8), np.nan, dtype=np.float32))
    assert out["mean_R"] == 0.0 and out["contrast_B"] == 0.0

    def boom(*a, **k):
        raise np.linalg.LinAlgError

    ff = FrameFeatures(flow_cells=(1, 1))
    a = np.zeros((16, 16), dtype=np.float32)
    a[4:8, 4:8] = 1.0
    ff.learn_one(a)
    monkeypatch.setattr(fmod.np.linalg, "lstsq", boom)
    assert ff.transform_one(np.roll(a, 1, axis=1))["flow_u_0_0"] == 0.0


def test_reduce_ensure_is_idempotent():
    rp = RandomProjection(k=2, seed=0)
    rp._ensure({"a": 1.0, "b": 2.0})
    w = rp._W
    rp._ensure({"a": 1.0, "b": 2.0, "c": 3.0})
    assert rp._W is w
    pca = IncrementalPCA(k=1)
    pca._ensure({"a": 1.0, "b": 2.0})
    v = pca._V
    pca._ensure({"a": 1.0})
    assert pca._V is v
