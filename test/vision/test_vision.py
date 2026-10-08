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
        f[8:16, x:x + 8] = 1.0
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
        Image.fromarray((f * 255).astype(np.uint8)).save(
            tmp_path / f"{i:03d}.png"
        )
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
    a = np.zeros((16, 16), dtype=np.float32)
    a[6:10, 6:10] = 1.0
    b = np.zeros((16, 16), dtype=np.float32)
    b[6:10, 7:11] = 1.0
    ff = FrameFeatures(flow_cells=(2, 2))
    _ = ff.transform_one(a)
    _ = ff.learn_one(a)
    out = ff.transform_one(b)
    us = [out[f"flow_u_{j // 2}_{j % 2}"] for j in range(4)]
    assert abs(float(np.mean(us)) - 1.0) < 0.5


def test_features_nan_frame_counted_missing():
    ff = FrameFeatures()
    bad = np.full((16, 16), np.nan, dtype=np.float32)
    _ = ff.transform_one(bad)
    assert ff.n_missing >= 1


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
    assert angle < 2.0


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
    feats = FrameFeatures()
    pca = IncrementalPCA(k=4)
    det = OnlineRobustMahalanobis(
        feature_keys=[f"pc{j}" for j in range(4)]
    )
    scores = []
    for frame in stream:
        f = feats.transform_one(frame)
        _ = feats.learn_one(frame)
        p = pca.transform_one(f)
        _ = pca.learn_one(f)
        s = det.score_one(p)
        _ = det.learn_one(p)
        scores.append(s)
    before = float(np.mean(scores[:80]))
    after = float(np.mean(scores[80:]))
    assert after > before


@pytest.mark.parametrize(
    "cls",
    [FrameFeatures, RandomProjection, IncrementalPCA],
)
def test_check_estimator(cls):
    kwargs = {}
    if cls is RandomProjection or cls is IncrementalPCA:
        kwargs = {"k": 4}
    check_estimator(cls(**kwargs))


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
