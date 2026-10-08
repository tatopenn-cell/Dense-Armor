"""Tests for the local patch features and the bounded patch memory."""

import numpy as np
import pytest

from dense_armor.checks import check_estimator
from dense_armor.utility.vision.patches import PatchFeatures, PatchMemory


def _textured(size=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.2, 0.8, size=(size, size)).astype(np.float32)


def _with_scratched(img, y=30, x=30, h=4, w=4, value=1.0):
    out = img.copy()
    out[y : y + h, x : x + w] = value
    return out


def _dict(desc, n):
    return {"descriptors": desc, "positions": np.zeros((n, 2), dtype=np.intp)}


def test_patch_features_shapes():
    pf = PatchFeatures()
    img = np.zeros((160, 120), dtype=np.float32)
    out = pf.transform_one(img)
    desc = out["descriptors"]
    pos = out["positions"]
    assert desc is not None and pos is not None
    assert desc.shape == (266, 11)
    assert pos.shape == (266, 2)


def test_patch_features_constant_frame_zero_gradient():
    pf = PatchFeatures(neighbourhood=1)
    img = np.full((32, 32), 0.5, dtype=np.float32)
    desc = pf.transform_one(img)["descriptors"]
    assert desc is not None
    for j in range(pf.n_bins):
        assert float(np.abs(desc[:, j]).max()) < 1e-9
    assert float(np.abs(desc[:, pf.n_bins] - 0.5).max()) < 1e-9
    assert float(np.abs(desc[:, pf.n_bins + 1]).max()) < 1e-9


def test_patch_features_horizontal_edge_one_bin():
    pf = PatchFeatures(patch=16, stride=16, n_bins=4, neighbourhood=1)
    img = np.zeros((16, 16), dtype=np.float32)
    img[8:, :] = 1.0
    desc = pf.transform_one(img)["descriptors"]
    assert desc is not None
    assert desc.shape == (1, 6)
    bins = desc[0, :4]
    assert float(bins.max()) > 0.9
    assert float(bins.sum() - bins.max()) < 0.1


def test_patch_features_aggregation_smooths():
    rng = np.random.default_rng(1)
    img = rng.uniform(0.0, 1.0, size=(48, 48)).astype(np.float32)
    pf1 = PatchFeatures(patch=16, stride=8, neighbourhood=1)
    pf3 = PatchFeatures(patch=16, stride=8, neighbourhood=3)
    d1 = pf1.transform_one(img)["descriptors"]
    d3 = pf3.transform_one(img)["descriptors"]
    assert d1 is not None and d3 is not None
    assert d1.shape == d3.shape
    assert not np.allclose(d1, d3)
    assert float(np.abs(np.diff(d3[:, 9])).mean()) < float(
        np.abs(np.diff(d1[:, 9])).mean()
    )


def test_patch_features_nan_frame_counted_missing():
    pf = PatchFeatures()
    bad = np.full((64, 64), np.nan, dtype=np.float32)
    out = pf.transform_one(bad)
    assert out["descriptors"] is None
    assert out["positions"] is None
    assert pf.n_missing == 1


def test_patch_features_too_small():
    pf = PatchFeatures(patch=32, stride=16)
    img = np.zeros((16, 16), dtype=np.float32)
    out = pf.transform_one(img)
    assert out["descriptors"] is None
    assert out["positions"] is None
    assert pf.n_missing == 1


def test_patch_features_rejects_bad_params():
    with pytest.raises(ValueError):
        PatchFeatures(patch=0)
    with pytest.raises(ValueError):
        PatchFeatures(stride=0)
    with pytest.raises(ValueError):
        PatchFeatures(patch=8, stride=16)
    with pytest.raises(ValueError):
        PatchFeatures(n_bins=0)
    with pytest.raises(ValueError):
        PatchFeatures(neighbourhood=0)


def test_patch_memory_rejects_bad_mode():
    with pytest.raises(ValueError):
        PatchMemory(mode="random")


def test_patch_memory_reservoir_scratch_scores_higher():
    pf = PatchFeatures()
    base = _textured(64, seed=0)
    scratch = _with_scratched(base)
    mem = PatchMemory(max_size=5000, seed=0, mode="reservoir")
    for _ in range(5):
        mem.learn_one(pf.transform_one(base))
    s_scr = mem.score_one(pf.transform_one(scratch))
    s_norm = mem.score_one(pf.transform_one(base))
    assert s_scr > s_norm


def test_patch_memory_reservoir_top_patch_at_scratch():
    pf = PatchFeatures()
    base = _textured(64, seed=0)
    scratch = _with_scratched(base, y=24, x=32, h=8, w=8)
    mem = PatchMemory(max_size=5000, seed=0, mode="reservoir")
    for _ in range(5):
        mem.learn_one(pf.transform_one(base))
    scores, positions = mem.patch_scores(pf.transform_one(scratch))
    top = int(np.argmax(scores))
    row, col = int(positions[top, 0]), int(positions[top, 1])
    y_expected = (24 + 32) // 2
    x_expected = (32 + 40) // 2
    r_expected = (y_expected - pf.patch // 2) // pf.stride
    c_expected = (x_expected - pf.patch // 2) // pf.stride
    assert abs(row - r_expected) <= 1 and abs(col - c_expected) <= 1


def test_patch_memory_reservoir_bounds_size():
    mem = PatchMemory(max_size=50, seed=0, mode="reservoir")
    rng = np.random.default_rng(0)
    for _ in range(200):
        mem.learn_one(_dict(rng.normal(size=(10, 5)), 10))
    assert mem.bank_size == 50


def test_patch_memory_empty_bank_zero_score():
    mem = PatchMemory(max_size=10, seed=0, mode="reservoir")
    assert mem.score_one(_dict(np.zeros((3, 4)), 3)) == 0.0


def test_patch_memory_rejects_mismatched_dim():
    mem = PatchMemory(max_size=10, seed=0, mode="reservoir")
    mem.learn_one(_dict(np.zeros((3, 4)), 3))
    with pytest.raises(ValueError):
        mem.learn_one(_dict(np.zeros((3, 5)), 3))


def test_patch_memory_reservoir_is_outlier():
    mem = PatchMemory(max_size=10, seed=0, mode="reservoir")
    d0 = _dict(np.zeros((3, 2)), 3)
    d1 = _dict(np.full((3, 2), 10.0), 3)
    mem.learn_one(d0)
    assert not mem.is_outlier(d0, threshold=1.0)
    assert mem.is_outlier(d1, threshold=1.0)


def test_patch_memory_nan_counted_missing():
    mem = PatchMemory(max_size=10, seed=0, mode="reservoir")
    mem.learn_one({"descriptors": None, "positions": None})
    assert mem.n_missing == 1


def test_patch_memory_coreset_before_finalize_raises():
    mem = PatchMemory(max_size=20, seed=0, mode="coreset")
    rng = np.random.default_rng(0)
    mem.learn_one(_dict(rng.normal(size=(10, 4)), 10))
    assert mem.bank_size == 0
    with pytest.raises(RuntimeError):
        mem.score_one(_dict(rng.normal(size=(3, 4)), 3))


def test_patch_memory_coreset_learn_after_finalize_raises():
    mem = PatchMemory(max_size=5, seed=0, mode="coreset")
    rng = np.random.default_rng(0)
    mem.learn_one(_dict(rng.normal(size=(10, 4)), 10))
    mem.finalize()
    with pytest.raises(RuntimeError):
        mem.learn_one(_dict(rng.normal(size=(3, 4)), 3))


def test_patch_memory_coreset_finalize_twice_raises():
    mem = PatchMemory(max_size=5, seed=0, mode="coreset")
    rng = np.random.default_rng(0)
    mem.learn_one(_dict(rng.normal(size=(10, 4)), 10))
    mem.finalize()
    with pytest.raises(RuntimeError):
        mem.finalize()


def test_patch_memory_coreset_bank_size():
    mem = PatchMemory(max_size=20, seed=0, mode="coreset")
    rng = np.random.default_rng(0)
    for _ in range(10):
        mem.learn_one(_dict(rng.normal(size=(10, 4)), 10))
    assert mem.n_seen == 100
    mem.finalize()
    assert mem.bank_size == 20
    assert mem.finalized
    assert mem.n_reductions_ == 1


def test_patch_memory_coreset_covers_better_than_random():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(500, 6))
    data[::7] += 20.0

    def _coverage(selected, all_data):
        sq = np.einsum("ij,ij->i", selected, selected)
        worst = 0.0
        for i in range(0, all_data.shape[0], 128):
            q = all_data[i : i + 128]
            sq_q = np.einsum("ij,ij->i", q, q)
            d2 = sq_q[:, None] + sq[None, :] - 2.0 * (q @ selected.T)
            worst = max(worst, float(np.sqrt(np.maximum(d2.min(axis=1), 0.0)).max()))
        return worst

    mem = PatchMemory(max_size=50, seed=0, mode="coreset")
    mem.learn_one(_dict(data, data.shape[0]))
    mem.finalize()
    coreset_bank = mem._bank
    assert coreset_bank is not None
    cov_coreset = _coverage(coreset_bank, data)
    idx = rng.choice(data.shape[0], size=50, replace=False)
    cov_random = _coverage(data[idx], data)
    assert cov_coreset < cov_random


def test_patch_memory_coreset_below_max_size_keeps_all():
    mem = PatchMemory(max_size=100, seed=0, mode="coreset")
    rng = np.random.default_rng(0)
    mem.learn_one(_dict(rng.normal(size=(10, 4)), 10))
    mem.finalize()
    assert mem.bank_size == 10


@pytest.mark.parametrize(
    "cls, kwargs",
    [
        (PatchFeatures, {}),
        (PatchMemory, {"max_size": 200, "mode": "reservoir"}),
    ],
)
def test_check_estimator(cls, kwargs):
    check_estimator(cls(**kwargs))


def test_patch_memory_patch_scores_empty_bank():
    mem = PatchMemory(max_size=10)
    scores, pos = mem.patch_scores(_dict(np.ones((3, 4)), 3))
    assert scores.shape == (0,) and pos.shape == (0, 2)
