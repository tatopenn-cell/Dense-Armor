"""Edge paths of the patch features and the patch memory."""

import numpy as np
import pytest

from dense_armor.utility.vision.patches import PatchFeatures, PatchMemory


def test_patch_features_invalid_frames():
    pf = PatchFeatures(patch=4, stride=2)
    for x in (object(), np.zeros((2, 2, 2, 2), dtype=np.float32)):
        assert pf.transform_one(x) == {"descriptors": None, "positions": None}
    assert pf.n_missing == 2


def test_patch_memory_rejects_bad_size():
    with pytest.raises(ValueError):
        PatchMemory(max_size=0)


def test_patch_memory_unpack_variants():
    mem = PatchMemory(max_size=10)
    d = np.ones((3, 2))
    for x in (
        "bad",
        {"descriptors": None},
        {"descriptors": np.ones(3)},
        {"descriptors": np.zeros((0, 2))},
        {"descriptors": np.full((2, 2), np.nan)},
    ):
        mem.learn_one(x)
    assert mem.n_missing == 5
    mem.learn_one((d, np.zeros((1, 2))))
    assert mem.bank_size == 3
    scores, pos = mem.patch_scores((d, None))
    assert scores.shape == (3,) and pos.shape == (3, 2)


def test_patch_memory_reservoir_overflow_paths():
    rng = np.random.default_rng(0)
    mem = PatchMemory(max_size=5, seed=0)
    mem.learn_one({"descriptors": rng.normal(size=(8, 2)), "positions": None})
    assert mem.bank_size == 5 and mem.n_seen == 8
    small = PatchMemory(max_size=5, seed=0)
    small.learn_one({"descriptors": rng.normal(size=(3, 2)), "positions": None})
    small.learn_one({"descriptors": rng.normal(size=(4, 2)), "positions": None})
    assert small.bank_size == 5 and small.n_seen == 7
    assert small.finalize() is small


def test_patch_memory_coreset_edge_paths():
    mem = PatchMemory(max_size=5, mode="coreset")
    mem.learn_one({"descriptors": np.ones((2, 3)), "positions": None})
    with pytest.raises(ValueError):
        mem.learn_one({"descriptors": np.ones((2, 4)), "positions": None})
    empty = PatchMemory(max_size=5, mode="coreset")
    empty.finalize()
    assert empty.finalized and empty.bank_size == 0
    assert empty.score_one({"descriptors": np.ones((2, 3)), "positions": None}) == 0.0
