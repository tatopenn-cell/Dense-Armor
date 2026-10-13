"""Edge-case tests for the anomaly detectors (input handling)."""

import pytest

from dense_armor.utility.anomaly.hst import HalfSpaceTrees
from dense_armor.utility.anomaly.loda import LODA
from dense_armor.utility.anomaly.mahalanobis import OnlineRobustMahalanobis
from dense_armor.utility.anomaly.ocsvm import OneClassSGD
from dense_armor.utility.anomaly.oiforest import OnlineIsolationForest

NAMES = ["hst", "loda", "ocsvm", "oiforest", "mahalanobis"]
BAD_INPUTS: list = [
    None,
    {},
    "abc",
    {"a": None},
    {"a": "x"},
    {"a": float("nan")},
    {"a": float("inf")},
]
BAD_IDS = ["none", "empty", "str", "a-none", "a-str", "a-nan", "a-inf"]


def _make(name, keys):
    if name == "hst":
        return HalfSpaceTrees(
            n_trees=3, depth=4, window=20, range_init=5, feature_keys=keys, seed=0
        )
    if name == "loda":
        return LODA(
            n_projections=5,
            n_bins=4,
            window=20,
            range_init=5,
            feature_keys=keys,
            seed=0,
        )
    if name == "ocsvm":
        return OneClassSGD(n_features_rff=16, n_init=5, feature_keys=keys, seed=0)
    if name == "oiforest":
        return OnlineIsolationForest(
            n_trees=4, window=20, max_leaf_samples=4, feature_keys=keys, seed=0
        )
    return OnlineRobustMahalanobis(n_init=5, feature_keys=keys)


def _count_bad_first_sample(name, keys, bad):
    det = _make(name, keys)
    det.learn_one(bad)
    assert det.n_missing == 1, f"{name}: n_missing={det.n_missing}, expected 1"


@pytest.mark.parametrize("keys", [None, ["a"]], ids=["no-keys", "keys-a"])
@pytest.mark.parametrize("bad", BAD_INPUTS, ids=BAD_IDS)
@pytest.mark.parametrize("name", NAMES)
def test_detector_counts_bad_first_sample_as_missing(name, bad, keys):
    """A malformed first sample is counted in n_missing and raises nothing."""
    _count_bad_first_sample(name, keys, bad)


def test_hst_ignores_extra_non_numeric_key_after_warmup():
    """A non-numeric value under a key outside feature_keys is ignored."""
    det = HalfSpaceTrees(
        n_trees=3,
        depth=4,
        window=20,
        range_init=5,
        feature_keys=["a", "b"],
        threshold=4.0,
        seed=0,
    )
    for i in range(6):
        det.learn_one({"a": 0.1 * i, "b": 0.2})
    before = det.n_missing
    det.learn_one({"a": 0.5, "b": 0.5, "note": "text"})
    assert det.n_missing == before, f"n_missing went from {before} to {det.n_missing}"
