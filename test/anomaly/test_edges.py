"""Edge-case tests for the anomaly detectors (input handling)."""

import pytest

from dense_armor.utility.anomaly.hst import HalfSpaceTrees
from dense_armor.utility.anomaly.loda import LODA
from dense_armor.utility.anomaly.mahalanobis import OnlineRobustMahalanobis
from dense_armor.utility.anomaly.ocsvm import OneClassSGD
from dense_armor.utility.anomaly.oiforest import OnlineIsolationForest

BAD_INPUTS: list = [None, {}, "abc"]


@pytest.mark.parametrize(
    "name",
    ["hst", "loda", "ocsvm", "oiforest", "mahalanobis"],
)
def test_detector_counts_none_and_empty_as_missing(name):
    """None, {}, "abc" counted in n_missing by every detector."""
    if name == "hst":
        det = HalfSpaceTrees(n_trees=3, depth=4, window=20, range_init=5, seed=0)
    elif name == "loda":
        det = LODA(n_projections=5, n_bins=4, window=20, range_init=5, seed=0)
    elif name == "ocsvm":
        det = OneClassSGD(n_features_rff=16, n_init=5, seed=0)
    elif name == "oiforest":
        det = OnlineIsolationForest(n_trees=4, window=20, max_leaf_samples=4, seed=0)
    else:
        det = OnlineRobustMahalanobis(n_init=5)
    before = det.n_missing
    for bad in BAD_INPUTS:
        det.learn_one(bad)
    assert det.n_missing == before + len(BAD_INPUTS), (
        f"{name}: n_missing={det.n_missing}, atteso {before + len(BAD_INPUTS)}"
    )


def test_hst_extra_non_numeric_key_is_not_missing():
    """Una chiave extra non numerica fuori feature_keys non conta."""
    det = HalfSpaceTrees(
        n_trees=3,
        depth=4,
        window=20,
        range_init=5,
        feature_keys=["a", "b"],
        threshold=4.0,
        seed=0,
    )
    before = det.n_missing
    det.learn_one({"a": 0.5, "b": 0.5, "note": "text"})
    assert det.n_missing == before, (
        f"chiave extra deve essere ignorata: n_missing={det.n_missing}"
    )
