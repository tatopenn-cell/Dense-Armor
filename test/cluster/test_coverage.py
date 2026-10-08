"""Edge paths of the clustering package and the visual vocabulary."""
import numpy as np
import pytest

from dense_armor.utility.cluster.denstream import DenStream
from dense_armor.utility.cluster.kmeans import OnlineKMeans
from dense_armor.utility.vision.vocabulary import BagOfVisualWords


def test_kmeans_invalid_inputs_and_arguments():
    for kw in ({"k": 0}, {"k": 2, "halflife": 0.0}, {"k": 3, "warmup": 2}):
        with pytest.raises(ValueError):
            OnlineKMeans(**kw)
    km = OnlineKMeans(k=2, warmup=2)
    assert km.predict_one({"x": [1.0]}) == 0
    assert km.transform_one({"x": [1.0]}) == {"cluster": 0, "distances": []}
    km.learn_one({})
    km.learn_one({"x": ["a"]})
    km.learn_one({"x": [0.0]})
    km.learn_one({"x": [0.0, 1.0]})
    km.learn_one({"x": [5.0]})
    km.learn_one({"x": [1.0, 2.0]})
    assert km.n_missing == 3
    assert km.predict_one({"x": [1.0, 2.0]}) == 0
    assert km.transform_one({"x": [1.0, 2.0]}) == {"cluster": 0, "distances": []}


def test_denstream_arguments_transform_and_chains():
    for kw in ({"eps": 0.0}, {"beta": 0.0}, {"mu": 0.0}, {"decay": 0.0}):
        with pytest.raises(ValueError):
            DenStream(**kw)
    ds = DenStream(eps=1.0, beta=0.5, mu=2.0, decay=0.01)
    assert ds.transform_one({"x": [0.0, 0.0]})["cluster"] == -1
    ds.learn_one({"x": ["bad"]})
    t = 0.0
    for c in [[0.0, 0.0]] * 5 + [[1.2, 0.0]] * 5 + [[0.55, 0.0]] * 6:
        ds.learn_one({"x": c}, t=t)
        t += 1.0
    assert len(ds.pmc_) == 2
    near = ds.transform_one({"x": [0.1, 0.0]})
    far = ds.transform_one({"x": [50.0, 50.0]})
    assert near["cluster"] >= 0 and far["cluster"] == -1
    assert ds.transform_one({}) == {"cluster": -1, "distance": float("inf")}
    clusters, labels = ds.macro_clusters()
    assert len(clusters) == 1 and len(set(labels)) == 1
    assert ds.n_missing == 1


def test_vocabulary_arguments_and_edge_frames():
    for kw in ({"n_words": 0}, {"patch_size": 1}, {"n_bins": 0}):
        with pytest.raises(ValueError):
            BagOfVisualWords(**kw)
    bow = BagOfVisualWords(n_words=2, patch_size=4, n_bins=4)
    assert bow.transform_one(np.zeros((8, 8), dtype=np.float32))["hist"] == [0.0, 0.0]
    assert bow.transform_one(object())["hist"] == [0.0, 0.0]
    assert bow.transform_one({"other": 1})["hist"] == [0.0, 0.0]
    bow.learn_one(np.zeros((2, 2), dtype=np.float32))
    bow.learn_one(np.full((8, 8), np.nan, dtype=np.float32))
    rgb = np.zeros((8, 8, 3), dtype=np.float32)
    rgb[:, 4:, 0] = 1.0
    for _ in range(10):
        bow.learn_one(rgb)
    assert abs(sum(bow.transform_one(rgb)["hist"]) - 1.0) < 1e-9
    assert bow.n_missing >= 3
