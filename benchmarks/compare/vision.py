"""Machine learning against online learning, family ``vision``: the difference on synthetic data."""

from __future__ import annotations

import common
import numpy as np

from dense_armor.ml import vision as ml
from dense_armor.utility.vision.dense import DenseFlow
from dense_armor.utility.vision.reduce import IncrementalPCA, RandomProjection


def _frames(seed):
    r = np.random.default_rng(seed)
    a = r.normal(0, 1, (32, 32)).astype(np.float32)
    return a, np.roll(a, (1, 2), axis=(0, 1))


def c_flow(seed):
    a, b = _frames(seed)
    c = ml.DenseFlow(alpha=1.0, n_iter=20).fit(a, b)
    o = DenseFlow(alpha=1.0, n_iter=20, use_jax=False)
    o.learn_one({"array": a})
    r = o.transform_one({"array": b})
    return (
        c.u_.ravel().tolist() + c.v_.ravel().tolist(),
        r["u"].ravel().tolist() + r["v"].ravel().tolist(),
        "abs",
    )


def c_rp(seed):
    X = np.random.default_rng(seed).normal(size=(20, 30))
    c = ml.RandomProjection(k=8, seed=seed).fit(X).transform(X)
    o = RandomProjection(k=8, seed=seed)
    rows = [
        list(o.transform_one({f"f{i:02d}": float(v) for i, v in enumerate(x)}).values())
        for x in X
    ]
    return c.ravel().tolist(), np.array(rows).ravel().tolist(), "abs"


def c_pca(seed):
    r = np.random.default_rng(seed)
    X = r.normal(0, 1, (600, 2)) * np.array([3.0, 0.5])
    c = ml.PCA(k=1).fit(X)
    o = IncrementalPCA(k=1)
    for x in X:
        o.learn_one({"a": float(x[0]), "b": float(x[1])})
    t = o.transform_one({"a": 1.0, "b": 0.0})
    proj_c = float(abs(c.transform([[1.0, 0.0]])[0, 0]))
    proj_o = abs(float(list(t.values())[0]))
    return [proj_c], [proj_o], "abs"


CASES = {
    "DenseFlow": c_flow,
    "RandomProjection": c_rp,
    "PCA": c_pca,
}
WHY = {
    "DenseFlow": "the same iteration on the same pair of frames",
    "RandomProjection": "the same matrix from the same seed",
    "PCA": "the eigenvectors of the covariance of the whole table, against the rule that approaches them one row at a time",
}


if __name__ == "__main__":
    common.run("vision", CASES, WHY)
