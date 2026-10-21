"""Machine learning against online learning, family ``neighbors``: the difference on synthetic data."""

from __future__ import annotations

import common
import numpy as np

from dense_armor.ml import neighbors as ml
from dense_armor.utility.neighbors.knn import KNNClassifier, KNNRegressor


def _data(seed, n=150):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return X, y


def _rows(X):
    return [{f"f{j}": float(v) for j, v in enumerate(row)} for row in X]


def c_clf(seed):
    X, y = _data(seed)
    probe = _data(seed + 7, 20)[0]
    c = ml.KNNClassifier(n_neighbors=5).fit(X, y)
    o = KNNClassifier(window_size=len(X), n_neighbors=5)
    for row, t in zip(_rows(X), y):
        o.learn_one(row, int(t))
    op = np.array(
        [
            [o.predict_proba_one(r).get(int(k), 0.0) for k in c.classes_]
            for r in _rows(probe)
        ]
    )
    return c.predict_proba(probe), op, "abs"


def c_reg(seed):
    X, y = _data(seed)
    t = X @ [1.0, -1.0, 0.5]
    probe = _data(seed + 7, 20)[0]
    c = ml.KNNRegressor(n_neighbors=5, weights="distance").fit(X, t)
    o = KNNRegressor(window_size=len(X), n_neighbors=5, weights="distance")
    for row, ti in zip(_rows(X), t):
        o.learn_one(row, float(ti))
    return c.predict(probe), [o.predict_one(r) for r in _rows(probe)], "abs"


CASES = {"KNNClassifier": c_clf, "KNNRegressor": c_reg}
WHY = {
    n: "the online model searches a window as large as the table, so the two see the same rows"
    for n in CASES
}

if __name__ == "__main__":
    common.run("neighbors", CASES, WHY)
