"""Machine learning against online learning, family ``tree``: the difference on synthetic data."""

from __future__ import annotations

import common
import numpy as np

from dense_armor.ml import tree as ml
from dense_armor.utility.tree.batch import BatchDecisionTreeClassifier
from dense_armor.utility.tree.mondrian import (
    MondrianForestClassifier,
    MondrianForestRegressor,
)


def _cls(seed):
    r = np.random.default_rng(seed)
    X = r.normal(size=(300, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X[:200], y[:200], X[200:], y[200:]


def _d(X):
    return [{"a": float(r[0]), "b": float(r[1])} for r in X]


def c_tree(seed):
    X, y, T, t = _cls(seed)
    c = ml.DecisionTreeClassifier(max_depth=4).fit(X, y)
    o = BatchDecisionTreeClassifier(max_depth=4)
    o.fit(_d(X), y.tolist())
    po = [o.predict_one(d) for d in _d(T)]
    return (
        [float((c.predict(T) == t).mean())],
        [float(np.mean(np.array(po) == t))],
        "abs",
    )


def c_mfc(seed):
    X, y, T, t = _cls(seed)
    c = ml.MondrianForestClassifier(n_trees=10, lifetime=3.0, seed=seed).fit(X, y)
    o = MondrianForestClassifier(n_trees=10, lifetime=3.0, seed=seed)
    for d, v in zip(_d(X), y.tolist()):
        o.learn_one(d, v)
    po = [o.predict_one(d) for d in _d(T)]
    return (
        [float((c.predict(T) == t).mean())],
        [float(np.mean(np.array(po) == t))],
        "abs",
    )


def c_mfr(seed):
    r = np.random.default_rng(seed)
    X = r.uniform(-2, 2, size=(300, 1))
    y = np.sin(X[:, 0]) + 0.1 * r.normal(size=300)
    T = r.uniform(-2, 2, size=(100, 1))
    c = ml.MondrianForestRegressor(n_trees=10, lifetime=3.0, seed=seed).fit(X, y)
    o = MondrianForestRegressor(n_trees=10, lifetime=3.0, seed=seed)
    for v, t in zip(X, y):
        o.learn_one({"a": float(v[0])}, float(t))
    po = np.array([o.predict_one({"a": float(v[0])}) for v in T])
    f = np.sin(T[:, 0])
    return (
        [float(np.sqrt(np.mean((c.predict(T) - f) ** 2)))],
        [float(np.sqrt(np.mean((po - f) ** 2)))],
        "abs",
    )


CASES = {
    "DecisionTreeClassifier": c_tree,
    "MondrianForestClassifier": c_mfc,
    "MondrianForestRegressor": c_mfr,
}
WHY = {
    "DecisionTreeClassifier": "the same gini splits on the same table",
    "MondrianForestClassifier": "the trees are sampled from the whole table at once, not grown row by row, so the random splits differ",
    "MondrianForestRegressor": "the trees are sampled from the whole table at once, not grown row by row, so the random splits differ",
}


if __name__ == "__main__":
    common.run("tree", CASES, WHY)
