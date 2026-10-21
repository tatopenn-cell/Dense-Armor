"""Machine learning against online learning, family ``multiclass``: the difference on synthetic data."""

from __future__ import annotations

import common
import numpy as np

from dense_armor.ml import linear as mll
from dense_armor.ml import multiclass as ml
from dense_armor.utility.linear.glm import LogisticRegression
from dense_armor.utility.multiclass.multiclass import (
    OneVsOneClassifier,
    OneVsRestClassifier,
    OutputCodeClassifier,
)
from dense_armor.utility.optim.sgd import SGD

NAMES = ["a", "b"]


def _data(seed, n=300):
    r = np.random.default_rng(seed)
    y = r.integers(0, 3, size=n)
    centres = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
    return centres[y] + r.normal(size=(n, 2)), y


def _acc(cls, ocls, seed, **kw):
    X, y = _data(seed)
    T, ty = _data(seed + 40, 200)
    c = cls(lambda: mll.LogisticRegression(l2=1e-3), **kw).fit(X, y)
    o = ocls(lambda: LogisticRegression(SGD(lr=0.05)), **kw)
    for row, t in zip(X, y):
        o.learn_one(dict(zip(NAMES, map(float, row))), int(t))
    op = [o.predict_one(dict(zip(NAMES, map(float, r)))) for r in T]
    return (
        [float(np.mean(c.predict(T) == ty))],
        [float(np.mean(np.array(op) == ty))],
        "abs",
    )


def c_ovr(seed):
    return _acc(ml.OneVsRestClassifier, OneVsRestClassifier, seed)


def c_ovo(seed):
    return _acc(ml.OneVsOneClassifier, OneVsOneClassifier, seed)


def c_ecoc(seed):
    return _acc(ml.OutputCodeClassifier, OutputCodeClassifier, seed, code_size=8)


CASES = {
    "OneVsRestClassifier": c_ovr,
    "OneVsOneClassifier": c_ovo,
    "OutputCodeClassifier": c_ecoc,
}
WHY = {
    n: "difference between the test accuracies; the online binary models see each row once"
    for n in CASES
}

if __name__ == "__main__":
    common.run("multiclass", CASES, WHY)
