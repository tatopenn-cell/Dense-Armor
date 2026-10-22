"""Machine learning against online learning, family ``linear``: the difference on synthetic data."""

from __future__ import annotations

import common
import numpy as np

from dense_armor.ml import linear as ml
from dense_armor.utility.linear.bayes import BayesianLinearRegression
from dense_armor.utility.linear.glm import LinearRegression, LogisticRegression
from dense_armor.utility.optim.sgd import SGD

NAMES = ["a", "b", "c"]


def _data(seed, n=300):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, 3))
    y = X @ [1.0, -2.0, 0.5] + 0.3 + 0.1 * r.normal(size=n)
    return X, y


def _rows(X):
    return [dict(zip(NAMES, map(float, row))) for row in X]


def c_lin(seed):
    X, y = _data(seed)
    probe = _data(seed + 50, 20)[0]
    c = ml.LinearRegression().fit(X, y).predict(probe)
    o = LinearRegression(SGD(lr=0.01))
    for row, t in zip(_rows(X), y):
        o.learn_one(row, float(t))
    return c, [o.predict_one(r) for r in _rows(probe)], "abs"


def c_log(seed):
    X, y = _data(seed)
    t = (y > 0.3).astype(int)
    probe = _data(seed + 50, 20)[0]
    c = ml.LogisticRegression(l2=1e-6).fit(X, t).predict_proba(probe)
    o = LogisticRegression(SGD(lr=0.05))
    for row, ti in zip(_rows(X), t):
        o.learn_one(row, int(ti))
    return c, [o.predict_proba_one(r).get(1, 0.0) for r in _rows(probe)], "abs"


def c_bayes(seed):
    X, y = _data(seed)
    probe = _data(seed + 50, 20)[0]
    c = ml.BayesianLinearRegression(alpha=1.0, beta=100.0).fit(X, y).predict(probe)
    o = BayesianLinearRegression(alpha=1.0, beta=100.0)
    for row, t in zip(_rows(X), y):
        o.learn_one(row, float(t))
    return c, [o.predict_one(r) for r in _rows(probe)], "abs"


CASES = {
    "LinearRegression": c_lin,
    "LogisticRegression": c_log,
    "BayesianLinearRegression": c_bayes,
}
WHY = {
    "LinearRegression": "the online model takes one stochastic gradient step per row, the batch model solves the problem exactly",
    "LogisticRegression": "the online model takes one stochastic gradient step per row, the batch model runs Newton steps to the optimum",
    "BayesianLinearRegression": "the same conjugate posterior: the two differ only by rounding",
}

if __name__ == "__main__":
    common.run("linear", CASES, WHY)
