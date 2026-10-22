"""Machine learning against online learning, family ``multioutput``: the difference on synthetic data."""

from __future__ import annotations

import common
import numpy as np

from dense_armor.ml import linear as mll
from dense_armor.ml import multioutput as ml
from dense_armor.utility.linear.glm import LinearRegression, LogisticRegression
from dense_armor.utility.multioutput.multioutput import (
    ClassifierChain,
    PerOutputClassifier,
    PerOutputRegressor,
    ProbabilisticClassifierChain,
    RegressorChain,
)
from dense_armor.utility.optim.sgd import SGD

NAMES = ["a", "b", "c"]
TARGETS = ["t0", "t1"]


def _data(seed, n=300):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, 3))
    Y = np.c_[X @ [1.0, -1.0, 0.5], X @ [0.5, 0.5, -1.0]] + 0.05 * r.normal(size=(n, 2))
    return X, Y


def _x(r):
    return dict(zip(NAMES, map(float, r)))


def _reg(cls, ocls, seed):
    X, Y = _data(seed)
    T = _data(seed + 40, 50)[0]
    c = cls(lambda: mll.LinearRegression(), order=[0, 1]).fit(X, Y)
    o = ocls(lambda: LinearRegression(SGD(lr=0.01)), order=TARGETS)
    for r, y in zip(X, Y):
        o.learn_one(_x(r), dict(zip(TARGETS, map(float, y))))
    op = np.array([[o.predict_one(_x(r))[t] for t in TARGETS] for r in T])
    return c.predict(T), op, "abs"


def _clf(cls, ocls, seed):
    X, Y = _data(seed)
    B = (Y > 0).astype(int)
    T, TY = _data(seed + 40, 100)
    c = cls(lambda: mll.LogisticRegression(l2=1e-3), order=[0, 1]).fit(X, B)
    o = ocls(lambda: LogisticRegression(SGD(lr=0.05)), order=TARGETS)
    for r, y in zip(X, B):
        o.learn_one(_x(r), dict(zip(TARGETS, map(int, y))))
    op = np.array([[o.predict_one(_x(r))[t] for t in TARGETS] for r in T])
    truth = (TY > 0).astype(int)
    return [float(np.mean(c.predict(T) == truth))], [float(np.mean(op == truth))], "abs"


def c_por(seed):
    return _reg(ml.PerOutputRegressor, PerOutputRegressor, seed)


def c_rc(seed):
    return _reg(ml.RegressorChain, RegressorChain, seed)


def c_poc(seed):
    return _clf(ml.PerOutputClassifier, PerOutputClassifier, seed)


def c_cc(seed):
    return _clf(ml.ClassifierChain, ClassifierChain, seed)


def c_pcc(seed):
    return _clf(ml.ProbabilisticClassifierChain, ProbabilisticClassifierChain, seed)


CASES = {
    "PerOutputRegressor": c_por,
    "RegressorChain": c_rc,
    "PerOutputClassifier": c_poc,
    "ClassifierChain": c_cc,
    "ProbabilisticClassifierChain": c_pcc,
}
WHY = {
    "PerOutputRegressor": "the online regressors take one stochastic step per row; the batch ones solve exactly",
    "RegressorChain": "the online regressors take one stochastic step per row; the batch ones solve exactly",
    "PerOutputClassifier": "difference between the test accuracies; the online models see each row once",
    "ClassifierChain": "difference between the test accuracies; the online models see each row once",
    "ProbabilisticClassifierChain": "difference between the test accuracies; the online models see each row once",
}

if __name__ == "__main__":
    common.run("multioutput", CASES, WHY)
