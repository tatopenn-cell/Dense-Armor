"""Machine learning against online learning, family ``optim``: the difference on synthetic data."""

from __future__ import annotations

import common
import numpy as np

from dense_armor.ml import optim as ml
from dense_armor.utility.optim import SGD, AdaGrad, Adam, FTRLProximal, OnlineNewtonStep


def _problem(seed, d=4):
    r = np.random.default_rng(seed)
    M = r.normal(size=(d, d))
    A = M @ M.T + np.eye(d)
    b = r.normal(size=d)
    return (lambda w: A @ w - b), r.normal(size=d)


def _pair(cls, ocls, seed, n=30, **kw):
    grad, w0 = _problem(seed)
    c = cls(**kw).minimize(grad, w0, n)
    o = ocls(**kw)
    w = w0.copy()
    for _ in range(n):
        w = np.asarray(o.step(w, grad(w)))
    return c, w, "abs"


def c_sgd(seed):
    return _pair(ml.GradientDescent, SGD, seed, lr=0.02)


def c_ada(seed):
    return _pair(ml.AdaGrad, AdaGrad, seed, lr=0.3)


def c_adam(seed):
    return _pair(ml.Adam, Adam, seed, lr=0.05)


def c_ftrl(seed):
    return _pair(
        ml.FTRLProximal, FTRLProximal, seed, alpha=0.5, beta=1.0, l1=0.01, l2=0.01
    )


def c_ons(seed):
    return _pair(ml.NewtonStep, OnlineNewtonStep, seed, gamma=1.0, eps=1e-3)


CASES = {
    "SGD": c_sgd,
    "AdaGrad": c_ada,
    "Adam": c_adam,
    "FTRLProximal": c_ftrl,
    "OnlineNewtonStep": c_ons,
}
WHY = {n: "the same update rule applied to the same gradient" for n in CASES}

if __name__ == "__main__":
    common.run("optim", CASES, WHY)
