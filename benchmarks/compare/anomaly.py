"""Machine learning against online learning, family ``anomaly``: the difference on synthetic data."""

from __future__ import annotations

import common
import numpy as np

from dense_armor.ml import anomaly as ml
from dense_armor.utility.anomaly import robust_filters as rf
from dense_armor.utility.anomaly.lof import LocalOutlierFactor


def _series(seed):
    r = np.random.default_rng(seed)
    x = r.normal(0, 1, 300)
    x[r.integers(0, 300, 8)] += 9.0
    return x


def _rule(model, fn, seed):
    x = _series(seed)
    c = model.fit(x)
    out, an = fn(x)
    return (
        [float(len(c.anomalies_))] + c.cleaned_.tolist(),
        [float(len(an))] + out.tolist(),
        "abs",
    )


def c_hampel(seed):
    return _rule(
        ml.HampelFilter(radius=10), lambda x: rf.hampel_filter(x, radius=10), seed
    )


def c_tukey(seed):
    return _rule(
        ml.TukeyScorer(radius=10), lambda x: rf.tukey_fences(x, radius=10), seed
    )


def c_chauvenet(seed):
    return _rule(
        ml.ChauvenetScorer(radius=10),
        lambda x: rf.chauvenet_criterion(x, radius=10),
        seed,
    )


def c_clip(seed):
    return _rule(
        ml.SigmaClipScorer(radius=10), lambda x: rf.sigma_clip(x, radius=10), seed
    )


def c_lof(seed):
    r = np.random.default_rng(seed)
    X = r.normal(0, 1, size=(100, 2))
    T = np.vstack([r.normal(0, 1, size=(10, 2)), [[6.0, 6.0]]])
    c = ml.LocalOutlierFactor(n_neighbors=10).fit(X)
    o = LocalOutlierFactor(n_neighbors=10, window=100)
    for v in X:
        o.learn_one({"a": float(v[0]), "b": float(v[1])})
    so = [o.score_one({"a": float(v[0]), "b": float(v[1])}) for v in T]
    return c.score(T).tolist(), so, "abs"


CASES = {
    "HampelFilter": c_hampel,
    "TukeyScorer": c_tukey,
    "ChauvenetScorer": c_chauvenet,
    "SigmaClipScorer": c_clip,
    "LocalOutlierFactor": c_lof,
}
WHY = {
    "HampelFilter": "the same rule on the same centred window",
    "TukeyScorer": "the same rule on the same centred window",
    "ChauvenetScorer": "the same rule on the same centred window",
    "SigmaClipScorer": "the same rule on the same centred window",
    "LocalOutlierFactor": "the same definitions, with the whole table as the reference set",
}


if __name__ == "__main__":
    common.run("anomaly", CASES, WHY)
