"""Machine learning against online learning, family ``bayes``: the difference on synthetic data."""

from __future__ import annotations

import common
import numpy as np

from dense_armor.ml import bayes as ml
from dense_armor.utility.bayes.bernoulli import BernoulliNB
from dense_armor.utility.bayes.categorical import CategoricalNB
from dense_armor.utility.bayes.complement import ComplementNB
from dense_armor.utility.bayes.multinomial import MultinomialNB


def _counts(seed, n=200, d=6):
    r = np.random.default_rng(seed)
    y = r.integers(0, 3, size=n)
    X = r.poisson(1.0 + 2.0 * (np.arange(d)[None, :] % 3 == y[:, None]), size=(n, d))
    return X, y


def _pair(cls, ocls, X, y, probe, **kw):
    c = cls(**kw).fit(X, y)
    o = ocls(**kw)
    for row, t in zip(X, y):
        o.learn_one({f"f{j}": row[j] for j in range(len(row))}, int(t))
    cp = c.predict_proba(probe)
    op = np.array(
        [
            [
                o.predict_proba_one({f"f{j}": r[j] for j in range(len(r))}).get(
                    int(k), 0.0
                )
                for k in c.classes_
            ]
            for r in probe
        ]
    )
    return cp, op, "abs"


def c_mult(seed):
    X, y = _counts(seed)
    return _pair(ml.MultinomialNB, MultinomialNB, X, y, _counts(seed + 9, 20)[0])


def c_bern(seed):
    X, y = _counts(seed)
    return _pair(ml.BernoulliNB, BernoulliNB, X, y, _counts(seed + 9, 20)[0])


def c_cat(seed):
    X, y = _counts(seed)
    X = np.minimum(X, 3)
    return _pair(
        ml.CategoricalNB, CategoricalNB, X, y, np.minimum(_counts(seed + 9, 20)[0], 3)
    )


def c_comp(seed):
    X, y = _counts(seed)
    return _pair(ml.ComplementNB, ComplementNB, X, y, _counts(seed + 9, 20)[0])


CASES = {
    "MultinomialNB": c_mult,
    "BernoulliNB": c_bern,
    "CategoricalNB": c_cat,
    "ComplementNB": c_comp,
}
WHY = {n: "the same counts and the same smoothing" for n in CASES}

if __name__ == "__main__":
    common.run("bayes", CASES, WHY)
