"""Machine learning against online learning, family ``preprocessing``: the difference on synthetic data."""

from __future__ import annotations

import common
import numpy as np
from common import feed

from dense_armor.ml import preprocessing as ml
from dense_armor.utility.preprocessing.joints import JointDerivatives, JointPower
from dense_armor.utility.preprocessing.scale import (
    EWScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
from dense_armor.utility.preprocessing.select import SelectKBest, VarianceThreshold
from dense_armor.utility.preprocessing.text import (
    TFIDF,
    BagOfWords,
    FeatureHasher,
    Tokenizer,
)
from dense_armor.utility.preprocessing.vectorizer import TextVectorizer

NAMES = ["a", "b", "c"]
DOCS = [
    "the robot arm moved the cup to the table",
    "a drone flew over the field and the river",
    "the arm and the drone share one controller",
    "cup table river field controller robot",
    "one more document about the robot and the cup",
    "drone controller for the arm",
]


def _X(seed, n=200):
    r = np.random.default_rng(seed)
    return r.normal(size=(n, 3)) * [1.0, 5.0, 0.2] + [0.0, 3.0, -1.0]


def _rows(X):
    return [dict(zip(NAMES, map(float, row))) for row in X]


def _scaler(cls, ocls, seed, **kw):
    X = _X(seed)
    probe = _X(seed + 100, 5)
    c = cls(**kw).fit(X).transform(probe)
    o = feed(ocls(**kw), _rows(X))
    return (
        c,
        np.array([[o.transform_one(r)[k] for k in NAMES] for r in _rows(probe)]),
        "abs",
    )


def c_std(seed):
    return _scaler(ml.StandardScaler, StandardScaler, seed)


def c_ew(seed):
    return _scaler(ml.EWScaler, EWScaler, seed, alpha=0.1)


def c_rob(seed):
    return _scaler(ml.RobustScaler, RobustScaler, seed, window=20)


def c_mm(seed):
    return _scaler(ml.MinMaxScaler, MinMaxScaler, seed)


def c_var(seed):
    X = _X(seed)
    X[:, 2] = 1.0
    c = ml.VarianceThreshold(threshold=0.01).fit(X)
    o = feed(VarianceThreshold(threshold=0.01), _rows(X))
    kept = set(o.transform_one(_rows(X)[0]))
    return c.keep_.astype(float), np.array([float(k in kept) for k in NAMES]), "abs"


def c_kb(seed):
    X = _X(seed)
    y = 2.0 * X[:, 1] + 0.1 * X[:, 0] + np.random.default_rng(seed).normal(size=len(X))
    c = ml.SelectKBest(k=2).fit(X, y)
    o = SelectKBest(k=2)
    for row, t in zip(_rows(X), y):
        o.learn_one(row, float(t))
    kept = set(o.transform_one(_rows(X)[0]))
    sel = np.zeros(3)
    sel[c.selected_] = 1.0
    return sel, np.array([float(k in kept) for k in NAMES]), "abs"


def c_jd(seed):
    r = np.random.default_rng(seed)
    q = np.cumsum(r.normal(size=(60, 2)), axis=0)
    t = np.arange(60) * 0.01
    c = ml.JointDerivatives(order=3).fit(q, t)
    o = JointDerivatives(order=3)
    out = None
    for qi, ti in zip(q, t):
        out = o.transform_one({"q": qi}, t=float(ti))
        o.learn_one({"q": qi}, t=float(ti))
    return (
        np.r_[c.qd_[-1], c.qdd_[-1], c.jerk_[-1]],
        np.r_[out["qd"], out["qdd"], out["jerk"]],
        "abs",
    )


def c_jp(seed):
    r = np.random.default_rng(seed)
    tau, qd = r.normal(size=(1, 3)), r.normal(size=(1, 3))
    c = ml.JointPower().fit(tau, qd)
    out = JointPower().transform_one({"tau": tau[0], "qd": qd[0]})
    return (
        np.r_[c.power_[0], c.total_[0]],
        np.r_[np.asarray(out["power"]), out["power_total"]],
        "abs",
    )


def c_tok(seed):
    t = ml.Tokenizer(ngrams=2, stop_words=["the"])
    o = Tokenizer(ngrams=2, stop_words=["the"])
    same = [
        float(t.tokenize(d) == o.transform_one({"text": d})["tokens"]) for d in DOCS
    ]
    return np.ones(len(DOCS)), np.array(same), "abs"


def c_bow(seed):
    toks = [ml.Tokenizer().tokenize(d) for d in DOCS]
    c = ml.BagOfWords().fit(toks).counts_
    same = [
        float(c[i] == BagOfWords().transform_one({"tokens": toks[i]})["counts"])
        for i in range(len(toks))
    ]
    return np.ones(len(toks)), np.array(same), "abs"


def c_tfidf(seed):
    toks = [ml.Tokenizer().tokenize(d) for d in DOCS]
    c = ml.TFIDF().fit(toks)
    o = feed(TFIDF(), [{"tokens": t} for t in toks])
    cs = c.transform_one(toks[0])
    os_ = o.transform_one({"tokens": toks[0]})["scores"]
    keys = sorted(cs)
    return [cs[k] for k in keys], [os_[k] for k in keys], "abs"


def c_hash(seed):
    docs = [{"robot": 2.0, "arm": 1.0}, {"drone": 3.0, "arm": 1.0}]
    c = ml.FeatureHasher(n_features=64, seed=seed).fit_transform(docs)
    o = FeatureHasher(n_features=64, seed=seed)
    rows = []
    for d in docs:
        h = o.transform_one({"counts": d})["hashed"]
        v = np.zeros(64)
        for k, w in h.items():
            v[int(k)] = w
        rows.append(v)
    return c, np.array(rows), "abs"


def c_vec(seed):
    c = ml.TextVectorizer(max_features=12).fit(DOCS)
    o = TextVectorizer(max_features=12).fit(DOCS)
    m = o.transform_matrix(DOCS[:3])
    return (
        c.transform(DOCS[:3]),
        np.asarray(m.todense() if hasattr(m, "todense") else m),
        "abs",
    )


CASES = {
    "StandardScaler": c_std,
    "EWScaler": c_ew,
    "RobustScaler": c_rob,
    "MinMaxScaler": c_mm,
    "VarianceThreshold": c_var,
    "SelectKBest": c_kb,
    "JointDerivatives": c_jd,
    "JointPower": c_jp,
    "Tokenizer": c_tok,
    "BagOfWords": c_bow,
    "TFIDF": c_tfidf,
    "FeatureHasher": c_hash,
    "TextVectorizer": c_vec,
}
WHY = {
    "JointDerivatives": "rounding of a polynomial through four samples 0.01 s apart",
    "TextVectorizer": "the online matrix is float32, the batch one float64",
}

if __name__ == "__main__":
    common.run("preprocessing", CASES, WHY)
