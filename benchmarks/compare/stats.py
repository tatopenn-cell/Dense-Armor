"""Classic against online, family ``stats``: the difference on synthetic data."""

from __future__ import annotations

import common
import numpy as np
from common import feed

from dense_armor.ml import stats as ml
from dense_armor.utility.stats.dependence import (
    Autocorrelation,
    RollingCorrelation,
    RollingCovariance,
    RunningCorrelation,
    RunningCovariance,
)
from dense_armor.utility.stats.moments import (
    EWStats,
    RunningMoments,
    RunningMomentsVector,
)
from dense_armor.utility.stats.quantiles import DDSketch, TDigest
from dense_armor.utility.stats.robust import RollingMedianVector
from dense_armor.utility.stats.shrinkage import OnlineShrinkageCovariance
from dense_armor.utility.stats.sketches import (
    BloomFilter,
    CountMinSketch,
    HyperLogLog,
    SpaceSaving,
)


def _xy(seed, n=300):
    r = np.random.default_rng(seed)
    x = r.normal(size=n) ** 2
    return x, 0.5 * x + r.normal(size=n)


def _pairs(x, y):
    return [{"x": float(a), "y": float(b)} for a, b in zip(x, y)]


def c_cov(seed):
    x, y = _xy(seed)
    c = ml.Covariance().fit(x, y)
    o = feed(RunningCovariance(), _pairs(x, y))
    return [c.cov_, c.corr_], [o.cov, o.corr], "abs"


def c_corr(seed):
    x, y = _xy(seed)
    c = ml.Correlation().fit(x, y)
    o = feed(RunningCorrelation(), _pairs(x, y))
    return [c.corr_], [o.corr], "abs"


def c_rcov(seed):
    x, y = _xy(seed)
    c = ml.RollingCovariance(window=20).fit(x, y)
    o = feed(RollingCovariance(window=20), _pairs(x, y))
    return [c.cov_], [o.cov], "abs"


def c_rcorr(seed):
    x, y = _xy(seed)
    c = ml.RollingCorrelation(window=20).fit(x, y)
    o = feed(RollingCorrelation(window=20), _pairs(x, y))
    return [c.corr_], [o.corr], "abs"


def c_acf(seed):
    x, _ = _xy(seed)
    c = ml.Autocorrelation(lag=1).fit(x)
    o = feed(Autocorrelation(lag=1), [{"x": float(v)} for v in x])
    return [c.value_], [o.value], "abs"


def c_mom(seed):
    x, _ = _xy(seed)
    c = ml.Moments().fit(x)
    o = feed(RunningMoments(), [{"x": float(v)} for v in x])
    return (
        [c.mean_, c.var_, c.skewness_, c.kurtosis_, c.min_, c.max_],
        [o.mean, o.var, o.skewness, o.kurtosis, o.min, o.max],
        "abs",
    )


def c_momv(seed):
    x, y = _xy(seed)
    c = ml.ColumnMoments().fit(np.c_[x, y])
    o = feed(
        RunningMomentsVector(["a", "b"]),
        [{"a": float(a), "b": float(b)} for a, b in zip(x, y)],
    )
    t = o.transform_one({"a": 0.0, "b": 0.0})
    return (
        np.r_[c.mean_, c.var_],
        [t["a"]["mean"], t["b"]["mean"], t["a"]["var"], t["b"]["var"]],
        "abs",
    )


def c_ew(seed):
    x, _ = _xy(seed)
    c = ml.EWStats(alpha=0.1).fit(x)
    o = feed(EWStats(alpha=0.1), [{"x": float(v)} for v in x])
    return [c.mean_, c.var_], [o.mean, o.var], "abs"


def _q(cls, ocls, seed, **kw):
    x, _ = _xy(seed, 2000)
    c = cls(**kw).fit(x)
    o = feed(ocls(**kw), [{"x": float(v)} for v in x])
    qs = [0.01, 0.1, 0.5, 0.9, 0.99]
    return [c.quantile(q) for q in qs], [o.quantile(q) for q in qs], "rel"


def c_dd(seed):
    return _q(ml.DDSketch, DDSketch, seed)


def c_td(seed):
    return _q(ml.TDigest, TDigest, seed)


def c_rmed(seed):
    x, y = _xy(seed)
    c = ml.RollingMedian(window=15).fit(np.c_[x, y])
    o = feed(
        RollingMedianVector(window=15, features=["a", "b"]),
        [{"a": float(a), "b": float(b)} for a, b in zip(x, y)],
    )
    t = o.transform_one({"a": 0.0, "b": 0.0})
    return c.median_, [t["a"]["value"], t["b"]["value"]], "abs"


def c_shr(seed):
    r = np.random.default_rng(seed)
    X = r.normal(size=(200, 4)) @ r.normal(size=(4, 4))
    c = ml.ShrinkageCovariance(4).fit(X)
    o = feed(
        OnlineShrinkageCovariance(4),
        [{f"v{j}": float(row[j]) for j in range(4)} for row in X],
    )
    return c.shrunk_, o.shrinkage()[0], "abs"


def _items(seed, n=3000):
    r = np.random.default_rng(seed)
    return [f"k{int(v)}" for v in r.zipf(1.5, size=n) % 500]


def c_cm(seed):
    it = _items(seed)
    c = ml.CountMinSketch().fit(it)
    o = feed(CountMinSketch(), [{"x": v} for v in it])
    keys = sorted(set(it))[:50]
    return [c.estimate(k) for k in keys], [o.estimate(k) for k in keys], "rel"


def c_hll(seed):
    it = _items(seed)
    c = ml.HyperLogLog().fit(it)
    o = feed(HyperLogLog(), [{"x": v} for v in it])
    return [c.count_distinct_], [o.count()], "rel"


def c_bloom(seed):
    it = _items(seed)
    c = ml.BloomFilter().fit(it)
    o = feed(BloomFilter(), [{"x": v} for v in it])
    probe = [f"k{i}" for i in range(500, 800)]
    fp = [float(o.check(p)) for p in probe]
    return [float(c.check(p)) for p in probe], fp, "abs"


def c_ss(seed):
    it = _items(seed)
    c = ml.SpaceSaving(k=5).fit(it)
    o = feed(SpaceSaving(k=5), [{"x": v} for v in it])
    top = [k for k, _ in c.top()]
    return [c.estimate(k) for k in top], [o.estimate(k) for k in top], "rel"


CASES = {
    "RunningCovariance": c_cov,
    "RunningCorrelation": c_corr,
    "RollingCovariance": c_rcov,
    "RollingCorrelation": c_rcorr,
    "Autocorrelation": c_acf,
    "RunningMoments": c_mom,
    "RunningMomentsVector": c_momv,
    "EWStats": c_ew,
    "DDSketch": c_dd,
    "TDigest": c_td,
    "RollingMedianVector": c_rmed,
    "OnlineShrinkageCovariance": c_shr,
    "CountMinSketch": c_cm,
    "HyperLogLog": c_hll,
    "BloomFilter": c_bloom,
    "SpaceSaving": c_ss,
}
WHY = {
    "Autocorrelation": "the online estimator differs from the textbook sample autocorrelation",
    "DDSketch": "the sketch is within its relative error alpha of the exact quantile",
    "TDigest": "the digest is within its rank error of the exact quantile",
    "CountMinSketch": "the sketch never undercounts and overcounts within epsilon * n",
    "HyperLogLog": "the standard error of the sketch is 1.04 / sqrt(2^precision)",
    "BloomFilter": "the filter has false positives at rate p_false and no false negatives",
    "SpaceSaving": "the counts of the kept items are upper bounds of the exact counts",
}

if __name__ == "__main__":
    common.run("stats", CASES, WHY)
