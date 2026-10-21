"""Machine learning against online learning, family ``proba``: the difference on synthetic data."""

from __future__ import annotations

import common
import numpy as np

from dense_armor.ml import proba as ml
from dense_armor.utility.proba.beta import Beta
from dense_armor.utility.proba.gaussian import Gaussian
from dense_armor.utility.proba.multinomial import Multinomial


def c_beta(seed):
    x = (np.random.default_rng(seed).random(300) < 0.3).astype(int)
    c = ml.Beta().fit(x)
    o = Beta()
    for v in x:
        o.update(int(v))
    return (
        [c.mean(), c.variance(), c.pdf(0.3)],
        [o.mean(), o.variance(), o.pdf(0.3)],
        "abs",
    )


def c_gauss(seed):
    x = np.random.default_rng(seed).normal(2.0, 3.0, size=300)
    c = ml.Gaussian().fit(x)
    o = Gaussian()
    for v in x:
        o.update(float(v))
    return (
        [c.mean_, c.var_, c.pdf(2.0), c.cdf(1.0)],
        [o.mean(), o.variance(), o.pdf(2.0), o.cdf(1.0)],
        "abs",
    )


def c_multi(seed):
    x = np.random.default_rng(seed).integers(0, 4, size=300)
    c = ml.Multinomial(alpha=1.0, n_categories=4).fit(x)
    o = Multinomial(alpha=1.0, n_categories=4)
    for v in x:
        o.update(int(v))
    return [c.pmf(k) for k in range(4)], [o.pmf(k) for k in range(4)], "abs"


CASES = {"Beta": c_beta, "Gaussian": c_gauss, "Multinomial": c_multi}
WHY = {n: "the same conjugate update, counted one row at a time" for n in CASES}


if __name__ == "__main__":
    common.run("proba", CASES, WHY)
