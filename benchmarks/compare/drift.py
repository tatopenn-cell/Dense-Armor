"""Machine learning against online learning, family ``drift``: the difference on synthetic data."""

from __future__ import annotations

import common
import numpy as np

from dense_armor.ml import drift as ml
from dense_armor.utility.drift.cusum import cusum_detector


def c_cusum(seed):
    r = np.random.default_rng(seed)
    x = np.concatenate([r.normal(0, 1, 300), r.normal(1.5, 1, 200)])
    m = ml.CUSUMDriftDetector(h=20.0).fit(x)
    f, s = cusum_detector(x, reference="fixed", h=20.0)
    return (
        [float(m.flagged_.sum())] + m.cusum_.tolist(),
        [float(f.sum())] + s.tolist(),
        "abs",
    )


CASES = {"CUSUMDriftDetector": c_cusum}
WHY = {
    "CUSUMDriftDetector": "the same fixed-reference recursion of Page, read over the whole series"
}


if __name__ == "__main__":
    common.run("drift", CASES, WHY)
