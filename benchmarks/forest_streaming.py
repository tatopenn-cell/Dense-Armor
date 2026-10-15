"""New forests against the library learners they compete with.

Stream: ``DriftStream`` with one sudden change at half, 2000 samples,
4 features, seeds 0 to 4, prequential protocol (predict, then learn).
Reported, median of the seeds: accuracy (or mean squared error of the
label for the regressors) on the 300 samples before the change, on the
first 100 after it, on 300 samples starting 200 after it, and the time
per sample (predict plus learn).

The aggregated Mondrian forests need a box: it is taken from the range
of the whole stream, which the other learners do not need.

Run: ``python benchmarks/forest_streaming.py``
"""

import statistics
import time

import numpy as np

from dense_armor.utility.datasets import DriftStream
from dense_armor.utility.drift.adwin import ADWIN
from dense_armor.utility.forest import (
    AdaptiveRandomForestClassifier,
    AggregatedMondrianForestClassifier,
    AggregatedMondrianForestRegressor,
)
from dense_armor.utility.tree import (
    HoeffdingAdaptiveTreeClassifier,
    MondrianForestClassifier,
    MondrianForestRegressor,
)

N, D, CUT = 2000, 4, 1000
WINDOWS = {"before": (700, 1000), "first100": (1000, 1100), "after+200": (1200, 1500)}


def _factories(lo: float, hi: float) -> dict:
    return {
        "ARF": lambda s: AdaptiveRandomForestClassifier(n_models=10, seed=s),
        "HoeffdingAdaptiveTree": lambda s: HoeffdingAdaptiveTreeClassifier(
            grace_period=50, drift_detector=ADWIN()
        ),
        "AMF classifier": lambda s: AggregatedMondrianForestClassifier(
            n_trees=10, low=lo, high=hi, seed=s
        ),
        "MondrianForest classifier": lambda s: MondrianForestClassifier(
            n_trees=10, seed=s
        ),
        "AMF regressor": lambda s: AggregatedMondrianForestRegressor(
            n_trees=10, low=lo, high=hi, seed=s
        ),
        "MondrianForest regressor": lambda s: MondrianForestRegressor(
            n_trees=10, seed=s
        ),
    }


def _run(est, X, y, regression: bool) -> tuple[dict, float]:
    names = [f"x{i}" for i in range(D)]
    score = []
    t0 = time.perf_counter()
    for xi, yi in zip(X, y):
        d = dict(zip(names, map(float, xi)))
        p = est.predict_one(d)
        if regression:
            score.append((float(p) - float(yi)) ** 2)
            est.learn_one(d, float(yi))
        else:
            score.append(float(p == yi))
            est.learn_one(d, int(yi))
    per_sample = (time.perf_counter() - t0) / len(y)
    arr = np.array(score)
    return {k: float(arr[a:b].mean()) for k, (a, b) in WINDOWS.items()}, per_sample


def main() -> None:
    rows: dict = {}
    for seed in range(5):
        X, y = DriftStream(
            n_samples=N, n_features=D, n_drifts=1, kind="sudden", seed=seed
        ).data()
        facs = _factories(float(X.min()), float(X.max()))
        for name, make in facs.items():
            res, dt = _run(make(seed), X, y, "regressor" in name)
            rows.setdefault(name, []).append((res, dt))
    print(
        f"{'learner':28s} {'before':>8s} {'first100':>9s} {'after+200':>10s} {'ms/sample':>10s}"
    )
    for name, runs in rows.items():
        med = {k: statistics.median(r[k] for r, _ in runs) for k in WINDOWS}
        ms = 1e3 * statistics.median(dt for _, dt in runs)
        print(
            f"{name:28s} {med['before']:8.3f} {med['first100']:9.3f} "
            f"{med['after+200']:10.3f} {ms:10.2f}"
        )


if __name__ == "__main__":
    main()
