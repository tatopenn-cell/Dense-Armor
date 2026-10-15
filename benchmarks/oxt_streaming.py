"""OXT measured against the library regressors it shares the online
regression task with.

The stream is small enough that the numbers are indicative, not
conclusive: at 2000 samples and 5 seeds the Wilcoxon test cannot
reject the null below about 0.06 on either side, and on this
particular DriftStream OXT is slower and slightly less accurate
than MondrianForest.

Stream: ``DriftStream`` with one sudden change at half, 2000 samples, 4
features, seeds 0 to 4, label as target, prequential protocol. Reported:
median squared error on the 300 samples before the change, on the first 100
after it and on 300 samples starting 200 after it, the time per sample, and
for OXT against each competitor the paired Wilcoxon signed-rank p-value
(one-sided, OXT lower) and the A12 effect size on the first 100 after the
change.

Run: ``python benchmarks/oxt_streaming.py``
"""

import statistics
import time

import numpy as np
from scipy.stats import wilcoxon

from dense_armor.utility.datasets import DriftStream
from dense_armor.utility.forest import OnlineExtraTreesRegressor
from dense_armor.utility.rules import AMRulesRegressor
from dense_armor.utility.tree import MondrianForestRegressor

N, D = 2000, 4
WINDOWS = {"before": (700, 1000), "first100": (1000, 1100), "after+200": (1200, 1500)}
LEARNERS = {
    "OXT": lambda s: OnlineExtraTreesRegressor(seed=s),
    "MondrianForest": lambda s: MondrianForestRegressor(n_trees=10, seed=s),
    "AMRules": lambda s: AMRulesRegressor(seed=s),
}


def _run(est, X, y):
    names = [f"x{i}" for i in range(D)]
    err = []
    t0 = time.perf_counter()
    for xi, yi in zip(X, y):
        d = dict(zip(names, map(float, xi)))
        err.append((float(est.predict_one(d)) - float(yi)) ** 2)
        est.learn_one(d, float(yi))
    dt = (time.perf_counter() - t0) / len(y)
    e = np.array(err)
    return {k: float(e[a:b].mean()) for k, (a, b) in WINDOWS.items()}, dt


def _a12(a, b):
    return float(
        np.mean([1.0 if p < q else 0.5 if p == q else 0.0 for p in a for q in b])
    )


def main() -> None:
    rows = {k: [] for k in LEARNERS}
    for seed in range(5):
        X, y = DriftStream(
            n_samples=N, n_features=D, n_drifts=1, kind="sudden", seed=seed
        ).data()
        for name, make in LEARNERS.items():
            rows[name].append(_run(make(seed), X, y))
    print(
        f"{'learner':16s} {'before':>8s} {'first100':>9s} {'after+200':>10s} {'ms/sample':>10s}"
    )
    for name, runs in rows.items():
        med = {k: statistics.median(r[k] for r, _ in runs) for k in WINDOWS}
        ms = 1e3 * statistics.median(dt for _, dt in runs)
        print(
            f"{name:16s} {med['before']:8.3f} {med['first100']:9.3f} {med['after+200']:10.3f} {ms:10.2f}"
        )
    mine = [r["first100"] for r, _ in rows["OXT"]]
    for name in ("MondrianForest", "AMRules"):
        other = [r["first100"] for r, _ in rows[name]]
        p = wilcoxon(mine, other, alternative="less").pvalue
        print(
            f"OXT against {name}, first 100 after the change: per seed {np.round(mine, 3)} vs {np.round(other, 3)}, Wilcoxon p = {p:.3f}, A12 = {_a12(mine, other):.2f}"
        )


if __name__ == "__main__":
    main()
