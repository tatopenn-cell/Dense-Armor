"""EFDT vs ARF vs HAT, three drift types, two seeds.

Result of the run of this session (median of two seeds, 2000 samples,
4 features, prequential):

    drift      learner    before  first100  after+200  ms/sample
    sudden     EFDT        0.998     1.000      0.955      0.027
    sudden     HAT         0.997     1.000      0.953      0.099
    sudden     ARF-10      0.998     1.000      0.972      1.563
    gradual    EFDT        1.000     0.940      0.983      0.051
    gradual    HAT         1.000     0.935      0.978      0.180
    gradual    ARF-10      1.000     0.940      0.985      1.319
    incremental EFDT       0.933     1.000      0.662      0.021
    incremental HAT        0.958     1.000      0.735      0.070
    incremental ARF-10     0.995     1.000      0.962      0.723

EFDT is 26-58x faster than ARF-10 (0.021-0.051 ms/sample against
0.723-1.563). On sudden and gradual drift the accuracy is at parity
with ARF-10 (within 2 pp). On incremental drift EFDT collapses:
before 0.933 against 0.995 (-6.2 pp), after+200 0.662 against 0.962
(-30.0 pp). The cause is structural: EFDT has no native drift
detection, no warning signal, no background tree; ARF-10 has ADWIN
per member. EFDT is therefore an opt-in speed option, not a
replacement for ARF-10 on drifting streams. The number -30.0 pp is
the declared limit of the option.
"""

import sys, time, statistics
sys.path.insert(0, "/content/Dense-Armor")
import numpy as np
from dense_armor.utility.datasets import DriftStream
from dense_armor.utility.tree import (
    HoeffdingAnytimeTreeClassifier,
    HoeffdingAdaptiveTreeClassifier,
)
from dense_armor.utility.forest import AdaptiveRandomForestClassifier
from dense_armor.utility.drift.adwin import ADWIN

N, D, CUT = 2000, 4, 1000
WINDOWS = {"before": (700, 1000), "first100": (1000, 1100), "after+200": (1200, 1500)}
NAMES = [f"x{i}" for i in range(D)]

def factories():
    return {
        "EFDT": lambda s: HoeffdingAnytimeTreeClassifier(),
        "HAT": lambda s: HoeffdingAdaptiveTreeClassifier(grace_period=50, drift_detector=ADWIN()),
        "ARF-10": lambda s: AdaptiveRandomForestClassifier(n_models=10, seed=s),
    }

def run(est, X, y):
    acc = []
    t0 = time.perf_counter()
    for xi, yi in zip(X, y):
        d = dict(zip(NAMES, map(float, xi)))
        p = est.predict_one(d)
        acc.append(float(p == yi))
        est.learn_one(d, int(yi))
    dt = (time.perf_counter() - t0) / len(y) * 1e3
    a = np.array(acc)
    return {k: float(a[x:y].mean()) for k, (x, y) in WINDOWS.items()}, dt

def main():
    rows = {}
    for kind in ("sudden", "gradual", "incremental"):
        for seed in (0, 1):
            X, y = DriftStream(n_samples=N, n_features=D, n_drifts=2, kind=kind, seed=seed).data()
            for name, make in factories().items():
                res, dt = run(make(seed), X, y)
                rows.setdefault((kind, name), []).append((res, dt))
    print(f"{'drift':10s} {'learner':8s} {'before':>8s} {'first100':>9s} {'after+200':>10s} {'ms/sample':>10s}")
    for kind in ("sudden", "gradual", "incremental"):
        for name in ("EFDT", "HAT", "ARF-10"):
            runs = rows[(kind, name)]
            med = {k: statistics.median(r[k] for r, _ in runs) for k in WINDOWS}
            ms = statistics.median(dt for _, dt in runs)
            print(f"{kind:10s} {name:8s} {med['before']:8.3f} {med['first100']:9.3f} "
                  f"{med['after+200']:10.3f} {ms:10.3f}")

    # Wilcoxon + A12 EFDT vs ARF-10 su before
    try:
        from scipy.stats import wilcoxon
        for kind in ("sudden", "gradual", "incremental"):
            a = np.array([r["before"] for r, _ in rows[(kind, "EFDT")]])
            b = np.array([r["before"] for r, _ in rows[(kind, "ARF-10")]])
            if len(a) >= 3:
                _, p = wilcoxon(a, b)
                gt = np.sum(a[:, None] > b[None, :])
                eq = np.sum(a[:, None] == b[None, :])
                a12 = (gt + 0.5 * eq) / (a.size * b.size)
                print(f"  {kind:10s} EFDT vs ARF-10 (before): p={p:.3f} A12={a12:.3f}")
    except Exception as e:
        print(f"  wilcoxon: {e}")

if __name__ == "__main__":
    main()
