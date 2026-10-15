"""Quality benchmark: every comparison of Task Z item A, 20 seeds, three streams.

Streams: ``DriftStream`` with one drift at half, 2000 samples, 4 features,
kinds ``sudden``, ``gradual``, ``incremental``. Prequential protocol
(predict, then learn). Reported per learner and per window: median over
seeds, IQR, and ms per sample. Reported per pair: two-sided paired
Wilcoxon p-value, the A12 effect size with a bootstrap 95% interval, and
the seeds on which each side is better.

Run: ``python benchmarks/bricks_quality.py`` in Colab (around 20 minutes
on a 1.6 GHz, 2-core CPU). Fast smoke test: ``--quick``.
"""

import argparse
import json
import multiprocessing as mp
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from dense_armor.utility.datasets import DriftStream
from dense_armor.utility.drift.adwin import ADWIN
from dense_armor.utility.forest import (
    AdaptiveRandomForestClassifier,
    AggregatedMondrianForestRegressor,
    OnlineExtraTreesRegressor,
)
from dense_armor.utility.rules import AMRulesRegressor
from dense_armor.utility.tree import (
    HoeffdingAdaptiveTreeClassifier,
    MondrianForestRegressor,
)

N = 2000
D = 4
WINDOWS = {
    "before": (700, 1000),
    "first100": (1000, 1100),
    "after+200": (1200, 1500),
}
STREAMS = ("sudden", "gradual", "incremental")
PAIRS = [
    ("ARF", "HAT"),
    ("AMFr", "Mondr"),
    ("OXT", "Mondr"),
    ("OXT", "AMRules"),
]
LEARNER_NAMES = ["ARF", "HAT", "AMFr", "Mondr", "OXT", "AMRules"]


def _make_learners(seed: int, low: float, high: float) -> dict:
    return {
        "ARF": (AdaptiveRandomForestClassifier(n_models=10, seed=seed), False),
        "HAT": (
            HoeffdingAdaptiveTreeClassifier(
                grace_period=50, drift_detector=ADWIN()
            ),
            False,
        ),
        "AMFr": (
            AggregatedMondrianForestRegressor(
                n_trees=10, low=low, high=high, seed=seed
            ),
            True,
        ),
        "Mondr": (MondrianForestRegressor(n_trees=10, seed=seed), True),
        "OXT": (OnlineExtraTreesRegressor(n_models=10, seed=seed), True),
        "AMRules": (AMRulesRegressor(seed=seed), True),
    }


def _prequential(est, X, y, regression: bool) -> tuple:
    names = [f"x{i}" for i in range(X.shape[1])]
    err = np.empty(len(y))
    t0 = time.perf_counter()
    for i, (xi, yi) in enumerate(zip(X, y)):
        d = dict(zip(names, map(float, xi)))
        p = est.predict_one(d)
        if regression:
            err[i] = (float(p) - float(yi)) ** 2
            est.learn_one(d, float(yi))
        else:
            err[i] = float(p != yi)
            est.learn_one(d, int(yi))
    dt = (time.perf_counter() - t0) / len(y)
    return err, dt


def _run_one(task: tuple) -> tuple:
    seed, kind = task
    X, y = DriftStream(
        n_samples=N, n_features=D, n_drifts=1, kind=kind, seed=seed
    ).data()
    low, high = float(X.min()), float(X.max())
    out = {}
    for name, (est, regression) in _make_learners(seed, low, high).items():
        err, dt = _prequential(est, X, y, regression)
        out[name] = {
            "windows": {
                k: float(err[a:b].mean()) for k, (a, b) in WINDOWS.items()
            },
            "ms_per_sample": 1e3 * dt,
        }
    return seed, kind, out


def _a12(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    diff = a[:, None] - b[None, :]
    return float((diff < 0).mean() + 0.5 * (diff == 0).mean())


def _a12_ci(a: np.ndarray, b: np.ndarray, rng, n_boot: int = 2000) -> tuple:
    a = np.asarray(a)
    b = np.asarray(b)
    n = len(a)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[i] = _a12(a[idx], b[idx])
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def _pairwise(a_vals, b_vals, label_a: str, label_b: str, window: str) -> dict:
    a = np.asarray(a_vals)
    b = np.asarray(b_vals)
    try:
        p = float(wilcoxon(a, b, alternative="two-sided").pvalue)
    except ValueError:
        p = float("nan")
    a12 = _a12(a, b)
    rng = np.random.default_rng(0)
    lo, hi = _a12_ci(a, b, rng)
    diff = a - b
    return {
        "a": label_a,
        "b": label_b,
        "window": window,
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "p_two_sided": p,
        "a12": a12,
        "a12_ci95": [lo, hi],
        "seeds_a_better": int(np.sum(diff < 0)),
        "seeds_b_better": int(np.sum(diff > 0)),
        "seeds_tie": int(np.sum(diff == 0)),
    }


def _machine() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": mp.cpu_count(),
    }


def main(seeds: int, streams: tuple, workers: int, out_path: Path) -> None:
    print("machine:", _machine(), flush=True)
    tasks = [(s, k) for k in streams for s in range(seeds)]
    n_tasks = len(tasks)
    print(f"tasks: {n_tasks}", flush=True)
    t0 = time.perf_counter()
    results = []
    if workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_run_one, tasks), 1):
                results.append(res)
                el = time.perf_counter() - t0
                print(f"  [{i:3d}/{n_tasks}]  seed={res[0]:2d} "
                      f"kind={res[1]:11s}  elapsed={el:6.1f}s", flush=True)
    else:
        for i, t in enumerate(tasks, 1):
            res = _run_one(t)
            results.append(res)
            el = time.perf_counter() - t0
            print(f"  [{i:3d}/{n_tasks}]  seed={res[0]:2d} "
                  f"kind={res[1]:11s}  elapsed={el:6.1f}s", flush=True)
    total_s = time.perf_counter() - t0
    print(f"\ntotal: {total_s:.1f}s = {total_s/60:.1f} min")

    per = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    ms = defaultdict(lambda: defaultdict(list))
    for seed, kind, res in results:
        for name, r in res.items():
            for w, v in r["windows"].items():
                per[name][w][kind].append(v)
            ms[name][kind].append(r["ms_per_sample"])

    print()
    header = (f"{'learner':10s} {'stream':12s} {'before':>9s} "
              f"{'IQR-bf':>9s} {'first100':>10s} {'IQR-f1':>9s} "
              f"{'after+2':>9s} {'ms/sample':>10s}")
    print(header)
    print("-" * len(header))
    for name in LEARNER_NAMES:
        for kind in streams:
            b = per[name]["before"][kind]
            f = per[name]["first100"][kind]
            a = per[name]["after+200"][kind]
            m = ms[name][kind]
            print(f"{name:10s} {kind:12s} "
                  f"{np.median(b):9.4f} "
                  f"{np.percentile(b, 75)-np.percentile(b, 25):9.4f} "
                  f"{np.median(f):10.4f} "
                  f"{np.percentile(f, 75)-np.percentile(f, 25):9.4f} "
                  f"{np.median(a):9.4f} {np.median(m):10.2f}")

    tests = []
    print()
    print("=" * 100)
    print("PAIRWISE, first 100 samples after the change")
    print("=" * 100)
    for kind in streams:
        for a_name, b_name in PAIRS:
            if a_name not in per or b_name not in per:
                continue
            r = _pairwise(
                per[a_name]["first100"][kind],
                per[b_name]["first100"][kind],
                a_name, b_name, kind,
            )
            tests.append(r)
            n = r["seeds_a_better"] + r["seeds_b_better"] + r["seeds_tie"]
            print(
                f"  {a_name:8s} vs {b_name:8s}  {kind:12s}  "
                f"medA={r['median_a']:.4f}  medB={r['median_b']:.4f}  "
                f"p={r['p_two_sided']:.4f}  "
                f"A12={r['a12']:.3f} "
                f"[{r['a12_ci95'][0]:.3f},{r['a12_ci95'][1]:.3f}]  "
                f"A-better {r['seeds_a_better']}/{n}"
            )

    payload = {
        "machine": _machine(),
        "seeds": seeds,
        "streams": list(streams),
        "windows": {k: list(v) for k, v in WINDOWS.items()},
        "raw": {
            name: {
                w: {kind: vals for kind, vals in per[name][w].items()}
                for w in WINDOWS
            }
            for name in per
        },
        "ms_per_sample": {
            name: {kind: vals for kind, vals in ms[name].items()}
            for name in ms
        },
        "pairwise_first100": tests,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print()
    print(f"JSON: {out_path}  ({out_path.stat().st_size} B)")


if __name__ == "__main__":
    mp.freeze_support()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--streams", nargs="+", default=list(STREAMS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="benchmarks/bricks_quality.json")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.seeds = 1
        args.streams = ["sudden"]
        args.workers = 1
        args.out = "benchmarks/bricks_quality_quick.json"
    main(args.seeds, tuple(args.streams), args.workers, Path(args.out))
