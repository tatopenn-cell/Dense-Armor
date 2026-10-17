"""Quality benchmark for the Task 16 code.

Run from the repository root::

    python benchmarks/facto_quality.py --seeds 5 --n-samples 2000

Writes ``docs/assets/facto/figures.json`` with every number the docs
pages quote. The JSON holds the machine and the Python version at the
top, the exact command, the checks of the task, the time per sample of
every model, and the comparison table on ``DriftStream`` with a sudden
change.

Checks (they match the criteria of the task):
- the ``O(kn)`` form of the FM equals the ``O(kn^2)`` double sum, on
  random sparse inputs (Lemma 3.1 of Rendle 2010, page 3);
- the SDFT equals ``numpy.fft.fft`` on every window
  (Jacobsen and Lyons 2003, recursion of the tutorial);
- the sliding-window skyline equals the brute-force one on every
  window (Börzsönyi et al. 2001, Def. 1, page 2);
- the error on the planted low-rank matrix of the recommender
  decreases as the number of ratings grows;
- the comparison on ``DriftStream`` with one sudden change: the four
  classifiers of the task against ``LogisticRegression`` with the same
  optimizer, over ``--seeds`` seeds (median, not mean), with the time
  per sample.
"""

import argparse
import json
import multiprocessing as mp
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dense_armor.utility.datasets import DriftStream
from dense_armor.utility.facto import (
    FFMClassifier,
    FMClassifier,
    FMRegressor,
    FwFMClassifier,
    HOFMClassifier,
)
from dense_armor.utility.linear.glm import LogisticRegression
from dense_armor.utility.misc import SDFT, Skyline
from dense_armor.utility.optim import SGD
from dense_armor.utility.reco import BiasedMF


def _machine():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": mp.cpu_count(),
    }


def _time_per_sample(make, samples, fit, score):
    model = make()
    lat = np.empty(len(samples))
    for i, s in enumerate(samples):
        t0 = time.perf_counter()
        fit(model, s)
        score(model, s)
        lat[i] = time.perf_counter() - t0
    return float(np.median(lat) * 1e3), float(np.percentile(lat, 99) * 1e3)


def check_fm_ok_equals_double_sum(n_samples=200):
    rng = np.random.default_rng(0)
    fm = FMRegressor(optimizer=SGD(lr=0.05), k=3, seed=1)
    keys = [f"x{i}" for i in range(8)]
    fm._resolve_keys({k: 0.0 for k in keys})
    fm._ensure_state(fm._keys)
    max_diff = 0.0
    for _ in range(n_samples):
        x = {k: float(rng.normal()) for k in keys if rng.random() < 0.4}
        if not x:
            continue
        fast = fm._y_hat(x)
        slow = fm._pairwise_naive(x)
        max_diff = max(max_diff, abs(fast - slow))
    return {"max_abs_diff": max_diff, "n_samples": n_samples}


def check_sdft_equals_fft(n=8, n_samples=200):
    rng = np.random.default_rng(1)
    vals = rng.normal(size=n_samples)
    s = SDFT(n=n)
    max_diff = 0.0
    for i, v in enumerate(vals):
        _ = s.update(v)
        if i + 1 >= n:
            window = vals[i + 1 - n : i + 1]
            expected = np.fft.fft(window)
            d = float(np.max(np.abs(s.X - expected)))
            max_diff = max(max_diff, d)
    return {"max_abs_diff": max_diff, "n_samples": n_samples, "n": n}


def check_skyline_equals_bruteforce(n_points=50, window=10):
    rng = np.random.default_rng(2)
    pts = rng.normal(size=(n_points, 2))
    s = Skyline(window=window)
    all_equal = True
    for p in pts:
        _ = s.update({"x": float(p[0]), "y": float(p[1])})

        def brute(points):
            keep = []
            for i, q in enumerate(points):
                dominated = False
                for j, r in enumerate(points):
                    if i == j:
                        continue
                    if np.all(r <= q) and np.any(r < q):
                        dominated = True
                        break
                if not dominated:
                    keep.append(i)
            return keep

        got = sorted(s.skyline())
        exp = sorted(brute(list(s._buffer)))
        all_equal = all_equal and got == exp
    return {"all_equal": all_equal, "n_points": n_points, "window": window}


def check_recovery(n_users=20, n_items=15, k=3, epochs=60, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.normal(size=(n_users, k))
    Q = rng.normal(size=(n_items, k))
    mu = 3.5
    ratings = []
    for u in range(n_users):
        for i in range(n_items):
            r = mu + float(P[u] @ Q[i]) + float(rng.normal(0, 0.01))
            ratings.append((u, i, r))
    model = BiasedMF(k=k, lr=0.02, l2=0.001, seed=0)
    trace = []
    for ep in range(epochs):
        for u, i, r in ratings:
            _ = model.learn_one(u, i, r)
        errs = [abs(model.predict_one(u, i) - r) for u, i, r in ratings]
        trace.append((ep + 1, float(np.median(errs))))
    return {
        "n_users": n_users,
        "n_items": n_items,
        "k": k,
        "n_ratings": len(ratings),
        "final_median_err": trace[-1][1],
        "trace": trace,
    }


def check_comparison(seeds, n_samples, n_features=4):
    builders = {
        "LogisticRegression": lambda: LogisticRegression(
            optimizer=SGD(lr=0.01), intercept=True
        ),
        "FM": lambda: FMClassifier(optimizer=SGD(lr=0.05), k=4, seed=0),
        "FFM": lambda: FFMClassifier(
            fields={f"x{j}": f"F{j % 2}" for j in range(n_features)},
            k=2,
            lr=0.1,
            seed=0,
        ),
        "FwFM": lambda: FwFMClassifier(
            optimizer=SGD(lr=0.05),
            fields={f"x{j}": f"F{j % 2}" for j in range(n_features)},
            k=2,
            seed=0,
        ),
        "HOFM": lambda: HOFMClassifier(optimizer=SGD(lr=0.05), m=3, k=2, seed=0),
    }
    names = list(builders)
    accs = {n: [] for n in names}
    times = {n: [] for n in names}
    for seed in range(seeds):
        ds = DriftStream(
            n_samples=n_samples,
            n_features=n_features,
            n_drifts=1,
            kind="sudden",
            seed=seed,
        )
        X, y = ds.data()
        X = [{f"x{j}": float(v) for j, v in enumerate(row)} for row in X]
        for name, make in builders.items():
            model = make()
            lat = np.empty(len(X))
            correct = 0
            for i, (xi, yi) in enumerate(zip(X, y)):
                t0 = time.perf_counter()
                p = model.predict_one(xi)
                model.learn_one(xi, int(yi))
                lat[i] = time.perf_counter() - t0
                if p is not None and p == int(yi):
                    correct += 1
            accs[name].append(correct / len(X))
            times[name].append(float(np.median(lat) * 1e3))
    rows = []
    for n in names:
        rows.append(
            {
                "model": n,
                "acc_median": float(np.median(accs[n])),
                "acc_iqr": float(
                    np.percentile(accs[n], 75) - np.percentile(accs[n], 25)
                ),
                "ms_per_sample": float(np.median(times[n])),
            }
        )
    return {"n_seeds": seeds, "n_samples": n_samples, "rows": rows}


def check_costs(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    samples = [
        {f"x{j}": float(rng.normal()) for j in range(4) if rng.random() < 0.6}
        for _ in range(n)
    ]
    samples = [s for s in samples if s]
    rec = [
        (int(rng.integers(0, 20)), int(rng.integers(0, 15)), float(rng.normal(3, 0.5)))
        for _ in range(n)
    ]
    vals = [float(rng.normal()) for _ in range(n)]
    pts = [{"x": float(rng.normal()), "y": float(rng.normal())} for _ in range(n)]
    fields = {f"x{j}": f"F{j % 2}" for j in range(4)}
    rows = []

    def add(name, make, samples, fit, score):
        med, p99 = _time_per_sample(make, samples, fit, score)
        rows.append({"model": name, "median_ms": med, "p99_ms": p99})

    add(
        "FMClassifier",
        lambda: FMClassifier(optimizer=SGD(lr=0.05), k=4, seed=0),
        samples,
        lambda m, s: m.learn_one(s, 1),
        lambda m, s: m.predict_proba_one(s),
    )
    add(
        "FFMClassifier",
        lambda: FFMClassifier(fields=fields, k=2, lr=0.1, seed=0),
        samples,
        lambda m, s: m.learn_one(s, 1),
        lambda m, s: m.predict_proba_one(s),
    )
    add(
        "FwFMClassifier",
        lambda: FwFMClassifier(optimizer=SGD(lr=0.05), fields=fields, k=2, seed=0),
        samples,
        lambda m, s: m.learn_one(s, 1),
        lambda m, s: m.predict_proba_one(s),
    )
    add(
        "HOFMClassifier",
        lambda: HOFMClassifier(optimizer=SGD(lr=0.05), m=3, k=2, seed=0),
        samples,
        lambda m, s: m.learn_one(s, 1),
        lambda m, s: m.predict_proba_one(s),
    )
    add(
        "BiasedMF",
        lambda: BiasedMF(k=4, lr=0.01, seed=0),
        rec,
        lambda m, s: m.learn_one(s[0], s[1], s[2]),
        lambda m, s: m.predict_one(s[0], s[1]),
    )
    add(
        "SDFT",
        lambda: SDFT(n=16),
        vals,
        lambda m, s: m.update(s),
        lambda m, s: m.spectrum(),
    )
    add(
        "Skyline",
        lambda: Skyline(window=10),
        pts,
        lambda m, s: m.update(s),
        lambda m, s: m.skyline(),
    )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--out", default="docs/assets/facto/figures.json")
    args = ap.parse_args()
    command = (
        f"python benchmarks/facto_quality.py --seeds {args.seeds} "
        f"--n-samples {args.n_samples}"
    )
    print("machine:", _machine(), flush=True)
    print("command:", command, flush=True)

    print("check fm ok == double sum ...", flush=True)
    fm_check = check_fm_ok_equals_double_sum()
    print(f"  max_abs_diff = {fm_check['max_abs_diff']:.3e}", flush=True)

    print("check sdft == numpy.fft ...", flush=True)
    sdft_check = check_sdft_equals_fft()
    print(f"  max_abs_diff = {sdft_check['max_abs_diff']:.3e}", flush=True)

    print("check skyline == brute force ...", flush=True)
    sky_check = check_skyline_equals_bruteforce()
    print(f"  all_equal = {sky_check['all_equal']}", flush=True)

    print("check recovery on planted low-rank matrix ...", flush=True)
    rec_check = check_recovery()
    print(f"  final median err = {rec_check['final_median_err']:.4f}", flush=True)

    print("costs ...", flush=True)
    costs = check_costs()
    for r in costs:
        print(
            f"  {r['model']:20s} median={r['median_ms']:.4f} ms "
            f"p99={r['p99_ms']:.4f} ms",
            flush=True,
        )

    print(f"comparison, {args.seeds} seeds ...", flush=True)
    comp = check_comparison(args.seeds, args.n_samples)
    for r in comp["rows"]:
        print(
            f"  {r['model']:20s} acc_median={r['acc_median']:.4f} "
            f"ms/sample={r['ms_per_sample']:.4f}",
            flush=True,
        )

    payload = {
        "machine": _machine(),
        "command": command,
        "checks": {
            "fm_ok_equals_double_sum": fm_check,
            "sdft_equals_numpy_fft": sdft_check,
            "skyline_equals_bruteforce": sky_check,
        },
        "recovery": rec_check,
        "costs": costs,
        "comparison": comp,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nJSON: {out}  ({out.stat().st_size} B)", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
