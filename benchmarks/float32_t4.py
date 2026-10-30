"""Benchmark of the float32 path against the default path.

Claim, declared before the run and not changed after it: the float32
path is at most as slow as the default path on every kept class, median
of 7 runs. A claim that does not hold stays in the delivery as it is,
with the numbers.

Classes: Gaussian, OnlineGaussianNB, SelectKBest, R2Score.

Reference of the correctness side is in test/utility/{proba,learn,
preprocessing,metrics}/*_float32.py, not here: this script only times.

Time is median and IQR of the per-update time in milliseconds.
The JSON is written next to the script and quoted in the delivery.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dense_armor.utility.learn.online_classifiers import OnlineGaussianNB
from dense_armor.utility.metrics.regression import R2Score
from dense_armor.utility.preprocessing.select import SelectKBest
from dense_armor.utility.proba import Gaussian

SEED = 0
N_RUNS = 7


def _time_ms(make, feed, n, n_runs=N_RUNS):
    ts = []
    for _ in range(n_runs):
        obj = make()
        t0 = time.perf_counter()
        feed(obj)
        ts.append((time.perf_counter() - t0) * 1e3 / n)
    ts = np.asarray(ts)
    return float(np.median(ts)), float(np.percentile(ts, 75) - np.percentile(ts, 25))


def _gauss(n, dtype):
    rng = np.random.default_rng(SEED)
    xs = 1e3 + 0.01 * rng.standard_normal(n)

    def feed(obj):
        for v in xs:
            obj.update(float(v))

    return _time_ms(lambda: Gaussian(dtype=dtype), feed, n)


def _nb(n, dtype):
    rng = np.random.default_rng(SEED)
    xs = 1e3 + 0.01 * rng.standard_normal(n)
    ys = rng.integers(0, 2, n)

    def feed(obj):
        for v, y in zip(xs, ys):
            obj.learn_one({"a": float(v)}, int(y))

    return _time_ms(lambda: OnlineGaussianNB(dtype=dtype), feed, n)


def _kbest(n, dtype):
    rng = np.random.default_rng(SEED)
    xs = 1e3 + 0.01 * rng.standard_normal(n)
    ys = 2.0 * (xs - 1e3) + 0.001 * rng.standard_normal(n)

    def feed(obj):
        for x, y in zip(xs, ys):
            obj.learn_one({"a": float(x)}, y=float(y))

    return _time_ms(lambda: SelectKBest(k=1, dtype=dtype), feed, n)


def _r2(n, dtype):
    rng = np.random.default_rng(SEED)
    ys = 1e3 + 0.01 * rng.standard_normal(n)
    yhat = ys + 0.001 * rng.standard_normal(n)

    def feed(obj):
        for a, b in zip(ys, yhat):
            obj.update(float(a), float(b))

    return _time_ms(lambda: R2Score(dtype=dtype), feed, n)


BENCHES = [
    ("Gaussian", _gauss),
    ("OnlineGaussianNB", _nb),
    ("SelectKBest", _kbest),
    ("R2Score", _r2),
]


def main(smoke):
    n = 2_000 if smoke else 20_000
    print(f"backend: cpu / python {platform.python_version()} / {platform.platform()}")
    print(f"N per run: {n}, seed: {SEED}, runs: {N_RUNS}")
    print(f"{'class':20s} {'path':10s} {'median ms':>12s} {'IQR ms':>12s}")
    rows = []
    for name, fn in BENCHES:
        m64, i64 = fn(n, None)
        m32, i32 = fn(n, np.float32)
        print(f"{name:20s} {'default':10s} {m64:12.6f} {i64:12.6f}")
        print(f"{name:20s} {'f32':10s} {m32:12.6f} {i32:12.6f}")
        rows.append(
            {
                "class": name,
                "default_median_ms": m64,
                "default_iqr_ms": i64,
                "f32_median_ms": m32,
                "f32_iqr_ms": i32,
                "claim_holds": m32 <= m64,
            }
        )

    out = Path(__file__).with_suffix(".json")
    out.write_text(
        json.dumps(
            {
                "backend": "cpu",
                "platform": platform.platform(),
                "processor": platform.processor(),
                "n": n,
                "seed": SEED,
                "runs": N_RUNS,
                "claim": "the float32 path is at most as slow as the default path on every kept class, median of 7 runs",
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"JSON: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    main(args.smoke)
