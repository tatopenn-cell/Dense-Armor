"""Time of the block path of the scikit-learn bridge against the row loop.

Claim, declared before the run and not changed after it: the block path is
not slower than the loop of ``learn_one`` on a table of 100,000 rows and 3
features, median of 7 runs. A claim that does not hold stays in the JSON as
it is.

Usage: ``python benchmarks/bridge_sklearn_blocks.py [--smoke] [--out PATH]``.
``--smoke`` uses 5,000 rows and 3 runs.
"""

import argparse
import json
import platform
import time

import numpy as np

from dense_armor.bridges.sklearn import SklearnTransformer
from dense_armor.utility.preprocessing.scale import StandardScaler

CLAIM = "the block path is not slower than the row loop, median of 7 runs"


class _LoopOnly(StandardScaler):
    learn_many_columns = False


def _time(factory, X, runs):
    out = []
    for _ in range(runs):
        est = SklearnTransformer(factory)
        t0 = time.perf_counter()
        est.fit(X)
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def _summary(ms):
    q1, med, q3 = np.percentile(ms, [25, 50, 75])
    return {"median_ms": float(med), "iqr_ms": float(q3 - q1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    n, runs = (5000, 3) if args.smoke else (100000, 7)
    X = np.random.default_rng(0).normal([1e3, 5.0, -2.0], [1.0, 2.0, 0.5], size=(n, 3))
    loop = _summary(_time(_LoopOnly, X, runs))
    block = _summary(_time(StandardScaler, X, runs))
    out = {
        "backend": "cpu",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "rows": n,
        "features": 3,
        "runs": runs,
        "claim": CLAIM,
        "loop": loop,
        "block": block,
        "speedup": loop["median_ms"] / block["median_ms"],
        "claim_holds": block["median_ms"] <= loop["median_ms"],
    }
    text = json.dumps(out, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
