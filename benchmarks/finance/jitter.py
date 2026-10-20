"""Jitter: how steady is the time per sample?

For each finance pipeline this script measures the time per sample over a
stated number of samples after a warm-up, and reports the median, the
interquartile range, the 99th percentile, the maximum, and the ratio of
the 99th percentile to the median. Never the bare mean.

The pipelines are the nine existing ones of ``pipelines/finance/`` and the
new ones of items A-D. The host (CPU, Python version) is printed and
written in the JSON: the numbers are the numbers of that host.

Run from the repository root:

    python benchmarks/finance/jitter.py
    python benchmarks/finance/jitter.py --smoke
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.finance.connectors import BoundedStream
from pipelines.finance.drift_vs_noise import (
    DriftVsNoisePipeline,
    synthetic_tick,
)
from pipelines.finance.execution import ExecutionPipeline
from pipelines.finance.order_book import order_flow_imbalance, synthetic_book
from pipelines.finance.portfolio import min_variance_weights
from pipelines.finance.regime import RegimePipeline, synthetic_two_regime
from pipelines.finance.stress import synthetic_flash_crash
from pipelines.finance.surveillance import FinancePipeline

FIXTURES = ROOT / "test" / "fixtures" / "finance"
N_SAMPLES = 500
WARMUP = 20
SMOKE_SAMPLES = 50
SMOKE_WARMUP = 5


def _stats(ts):
    a = np.asarray(ts, dtype=float)
    med = float(np.median(a))
    p99 = float(np.percentile(a, 99))
    return {"n": int(a.size), "median_ms": med * 1e3,
            "iqr_ms": float(np.percentile(a, 75) - np.percentile(a, 25)) * 1e3,
            "p99_ms": p99 * 1e3, "max_ms": float(a.max()) * 1e3,
            "p99_over_median": p99 / med if med > 0 else float("nan")}


def _surveillance(n, warmup):
    p = FinancePipeline(range_ret=(-0.01, 0.01), range_vol=(0.5, 1.5))
    prices = 100.0 * (1.0 + 0.001 * np.cumsum(np.random.default_rng(0).normal(size=n + warmup)))
    ts = []
    for i, price in enumerate(prices):
        t0 = time.perf_counter()
        p.process_tick(float(price), 1.0)
        dt = time.perf_counter() - t0
        if i >= warmup:
            ts.append(dt)
    return _stats(ts)


def _execution(n, warmup):
    p = ExecutionPipeline(warmup=10)
    rng = np.random.default_rng(0)
    sizes = np.exp(rng.normal(0.0, 0.2, n + warmup))
    ts = []
    for i, s in enumerate(sizes):
        t0 = time.perf_counter()
        p.process_order(float(s), 0.0)
        dt = time.perf_counter() - t0
        if i >= warmup:
            ts.append(dt)
    return _stats(ts)


def _regime(n, warmup):
    x, _ = synthetic_two_regime(n=n + warmup, seed=0, change_at=(n + warmup) // 2)
    pipe = RegimePipeline()
    ts = []
    for i, v in enumerate(x):
        t0 = time.perf_counter()
        pipe.run(np.array([v]))
        dt = time.perf_counter() - t0
        if i >= warmup:
            ts.append(dt)
    return _stats(ts)


def _drift_vs_noise(n, warmup):
    _, tick = synthetic_tick(n=n + warmup, seed=0, noise_std=0.005,
                             vol=0.001, noise_kind="student", noise_df=3.0)
    pipe = DriftVsNoisePipeline()
    ts = []
    for i in range(warmup, len(tick)):
        t0 = time.perf_counter()
        pipe.run(tick[i - warmup:i + 1])
        dt = time.perf_counter() - t0
        ts.append(dt)
    return _stats(ts)


def _portfolio(n, warmup):
    rng = np.random.default_rng(0)
    data = rng.normal(size=(n + warmup, 4))
    cov = np.cov(data, rowvar=False, ddof=1)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        min_variance_weights(cov)
        dt = time.perf_counter() - t0
        ts.append(dt)
    return _stats(ts)


def _order_book(n, warmup):
    book, _ = synthetic_book(n=n + warmup, seed=0)
    ts = []
    for i in range(warmup, len(book)):
        t0 = time.perf_counter()
        order_flow_imbalance(book[i - warmup:i + 1])
        dt = time.perf_counter() - t0
        ts.append(dt)
    return _stats(ts)


def _stress(n, warmup):
    crash = synthetic_flash_crash(n=n + warmup, seed=0, crash_at=(n + warmup) // 2)
    ts = []
    for i in range(warmup, len(crash.prices)):
        t0 = time.perf_counter()
        crash_metrics = __import__(
            "pipelines.finance.stress", fromlist=["crash_metrics"]).crash_metrics
        crash_metrics(crash)
        dt = time.perf_counter() - t0
        ts.append(dt)
    return _stats(ts)


def _connectors(n, warmup):
    q = BoundedStream(capacity=50, policy="drop_oldest")
    ts = []
    for i in range(n + warmup):
        t0 = time.perf_counter()
        q.push(i)
        q.pop()
        dt = time.perf_counter() - t0
        if i >= warmup:
            ts.append(dt)
    return _stats(ts)


PIPELINES = {
    "surveillance": _surveillance,
    "execution": _execution,
    "regime": _regime,
    "drift_vs_noise": _drift_vs_noise,
    "portfolio": _portfolio,
    "order_book": _order_book,
    "stress": _stress,
    "connectors": _connectors,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    n = SMOKE_SAMPLES if args.smoke else N_SAMPLES
    warmup = SMOKE_WARMUP if args.smoke else WARMUP
    host = {
        "cpu": platform.processor() or "unknown",
        "python": platform.python_version(),
        "machine": platform.machine(),
    }
    out = {"host": host, "n_samples": n, "warmup": warmup,
           "warmup_note": "surveillance and execution use the same n and warmup",
           "rows": {}}
    for name, fn in PIPELINES.items():
        t0 = time.time()
        out["rows"][name] = fn(n, warmup)
        out["rows"][name]["wall_time_s"] = time.time() - t0
        print(f"{name:18s} median {out['rows'][name]['median_ms']:.4f} ms "
              f"p99 {out['rows'][name]['p99_ms']:.4f} ms "
              f"p99/med {out['rows'][name]['p99_over_median']:.2f}")
    dest = args.out or (ROOT / "benchmarks" / "finance" / "jitter.json")
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"host: {host}")
    print(f"written: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
