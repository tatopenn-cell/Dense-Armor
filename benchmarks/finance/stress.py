"""Benchmark: the existing finance pipelines on a synthetic flash crash.

The crash is synthetic (Braun et al. 2017, arXiv:1707.05580, page 1 for
its shape); the two pipelines are the ones of the repository and are not
modified. The two have very different costs: ``FinancePipeline`` calls
Orca on every tick (121 ms/tick on this host), ``ExecutionPipeline``
costs 0.2 ms/order. The benchmark runs the two on separate budgets,
declared here:

- ``FinancePipeline``: 100 ticks and 5 seeds, because 2,000 ticks and 20
  seeds would be 80 minutes. With 100 ticks the crash of 5 ticks is
  visible; with 5 seeds the median is a small-sample median, and the
  page says so.
- ``ExecutionPipeline``: 2,000 ticks and 20 seeds, because it is fast.

The order sizes and the relative volumes are not constant: a constant
stream has nothing to detect. Each tick carries a size drawn around one
with a log-normal, and the crash multiplies the size by the drop, so the
orders that arrive during the crash are large.

Run from the repository root:

    python benchmarks/finance/stress.py
    python benchmarks/finance/stress.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.finance.execution import ExecutionPipeline
from pipelines.finance.stress import synthetic_flash_crash
from pipelines.finance.surveillance import FinancePipeline

N_TICKS = 2000
CRASH_AT = 1000
DURATION = 5
DROP = 0.05
RECOVERY = 0.6
SEEDS_SURV = 5
TICKS_SURV = 100
SEEDS_EXEC = 20
SMOKE_SEEDS = 2
SMOKE_TICKS_SURV = 50


def _stream(seed, n, crash_at):
    crash = synthetic_flash_crash(
        n=n, seed=seed, crash_at=crash_at, duration_ticks=DURATION,
        drop=DROP, recovery_fraction=RECOVERY, vol=0.001)
    rng = np.random.default_rng(seed + 10_000)
    size = np.exp(rng.normal(0.0, 0.2, n))
    volume = np.exp(rng.normal(0.0, 0.2, n))
    for i in range(crash_at, min(crash_at + DURATION, n)):
        size[i] *= 10.0
        volume[i] *= 10.0
    return crash.prices, size, volume


def _surv_run(seed, n, crash_at):
    prices, _, volume = _stream(seed, n, crash_at)
    surv = FinancePipeline(range_ret=(-0.01, 0.01), range_vol=(0.5, 1.5))
    alarms = []
    for i in range(n):
        r = surv.process_tick(float(prices[i]), float(volume[i]))
        if r.anomalous:
            alarms.append(i)
    first = next((a for a in alarms if a >= crash_at), None)
    return {"seed": seed, "n_alarms": len(alarms),
            "delay": None if first is None else int(first - crash_at)}


def _exec_run(seed, n, crash_at):
    _, size, _ = _stream(seed, n, crash_at)
    execu = ExecutionPipeline(warmup=20)
    refused = 0
    for i in range(n):
        o = execu.process_order(float(size[i]), 0.0)
        if not o.approved:
            refused += 1
    return {"seed": seed, "n_refused": refused}


def _summary(values):
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return {"median": None, "iqr": None, "n": 0}
    return {"median": float(np.median(v)),
            "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
            "n": int(v.size)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    n_surv = SMOKE_SEEDS if args.smoke else SEEDS_SURV
    t_surv = SMOKE_TICKS_SURV if args.smoke else TICKS_SURV
    n_exec = SMOKE_SEEDS if args.smoke else SEEDS_EXEC

    t0 = time.time()
    surv = [_surv_run(s, t_surv, t_surv // 2) for s in range(n_surv)]
    dt_surv = time.time() - t0

    t0 = time.time()
    execu = [_exec_run(s, N_TICKS, CRASH_AT) for s in range(n_exec)]
    dt_exec = time.time() - t0

    alarms = _summary([float(r["n_alarms"]) for r in surv])
    delays = _summary([r["delay"] for r in surv])
    refused = _summary([float(r["n_refused"]) for r in execu])
    payload = {
        "dataset": "synthetic flash crash (Braun et al. 2017, page 1)",
        "crash": {"crash_at": CRASH_AT, "duration_ticks": DURATION,
                  "drop": DROP, "recovery_fraction": RECOVERY},
        "surveillance": {
            "seeds": n_surv, "ticks": t_surv,
            "alarms": alarms, "delay": delays,
            "n_with_delay": sum(1 for r in surv if r["delay"] is not None),
            "wall_time_s": dt_surv,
        },
        "execution": {
            "seeds": n_exec, "ticks": N_TICKS,
            "refused": refused, "wall_time_s": dt_exec,
        },
        "per_seed": {"surveillance": surv, "execution": execu},
    }
    out = args.out or (ROOT / "benchmarks" / "finance" / "stress.json")
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"synthetic flash crash, surveillance {n_surv} seeds x {t_surv} ticks")
    print(f"  alarms: median {alarms['median']} IQR {alarms['iqr']}")
    print(f"  delay: median {delays['median']} IQR {delays['iqr']} "
          f"({payload['surveillance']['n_with_delay']}/{n_surv} with an alarm)")
    print(f"  wall {dt_surv:.1f} s")
    print(f"execution {n_exec} seeds x {N_TICKS} ticks")
    print(f"  refused: median {refused['median']} IQR {refused['iqr']}")
    print(f"  wall {dt_exec:.1f} s")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
