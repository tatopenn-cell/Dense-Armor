"""Energy benchmark: the batch CART tree against the online Hoeffding tree.

For each seed the same stream, the same 50/50 split and the same test
half of ``pipelines/drones/static_vs_online.py`` are used: the batch tree
fits on the training half and predicts the test half; the online tree
predicts and learns on the test half, in the same order. Around each
model a power source is started and stopped; the energy is the integral
of the power over the wall time of the run, and the idle power measured
before the runs is subtracted. The two runs of a seed are paired: the
same stream, the same test half, only the model changes.

The comparison is on the net energy per 1000 samples, with a paired
Wilcoxon two-sided p-value and the A12 effect size. If the number of
pairs is less than six, the JSON says so and the p-value is ``null``.

The power source is one of:

- a sysfs node (``--power-node PATH``): a Jetson exposes the whole-board
  power through the INA3221 sensors as a text file, one integer in mW
  per read (2306.13107, page 2). The path is a parameter without a
  default.
- a synthetic constant power (``--smoke``): the trace is labelled
  ``"power": "synthetic"`` in the JSON and on the page.
- none (neither flag): the energy fields are ``null`` and the CPU time
  and the wall time are still written. The JSON says ``"power": null``.

Run from the repository root:

    python -m edge.energy_bench --smoke
    python -m edge.energy_bench --power-node /sys/devices/.../power1_input

The ``--smoke`` run uses 3 seeds and 600 samples per stream, a synthetic
power source at 5000 mW, and finishes in under one minute on x86. The
full run uses 10 seeds and 2000 samples per stream.

The JSON goes to ``benchmarks/drones/energy_bench_<host>.json``.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from dense_armor.utility.tree.batch import BatchDecisionTreeClassifier
from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier
from edge.power import (
    PowerTrace,
    SyntheticPowerSource,
    SysfsPowerSource,
    energy_j,
    net_energy_j,
)
from pipelines.drones.io import synthetic_flight
from pipelines.drones.static_vs_online import _features

N_FEATURES = 16
DEFAULT_MAX_DEPTH = 6
SMOKE_MW = 5000.0
SMOKE_SEEDS = 3
SMOKE_SAMPLES = 600
FULL_SEEDS = 10
FULL_SAMPLES = 2000
MIN_PAIRS_FOR_WILCOXON = 6


def _a12(a: np.ndarray, b: np.ndarray) -> float:
    diff = a[:, None] - b[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def _a12_ci(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n: int = 2000
) -> tuple[float, float]:
    k = len(a)
    vals = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, k, size=k)
        vals[i] = _a12(a[idx], b[idx])
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def _wilcoxon(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < MIN_PAIRS_FOR_WILCOXON:
        return None
    if bool(np.all(a == b)):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(wilcoxon(a, b).pvalue)


def _wilcoxon_note(p: float | None, n_pairs: int) -> str:
    if p is not None:
        return "computed on the net joules per 1000 samples, paired by seed"
    if n_pairs < MIN_PAIRS_FOR_WILCOXON:
        return f"not computed, fewer than {MIN_PAIRS_FOR_WILCOXON} pairs"
    return "not computed, all paired differences are zero"


def _board_model() -> str | None:
    p = Path("/proc/device-tree/model")
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8", errors="replace").strip("\x00 \n")
    except OSError:
        return None


def _idle_mw(source_factory, idle_s: float) -> tuple[float, str]:
    if idle_s <= 0.0:
        return 0.0, "not measured (idle_s = 0)"
    src = source_factory()
    src.start()
    time.sleep(idle_s)
    trace: PowerTrace = src.stop()
    return float(np.mean(trace.mw)), f"measured for {idle_s:.2f} s"


def _measure(fn, source) -> tuple[float, float, PowerTrace | None]:
    cpu0 = time.process_time()
    wall0 = time.perf_counter()
    src = source() if source is not None else None
    if src is not None:
        src.start()
    fn()
    trace = src.stop() if src is not None else None
    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0
    return wall, cpu, trace


def _energy_per_1000(trace: PowerTrace, idle_mw: float, n_samples: int) -> float:
    e = net_energy_j(trace, idle_mw)
    return 1000.0 * e / float(n_samples)


@dataclass
class _Run:
    wall_s: float
    cpu_s: float
    energy_j: float
    j_per_1000: float
    n_samples: int


def _run_pair(
    seed: int,
    n_samples: int,
    max_depth: int,
    source_factory,
    idle_mw: float,
) -> tuple[_Run, _Run, str]:
    stream = list(synthetic_flight(n=n_samples, seed=seed, fault="actuator_loss"))
    X = [_features(s) for s in stream]
    y = [int(s["fault_started"]) for s in stream]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    n_train = n_samples // 2
    tr_idx = perm[:n_train]
    te_idx = perm[n_train:]
    X_tr = [X[i] for i in tr_idx]
    y_tr = [y[i] for i in tr_idx]
    X_te = [X[i] for i in te_idx]
    y_te = [y[i] for i in te_idx]

    clf_b = BatchDecisionTreeClassifier(max_depth=max_depth)

    def do_batch() -> None:
        clf_b.fit(X_tr, y_tr)
        for x in X_te:
            clf_b.predict_one(x)

    w_b, c_b, tr_b = _measure(do_batch, source_factory)
    e_b = energy_j(tr_b) if tr_b is not None else float("nan")
    j_b = (
        _energy_per_1000(tr_b, idle_mw, len(X_te))
        if tr_b is not None
        else float("nan")
    )

    clf_o = HoeffdingTreeClassifier(grace_period=50)

    def do_online() -> None:
        for x, t in zip(X_te, y_te):
            clf_o.predict_one(x)
            clf_o.learn_one(x, t)

    w_o, c_o, tr_o = _measure(do_online, source_factory)
    e_o = energy_j(tr_o) if tr_o is not None else float("nan")
    j_o = (
        _energy_per_1000(tr_o, idle_mw, len(X_te))
        if tr_o is not None
        else float("nan")
    )

    rb = _Run(wall_s=w_b, cpu_s=c_b, energy_j=e_b, j_per_1000=j_b, n_samples=len(X_te))
    ro = _Run(wall_s=w_o, cpu_s=c_o, energy_j=e_o, j_per_1000=j_o, n_samples=len(X_te))
    return rb, ro, ""


def _summarise(runs: list[_Run], name: str) -> dict:
    wall = np.array([r.wall_s for r in runs])
    cpu = np.array([r.cpu_s for r in runs])
    energy = np.array([r.energy_j for r in runs])
    j1000 = np.array([r.j_per_1000 for r in runs])

    def med_iqr(x: np.ndarray) -> tuple[float, float]:
        if np.all(np.isnan(x)):
            return float("nan"), float("nan")
        return float(np.median(x)), float(np.quantile(x, 0.75) - np.quantile(x, 0.25))

    wm, wi = med_iqr(wall)
    cm, ci = med_iqr(cpu)
    em, ei = med_iqr(energy)
    jm, ji = med_iqr(j1000)
    return {
        "name": name,
        "wall_s_median": wm,
        "wall_s_iqr": wi,
        "cpu_s_median": cm,
        "cpu_s_iqr": ci,
        "energy_j_median": em,
        "energy_j_iqr": ei,
        "j_per_1000_median": jm,
        "j_per_1000_iqr": ji,
    }


def _make_source_factory(args) -> tuple[object, str | None]:
    if args.smoke and args.power_node:
        raise SystemExit("--smoke and --power-node are mutually exclusive")
    if args.smoke:
        def factory():
            return SyntheticPowerSource(mw=SMOKE_MW, dt_s=args.dt_s)

        return factory, "synthetic"
    if args.power_node:
        def factory():
            return SysfsPowerSource(args.power_node, dt_s=args.dt_s)

        return factory, "sysfs"
    return None, None


def run(args) -> dict:
    source_factory, power_kind = _make_source_factory(args)
    n_seeds = SMOKE_SEEDS if args.smoke else FULL_SEEDS
    n_samples = SMOKE_SAMPLES if args.smoke else FULL_SAMPLES
    idle_s = 0.0 if args.smoke else args.idle_s

    idle_mw = 0.0
    idle_note = "not measured (no power source)"
    if source_factory is not None:
        idle_mw, idle_note = _idle_mw(source_factory, idle_s)

    runs_batch: list[_Run] = []
    runs_online: list[_Run] = []
    for seed in range(n_seeds):
        rb, ro, _ = _run_pair(
            seed=seed,
            n_samples=n_samples,
            max_depth=DEFAULT_MAX_DEPTH,
            source_factory=source_factory,
            idle_mw=idle_mw,
        )
        runs_batch.append(rb)
        runs_online.append(ro)
        print(
            f"  seed {seed:2d}: batch j/1000 = {rb.j_per_1000:.3f}, "
            f"online j/1000 = {ro.j_per_1000:.3f}",
            flush=True,
        )

    summary_batch = _summarise(runs_batch, "BatchDecisionTreeClassifier (CART)")
    summary_online = _summarise(runs_online, "HoeffdingTreeClassifier")

    j_b = np.array([r.j_per_1000 for r in runs_batch])
    j_o = np.array([r.j_per_1000 for r in runs_online])
    have_energy = not np.all(np.isnan(j_b)) and not np.all(np.isnan(j_o))
    if not have_energy:
        for s in (summary_batch, summary_online):
            s["energy_j_median"] = None
            s["energy_j_iqr"] = None
            s["j_per_1000_median"] = None
            s["j_per_1000_iqr"] = None

    if have_energy:
        p = _wilcoxon(j_b, j_o)
        a12 = _a12(j_b, j_o)
        ci = _a12_ci(j_b, j_o, np.random.default_rng(0))
        comparison = {
            "j_per_1000_delta_median": float(np.median(j_b - j_o)),
            "wilcoxon_p": p,
            "wilcoxon_note": _wilcoxon_note(p, len(j_b)),
            "a12_batch_over_online": a12,
            "a12_ci95": list(ci),
            "seeds_batch_better": int((j_b < j_o).sum()),
            "seeds_online_better": int((j_o < j_b).sum()),
            "seeds_tie": int((j_b == j_o).sum()),
        }
    else:
        comparison = {
            "j_per_1000_delta_median": None,
            "wilcoxon_p": None,
            "wilcoxon_note": "not computed, no power source",
            "a12_batch_over_online": None,
            "a12_ci95": None,
            "seeds_batch_better": None,
            "seeds_online_better": None,
            "seeds_tie": None,
        }

    return {
        "host": socket.gethostname(),
        "board": _board_model(),
        "power": power_kind,
        "power_node": args.power_node,
        "dt_s": float(args.dt_s),
        "idle_mw": float(idle_mw),
        "idle_s": float(idle_s),
        "idle_note": idle_note,
        "n_seeds": int(n_seeds),
        "n_samples": int(n_samples),
        "batch": summary_batch,
        "online": summary_online,
        "comparison": comparison,
    }


def _default_out() -> Path:
    host = socket.gethostname().replace("/", "_")
    return Path("benchmarks/drones") / f"energy_bench_{host}.json"


def main() -> int:
    parser = argparse.ArgumentParser(prog="edge.energy_bench")
    parser.add_argument("--smoke", action="store_true",
                        help="3 seeds, synthetic power, under one minute on x86")
    parser.add_argument("--power-node", type=str, default=None,
                        help="path of a sysfs node that returns mW")
    parser.add_argument("--dt-s", type=float, default=0.1)
    parser.add_argument("--idle-s", type=float, default=30.0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    payload = run(args)

    out = Path(args.out) if args.out else _default_out()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"json: {out}")

    print()
    print(f"host: {payload['host']}")
    print(f"board: {payload['board'] or 'not a Jetson'}")
    print(f"power: {payload['power'] or 'not measured'}")
    print(f"idle_mw: {payload['idle_mw']:.1f} ({payload['idle_note']})")
    print(f"n_seeds: {payload['n_seeds']}, n_samples: {payload['n_samples']}")
    print()
    header = (f"{'model':45s} {'wall_s':>10s} {'cpu_s':>10s} "
              f"{'J/1000':>10s}")
    print(header)
    print("-" * len(header))
    for m in (payload["batch"], payload["online"]):
        j = m["j_per_1000_median"]
        j_str = f"{j:10.3f}" if j is not None else f"{'null':>10s}"
        print(f"{m['name']:45s} "
              f"{m['wall_s_median']:10.3f} {m['cpu_s_median']:10.3f} "
              f"{j_str}")
    c = payload["comparison"]
    print()
    print(f"delta J/1000 (batch - online): {c['j_per_1000_delta_median']}")
    print(f"wilcoxon p: {c['wilcoxon_p']} ({c['wilcoxon_note']})")
    print(f"A12 (batch > online): {c['a12_batch_over_online']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
