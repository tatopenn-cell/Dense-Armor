"""Edge benchmark: each pipeline on synthetic data, N samples, on this host.

The six pipelines of the drones section are run on synthetic data, one
at a time, with a fixed number of samples per run, and the time per
sample is measured M times. The page reports the median, the
interquartile range and the 99th percentile of the time per sample,
the resident memory before and after the run, and (if the host exposes
it) the CPU temperature. The host name and the CPU count are in the
JSON, so a table from a Raspberry Pi and a table from an NVIDIA Jetson
are comparable: same pipelines, same data, same measurement.

The measurement is not the time of a single learn_one. The pipelines
take a whole list and return after processing it; there is no per-sample
hook. The time per sample is measured as the total time of one run
divided by the number of samples in that run, repeated M times on
disjoint slices of the stream, and the distribution is the distribution
of those M values. This is what the spec asks (median, IQR, p99, not
the bare mean); it is not the cost of a single learn_one, and the page
says so in Details.

Run from the repository root:

    python edge/edge_bench.py --smoke
    python edge/edge_bench.py

The --smoke flag uses 200 samples per run and 5 runs, and finishes in
under one minute on x86. Without the flag it uses 2000 samples and 20
runs.

The JSON is written to benchmarks/drones/edge_bench_<host>.json.

No board is available in this session. The tables produced here are
for the x86 host of Colab; the same command, on a Raspberry Pi or a
Jetson, writes the file with the board name in it. The page states
this: "written, not run on the board".
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import psutil

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipelines.drones.actuator_fault import run as run_actuator
from pipelines.drones.battery_endurance import run as run_battery
from pipelines.drones.io import synthetic_flight
from pipelines.drones.link_health import run as run_link
from pipelines.drones.link_health import synthetic_mqtt
from pipelines.drones.propeller_health import run as run_propeller
from pipelines.drones.safety_filter import run as run_safety
from pipelines.drones.sensor_consistency import run as run_sensor

PIPELINES = {
    "actuator_fault": (run_actuator, "stream"),
    "sensor_consistency": (run_sensor, "stream"),
    "propeller_health": (run_propeller, "stream"),
    "battery_endurance": (run_battery, "stream"),
    "safety_filter": (run_safety, "scans"),
    "link_health": (run_link, "packets"),
}


def _make_data(name: str, n: int, seed: int):
    """Build n input samples for the named pipeline."""
    kind = PIPELINES[name][1]
    if kind == "stream":
        return list(synthetic_flight(n=n, seed=seed, fault=None))
    if kind == "scans":
        import benchmarks.lidar_sim as ls
        rng = np.random.default_rng(seed)
        return [ls.scan(i, rng, None, None) for i in range(n)]
    if kind == "packets":
        return synthetic_mqtt(n=n, seed=seed)
    raise ValueError(f"unknown kind {kind!r}")


def _time_one(run_fn, data) -> tuple[float, float]:
    """Run the pipeline once. Returns (ms per sample, delta RSS KB)."""
    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss
    t0 = time.perf_counter()
    _ = run_fn(data)
    dt = time.perf_counter() - t0
    rss_after = process.memory_info().rss
    n = max(len(data), 1)
    return dt * 1000.0 / n, (rss_after - rss_before) / 1024.0


def _cpu_temp_c() -> float | None:
    """CPU temperature in Celsius, or None if the host does not expose it."""
    try:
        temps = psutil.sensors_temperatures()
    except (AttributeError, OSError):
        return None
    if not temps:
        return None
    for entries in temps.values():
        if entries:
            return float(entries[0].current)
    return None


def _bench_pipeline(
    name: str, n_samples: int, n_runs: int, seed: int,
) -> dict:
    """Benchmark one pipeline on ``n_runs`` disjoint slices."""
    run_fn = PIPELINES[name][0]
    per_run_ms = np.empty(n_runs)
    per_run_drss = np.empty(n_runs)
    for m in range(n_runs):
        data = _make_data(name, n_samples, seed + m)
        ms, drss = _time_one(run_fn, data)
        per_run_ms[m] = ms
        per_run_drss[m] = drss
    q1: float = float(np.percentile(per_run_ms, 25))
    q3: float = float(np.percentile(per_run_ms, 75))
    return {
        "name": name,
        "n_samples": n_samples,
        "n_runs": n_runs,
        "ms_per_sample_median": float(np.median(per_run_ms)),
        "ms_per_sample_iqr": float(q3 - q1),
        "ms_per_sample_p99": float(np.percentile(per_run_ms, 99)),
        "rss_delta_kb_median": float(np.median(per_run_drss)),
    }


def _host() -> dict:
    return {
        "hostname": socket.gethostname(),
        "cpu_count": psutil.cpu_count(logical=True),
        "cpu_freq_mhz": (
            psutil.cpu_freq().current if psutil.cpu_freq() else None
        ),
        "cpu_temp_c": _cpu_temp_c(),
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="200 samples per run, 5 runs")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    n_samples = args.samples or (200 if args.smoke else 2000)
    n_runs = args.runs or (5 if args.smoke else 20)
    host = _host()
    out = args.out or (
        Path("benchmarks/drones") / f"edge_bench_{host['hostname']}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"host: {host['hostname']}, {host['cpu_count']} cpus, "
          f"temp={host['cpu_temp_c']}")
    print(f"{'pipeline':22s} {'median':>9s} {'iqr':>7s} "
          f"{'p99':>7s} {'dRSS KB':>10s}")
    print("-" * 60)
    results = []
    t0 = time.perf_counter()
    for name in PIPELINES:
        row = _bench_pipeline(name, n_samples, n_runs, args.seed)
        results.append(row)
        print(f"{name:22s} {row['ms_per_sample_median']:9.3f} "
              f"{row['ms_per_sample_iqr']:7.3f} "
              f"{row['ms_per_sample_p99']:7.3f} "
              f"{row['rss_delta_kb_median']:10.1f}", flush=True)
    total = time.perf_counter() - t0
    print(f"\ntotal wall time: {total:.1f} s")
    payload = {
        "host": host,
        "smoke": args.smoke,
        "n_samples": n_samples,
        "n_runs": n_runs,
        "total_s": total,
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
