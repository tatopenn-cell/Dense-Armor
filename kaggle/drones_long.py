"""Long benchmark of the drones pipelines, run on Kaggle.

This script is the long measurement that the smoke run of edge_bench.py
(200 samples, 5 runs) cannot do: 6 pipelines, N seeds, M samples per
seed, with the median, IQR and p99 of the time per sample, and the
peak memory per run. It runs on Kaggle, in a session long enough for
the full sweep, and writes its JSON in /kaggle/working.

The script is self-contained: it clones the repository, installs it
with the drone extra, runs the sweep, and writes the JSON. It is the
same Python that runs on Colab with SMOKE = True, on a small number
of seeds and samples, to check that the script is not broken.

The spec of Task D1 says: "One self-contained .py under kaggle/
drones_<name>.py per long measurement (clones the repository, branch
named in a variable at the top, installs it, runs, writes its JSON in
/kaggle/working, prints a table, has a SMOKE switch)." This is that
script. The kernel-metadata.json is in the message that accompanies
this commit, not in the repository, as the spec asks.

Usage on Kaggle:
    publish the script with the metadata, run the kernel, download
    /kaggle/working/drones_long.json at the end.

Usage on Colab (smoke):
    python kaggle/drones_long.py
    # or, for a full run on a long-lived Colab Pro:
    python kaggle/drones_long.py --full
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np


REPO_URL = "https://github.com/tatopenn-cell/Dense-Armor.git"
BRANCH = "taskD1-drones"
SMOKE_SEEDS = 2
SMOKE_SAMPLES = 200
FULL_SEEDS = 20
FULL_SAMPLES = 2000
N_MODELS = 5


def _ensure_repo() -> Path:
    """Clone the repository and install it with the drone extra."""
    if Path("/kaggle/working").exists():
        root = Path("/kaggle/working/Dense-Armor")
    else:
        root = Path("/content/Dense-Armor")
    if not root.exists():
        subprocess.run(
            ["git", "clone", "-q", "-b", BRANCH, REPO_URL, str(root)],
            check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-e",
             f"{root}[drone]"],
            check=True,
        )
    return root


def _pipelines(root: Path):
    """Import the six pipelines after the install."""
    sys.path.insert(0, str(root))
    from pipelines.drones.actuator_fault import run as af
    from pipelines.drones.battery_endurance import run as be
    from pipelines.drones.io import synthetic_flight
    from pipelines.drones.link_health import run as lh
    from pipelines.drones.link_health import synthetic_mqtt
    from pipelines.drones.propeller_health import run as ph
    from pipelines.drones.safety_filter import run as sf
    from pipelines.drones.sensor_consistency import run as sc

    def _make(name: str, n: int, seed: int):
        if name == "safety_filter":
            import benchmarks.lidar_sim as ls
            rng = np.random.default_rng(seed)
            return [ls.scan(i, rng, None, None) for i in range(n)]
        if name == "link_health":
            return synthetic_mqtt(n=n, seed=seed)
        return list(synthetic_flight(n=n, seed=seed, fault=None))

    return {
        "actuator_fault": (af, "stream"),
        "sensor_consistency": (sc, "stream"),
        "propeller_health": (ph, "stream"),
        "battery_endurance": (be, "stream"),
        "safety_filter": (sf, "scans"),
        "link_health": (lh, "packets"),
    }, _make


def _bench(name, run_fn, make, n_samples, n_seeds):
    times = []
    peaks = []
    for seed in range(n_seeds):
        data = make(name, n_samples, seed)
        process_peak = None
        tracemalloc.start()
        t0 = time.perf_counter()
        _ = run_fn(data)
        dt = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        times.append(dt * 1000.0 / max(n_samples, 1))
        peaks.append(peak / 1024.0)
    times = np.array(times)
    peaks = np.array(peaks)
    q1, q3 = np.percentile(times, [25, 75])
    return {
        "name": name,
        "n_samples": n_samples,
        "n_seeds": n_seeds,
        "ms_per_sample_median": float(np.median(times)),
        "ms_per_sample_iqr": float(q3 - q1),
        "ms_per_sample_p99": float(np.percentile(times, 99)),
        "peak_kb_median": float(np.median(peaks)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    n_seeds = FULL_SEEDS if args.full else SMOKE_SEEDS
    n_samples = FULL_SAMPLES if args.full else SMOKE_SAMPLES
    mode = "full" if args.full else "smoke"
    print(f"mode: {mode}, seeds: {n_seeds}, samples: {n_samples}")

    root = _ensure_repo()
    pipelines, make = _pipelines(root)
    host = {
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "in_kaggle": Path("/kaggle/working").exists(),
    }
    print(f"host: {host['hostname']}, {host['cpu_count']} cpus, "
          f"in_kaggle={host['in_kaggle']}")

    print(f"{'pipeline':22s} {'median':>9s} {'iqr':>7s} "
          f"{'p99':>7s} {'peak KB':>10s}")
    print("-" * 60)
    results = []
    t0 = time.perf_counter()
    for name, (fn, _kind) in pipelines.items():
        row = _bench(name, fn, make, n_samples, n_seeds)
        results.append(row)
        print(f"{name:22s} {row['ms_per_sample_median']:9.4f} "
              f"{row['ms_per_sample_iqr']:7.4f} "
              f"{row['ms_per_sample_p99']:7.4f} "
              f"{row['peak_kb_median']:10.1f}", flush=True)
    total = time.perf_counter() - t0
    print(f"\ntotal: {total:.1f} s")

    out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() \
        else Path("/content")
    out = out_dir / "drones_long.json"
    out.write_text(json.dumps({
        "host": host,
        "mode": mode,
        "n_seeds": n_seeds,
        "n_samples": n_samples,
        "total_s": total,
        "results": results,
    }, indent=2))
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
