"""Run the actuator-fault pipeline on the processed ALFA flights.

One row per flight: the detection delay (the metric of Keipour,
Mousaei & Scherer 2020, section 4, page 5) and the alarm episodes per
hour of normal flight, with ``reset_on_alarm=True`` so a sustained
condition gives one episode, not one per sample. The summary reports
median and interquartile range over the flights, never the bare mean.

The flights are the folders of ``processed/`` in the ALFA dataset. The
ground truth is the ``failure_status-*.csv`` file of each folder (paper
section 3.3, page 4), read by ``pipelines.drones.alfa.alfa_fault_time``.

Run from the repository root:

    python benchmarks/drones/alfa_actuator.py --data /path/to/processed
    python benchmarks/drones/alfa_actuator.py --data /path/to/processed --smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pipelines.drones.actuator_fault import run
from pipelines.drones.alfa import alfa_fault_time, read_alfa

ROOT = Path(__file__).resolve().parents[2]
SMOKE_FLIGHTS = 2


def _one_flight(flight_dir: Path) -> dict:
    t_fault, kind = alfa_fault_time(flight_dir)
    samples = list(read_alfa(flight_dir))
    r = run(samples)
    delay = None
    if r.delays_s:
        delay = float(r.delays_s[0])
    return {
        "flight": flight_dir.name,
        "n_samples": r.n_samples,
        "fault_kind": kind,
        "t_fault_s": t_fault,
        "n_episodes": r.n_episodes,
        "delay_s": delay,
        "false_alarms": r.false_alarms,
        "normal_seconds": r.normal_seconds,
        "false_alarms_per_hour": r.false_alarms_per_hour,
    }


def _summary(rows: list[dict]) -> dict:
    delays = np.array([r["delay_s"] for r in rows if r["delay_s"] is not None])
    fah = np.array([r["false_alarms_per_hour"] for r in rows
                    if np.isfinite(r["false_alarms_per_hour"])])
    out: dict = {"n_flights": len(rows),
                 "n_with_detection": int(delays.size)}
    for name, arr in (("delay_s", delays),
                      ("false_alarms_per_hour", fah)):
        if arr.size == 0:
            out[f"{name}_median"] = None
            out[f"{name}_iqr"] = None
            continue
        qs = np.percentile(arr, [25, 50, 75]).tolist()
        q1, q2, q3 = float(qs[0]), float(qs[1]), float(qs[2])
        out[f"{name}_median"] = float(q2)
        out[f"{name}_iqr"] = float(q3 - q1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True,
                    help="folder that contains the processed ALFA flights")
    ap.add_argument("--smoke", action="store_true",
                    help="only the first two flights")
    ap.add_argument("--out", type=Path, default=None,
                    help="JSON output, default benchmarks/drones/alfa_actuator.json")
    args = ap.parse_args()
    flights = sorted(p for p in args.data.iterdir() if p.is_dir())
    if args.smoke:
        flights = flights[:SMOKE_FLIGHTS]
    rows = [_one_flight(d) for d in flights]
    summary = _summary(rows)
    payload = {"dataset": "ALFA", "data": str(args.data),
               "n_flights": len(rows), "rows": rows, "summary": summary}
    out = args.out or (ROOT / "benchmarks" / "drones" / "alfa_actuator.json")
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"{'flight':<58} {'delay_s':>8} {'fa/h':>8}")
    for r in rows:
        d = "-" if r["delay_s"] is None else f"{r['delay_s']:.2f}"
        f = "nan" if not np.isfinite(r["false_alarms_per_hour"]) else f"{r['false_alarms_per_hour']:.1f}"
        print(f"{r['flight']:<58} {d:>8} {f:>8}")
    print()
    print(f"flights: {summary['n_flights']}, with a detection: "
          f"{summary['n_with_detection']}")
    for k in ("delay_s_median", "delay_s_iqr",
              "false_alarms_per_hour_median", "false_alarms_per_hour_iqr"):
        v = summary[k]
        print(f"{k}: {'-' if v is None else round(v, 3)}")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
