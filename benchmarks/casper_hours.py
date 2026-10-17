"""CASPER false alarms on every normal hour of the recording.

Runs the zero-false-alarm configuration of docs/benchmarks/real_robot_benchmark.md
(smoothed energy of the cycle residual, trailing mean over 2T, one-sided CUSUM with
a fixed reference of one cycle, k = 2.0, h = 20) on each hour of the recording,
using the functions of benchmarks/casper_benchmark.py unchanged. An hour that holds
an anomalous label is reported and left out of the statistics.

Command, from the repository root, with the Kaggle copy of the dataset in place:

    python benchmarks/casper_hours.py --hours 0-14

    python benchmarks/casper_hours.py --hours 0-1 --minutes 10 --smoke
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import casper_benchmark as cb

K = 2.0
H = 20.0
JOINT_COL = "Actual Joint Velocities"
LABEL_COL = "Anomaly State"


def find_arm():
    for root in (Path("/kaggle/input"), Path.home() / ".cache" / "kagglehub"):
        arm = next(root.rglob("right_arm.csv"), None) if root.exists() else None
        if arm is not None:
            return arm.parent, arm
    base, arm, _ = cb.find_dataset()
    return base, arm


def moving_joint(arm, t0):
    df = cb.load_slice(arm, t0, t0 + 3600.0, [JOINT_COL])
    best, var_best = None, -1.0
    for j in range(6):
        v = cb.parse_joint(df, JOINT_COL, j)
        lag, _ = cb.strongest_period(v)
        if lag is not None and float(np.var(v)) > var_best:
            best, var_best = j, float(np.var(v))
    return best


def one_window(df_all, t0, joint, hour, minutes):
    ta = t0 + hour * 3600.0
    tb = ta + minutes * 60.0
    ts = df_all["Timestamp"].to_numpy()
    df = df_all.loc[(ts >= ta - cb.T / cb.FS) & (ts <= tb)]
    v = cb.parse_joint(df, JOINT_COL, joint)
    lab = df[LABEL_COL].to_numpy()[cb.T:]
    s = cb.trailing_mean(cb.shift_residual_fixed(v, cb.T) ** 2, 2 * cb.T)
    t = time.perf_counter()
    _, fa_h = cb.run_cusum(s, len(s), k=K, h=H, reference="fixed", ref_mult=cb.CUSUM_REF_1C)
    seconds = time.perf_counter() - t
    return {
        "hour": hour,
        "rows": len(s),
        "anomalous_rows": int(np.sum(lab == 1)),
        "alarms": round(fa_h * len(s) / cb.FS / 3600.0),
        "false_alarms_per_hour": float(fa_h),
        "cusum_seconds": round(seconds, 2),
    }


def parse_hours(spec):
    a, _, b = spec.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", default="0-14")
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="benchmarks/casper_hours.json")
    a = ap.parse_args()
    base, arm = find_arm()
    t0 = float(cb.pd.read_csv(arm, nrows=1, usecols=["Timestamp"])["Timestamp"].iloc[0])
    joint = moving_joint(arm, t0)
    print(f"dataset {base}  t0 = {t0}  moving joint = {joint}  k = {K}  h = {H}")
    hours = parse_hours(a.hours)
    t_first = t0 + hours[0] * 3600.0 - cb.T / cb.FS
    t_last = t0 + hours[-1] * 3600.0 + a.minutes * 60.0
    df_all = cb.load_slice(arm, t_first, t_last, [JOINT_COL, LABEL_COL])
    print(f"rows read once: {len(df_all):,}")
    rows = []
    print(f"{'hour':>5} {'rows':>8} {'anomalous':>10} {'alarms':>7} {'FA/h':>9}")
    for hour in hours:
        r = one_window(df_all, t0, joint, hour, a.minutes)
        rows.append(r)
        print(f"{r['hour']:>5} {r['rows']:>8} {r['anomalous_rows']:>10} {r['alarms']:>7} {r['false_alarms_per_hour']:>9.1f}")
    normal = [r for r in rows if r["anomalous_rows"] == 0]
    fa = [r["false_alarms_per_hour"] for r in normal]
    summary = {
        "windows": len(rows),
        "normal_windows": len(normal),
        "median_false_alarms_per_hour": float(np.median(fa)) if fa else None,
        "max_false_alarms_per_hour": float(np.max(fa)) if fa else None,
        "windows_with_zero_alarms": int(sum(1 for x in fa if x == 0.0)),
        "excluded_with_anomalous_label": [r["hour"] for r in rows if r["anomalous_rows"] > 0],
    }
    print(json.dumps(summary, indent=1))
    out = Path(a.out)
    out.write_text(json.dumps({"config": {"k": K, "h": H, "signal": "trailing mean(e^2, 2T)",
                                           "reference": "fixed", "joint": joint,
                                           "minutes": a.minutes, "smoke": a.smoke},
                               "windows": rows, "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
