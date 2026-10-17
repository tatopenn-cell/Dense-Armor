"""CASPER false alarms as episodes, with and without reset_on_alarm.

Runs the same hours and configuration as ``benchmarks/casper_hours.py`` and
reports, per hour, four numbers side by side:

- flagged samples with ``reset_on_alarm=False`` (the old definition);
- alarm episodes (rising edges of ``drift_detected``) with
  ``reset_on_alarm=False``;
- flagged samples with ``reset_on_alarm=True``;
- alarm episodes with ``reset_on_alarm=True``.

The episodes are rising edges of ``drift_detected``. An hour that holds an
anomalous label is reported and left out of the statistics, as in
``casper_hours.py``.

Command, from the repository root, with the Kaggle copy of the dataset in
place:

    python benchmarks/casper_hours_episodes.py --hours 0-14

    python benchmarks/casper_hours_episodes.py --hours 0-0 --minutes 10 --smoke
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import casper_benchmark as cb
from dense_armor.utility.drift.detector import CUSUMDriftDetector

K = 2.0
H = 20.0
JOINT_COL = "Actual Joint Velocities"
LABEL_COL = "Anomaly State"


def count_alarms(det, series):
    """Count flagged samples and alarm episodes on ``series``.

    Args:
        det: a :class:`CUSUMDriftDetector`.
        series: an iterable of floats.

    Returns:
        ``(flagged, episodes)``, where ``flagged`` is the number of samples
        with ``drift_detected`` true and ``episodes`` the number of rising
        edges of ``drift_detected``.
    """
    flagged = 0
    episodes = 0
    prev = False
    for x in series:
        det.update(float(x))
        now = bool(det.drift_detected)
        if now:
            flagged += 1
        if now and not prev:
            episodes += 1
        prev = now
    return flagged, episodes


def _find_arm():
    for root in (Path("/kaggle/input"), Path.home() / ".cache" / "kagglehub"):
        arm = next(root.rglob("right_arm.csv"), None) if root.exists() else None
        if arm is not None:
            return arm.parent, arm
    base, arm, _ = cb.find_dataset()
    return base, arm


def _moving_joint(arm, t0):
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

    det_off = CUSUMDriftDetector(reference="fixed", ref_mult=cb.CUSUM_REF_1C,
                                 k=K, h=H, reset_on_alarm=False)
    det_on = CUSUMDriftDetector(reference="fixed", ref_mult=cb.CUSUM_REF_1C,
                                k=K, h=H, reset_on_alarm=True)
    flagged_off, alarms_off = count_alarms(det_off, s)
    flagged_on, alarms_on = count_alarms(det_on, s)
    hours = len(s) / cb.FS / 3600.0 if len(s) > 0 else 0.0
    inv = 1.0 / hours if hours > 0 else float("nan")
    return {
        "hour": hour,
        "rows": len(s),
        "anomalous_rows": int(np.sum(lab == 1)),
        "flagged_off": flagged_off,
        "alarms_off": alarms_off,
        "flagged_on": flagged_on,
        "alarms_on": alarms_on,
        "flagged_per_hour_off": flagged_off * inv,
        "episodes_per_hour_off": alarms_off * inv,
        "flagged_per_hour_on": flagged_on * inv,
        "episodes_per_hour_on": alarms_on * inv,
    }


def _parse_hours(spec):
    a, _, b = spec.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", default="0-14")
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="benchmarks/casper_hours_episodes.json")
    a = ap.parse_args()

    if a.smoke:
        a.hours = "0-0"
        a.minutes = min(a.minutes, 10.0)

    base, arm = _find_arm()
    t0 = float(cb.pd.read_csv(arm, nrows=1, usecols=["Timestamp"])["Timestamp"].iloc[0])
    joint = _moving_joint(arm, t0)
    print(f"dataset {base}  t0 = {t0}  joint = {joint}  k = {K}  h = {H}")

    hours = _parse_hours(a.hours)
    t_first = t0 + hours[0] * 3600.0 - cb.T / cb.FS
    t_last = t0 + hours[-1] * 3600.0 + a.minutes * 60.0
    df_all = cb.load_slice(arm, t_first, t_last, [JOINT_COL, LABEL_COL])
    print(f"rows: {len(df_all):,}")

    print(f"{'hour':>4} {'rows':>7} {'anom':>6} "
          f"{'flag_off':>9} {'ep_off':>7} {'flag_on':>8} {'ep_on':>6}")
    rows = []
    for hour in hours:
        r = one_window(df_all, t0, joint, hour, a.minutes)
        rows.append(r)
        print(f"{r['hour']:>4} {r['rows']:>7} {r['anomalous_rows']:>6} "
              f"{r['flagged_off']:>9} {r['alarms_off']:>7} "
              f"{r['flagged_on']:>8} {r['alarms_on']:>6}")

    normal = [r for r in rows if r["anomalous_rows"] == 0]
    summary = {
        "windows": len(rows),
        "normal_windows": len(normal),
        "flagged_off_median": float(np.median([r["flagged_per_hour_off"] for r in normal])) if normal else None,
        "episodes_off_median": float(np.median([r["episodes_per_hour_off"] for r in normal])) if normal else None,
        "flagged_on_median": float(np.median([r["flagged_per_hour_on"] for r in normal])) if normal else None,
        "episodes_on_median": float(np.median([r["episodes_per_hour_on"] for r in normal])) if normal else None,
    }
    print(json.dumps(summary, indent=1))
    Path(a.out).write_text(json.dumps(
        {"config": {"k": K, "h": H, "smoke": a.smoke, "joint": joint,
                    "minutes": a.minutes},
         "windows": rows, "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
