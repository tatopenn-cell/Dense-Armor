"""Score the alerts of the ECG lead guard against the MIT-BIH beat annotations.

The ECG of scipy is the excerpt of record 208 (lead MLII) of the MIT-BIH
Arrhythmia Database that starts at sample 422 820. This script runs
``ecg_lead_guard.py``, reads the beat annotations of the same record from
PhysioNet, and reports for each beat type the share of beats that have an
alert within 150 ms. The reference is the same alert track shifted at
random: if the alerts did not depend on the beats, the shifted track would
score as well as the real one.

Install: pip install "dense-armor[biomedical]"
"""

import runpy
from pathlib import Path

import numpy as np
import wfdb

START = 422820
WINDOW = 54
N_SHIFTS = 1000

guard = runpy.run_path(str(Path(__file__).with_name("ecg_lead_guard.py")), run_name="ecg_run")
alerts = np.asarray(guard["hst_alerts"]).astype(bool)
alert_start = int(guard["alert_start"])
n = len(alerts)

ann = wfdb.rdann("208", "atr", pn_dir="mitdb")
keep = (ann.sample >= START) & (ann.sample < START + n)
beat_s = ann.sample[keep] - START
beat_sym = np.array(ann.symbol)[keep]


def hit(beats, track):
    c = np.concatenate([[0], np.cumsum(track)])
    lo = np.clip(beats - WINDOW, 0, n)
    hi = np.clip(beats + WINDOW + 1, 0, n)
    return (c[hi] - c[lo]) > 0


beats = {k: beat_s[(beat_sym == k) & (beat_s >= alert_start)] for k in ("V", "F", "N")}
print(f"Beats evaluated from sample {alert_start} ({alert_start / 360:.1f} s); window +-{WINDOW} samples")
rng = np.random.default_rng(0)
null = {k: [] for k in beats}
for _ in range(N_SHIFTS):
    shifted = np.roll(alerts, int(rng.integers(1080, n - 1080)))
    shifted[:alert_start] = False
    for k, b in beats.items():
        null[k].append(hit(b, shifted).mean())
for k, b in beats.items():
    obs = hit(b, alerts).mean()
    nul = np.array(null[k])
    print(f"  beat {k}: n={len(b)}, alert within window: {obs * 100:.1f}%; "
          f"shifted alerts: mean {nul.mean() * 100:.1f}%, p95 {np.percentile(nul, 95) * 100:.1f}%, "
          f"z={(obs - nul.mean()) / (nul.std() + 1e-12):.1f}")

idx = np.flatnonzero(alerts)
runs = np.split(idx, np.flatnonzero(np.diff(idx) > WINDOW) + 1) if len(idx) else []
ectopic = beat_s[np.isin(beat_sym, ["V", "F"])]
normal = beat_s[beat_sym == "N"]
near_ectopic = near_normal = near_none = 0
for r in runs:
    a0, a1 = r[0], r[-1]
    if np.any((ectopic >= a0 - WINDOW) & (ectopic <= a1 + WINDOW)):
        near_ectopic += 1
    elif np.any((normal >= a0 - WINDOW) & (normal <= a1 + WINDOW)):
        near_normal += 1
    else:
        near_none += 1
print(f"Alert episodes (merged within {WINDOW} samples): {len(runs)}; near a V or F beat: {near_ectopic}; "
      f"near only N beats: {near_normal}; near no beat: {near_none}")
