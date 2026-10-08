from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.utility.drift.detector import CUSUMDriftDetector
from dense_armor.utility.metrics import events

out = Path(__file__).parent
rng = np.random.default_rng(1)
x = np.concatenate([rng.normal(0, 1, 3000), rng.normal(3, 1, 1000)])
det = CUSUMDriftDetector(reference="fixed", radius=76, ref_mult=10, k=0.5, h=20.0)
alarms = []
for i, v in enumerate(x):
    det.update(float(v))
    if det.drift_detected:
        alarms.append(i / 20)
t = np.arange(len(x)) / 20
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.plot(t, x, color="0.6", lw=0.6, label="stream (mean steps from 0 to 3 at t = 150 s)")
ax.axvspan(150.0, 199.95, color="tab:orange", alpha=0.15, label="labelled event")
ax.axvline(alarms[0], color="tab:red", lw=1.5, label=f"first alarm (+{alarms[0] - 150:.2f} s)")
ax.set_xlim(120, 200)
ax.set_xlabel("time (s)")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(out / "event_detection.png", dpi=130)

y = np.linspace(-1.0, 2.0, 400)
s = [events._nab_sigmoid(v) for v in y]
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.plot(y, s, color="tab:blue")
ax.axvspan(-1.0, 0.0, color="tab:orange", alpha=0.15, label="anomaly window")
ax.axhline(0.0, color="0.5", lw=0.8)
ax.set_xlabel("relative position y (window from -1 to 0)")
ax.set_ylabel("s(y)")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(out / "nab_sigmoid.png", dpi=130)
print(alarms[0] - 150, len(alarms))
