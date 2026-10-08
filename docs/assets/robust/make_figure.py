from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.utility.stats import RollingIQR, RollingMAD, RollingMedian, RunningMoments

rng = np.random.default_rng(0)
x = rng.normal(0, 1, 200)
x[80], x[140] = 15.0, -12.0
m, med, mad, iqr = RunningMoments(), RollingMedian(window=31), RollingMAD(window=31), RollingIQR(window=31)
tr = {k: [] for k in ("mean", "std", "median", "MAD", "IQR")}
for v in x:
    s = {"x": float(v)}
    for e in (m, med, mad, iqr):
        e.learn_one(s)
    tr["mean"].append(m.mean)
    tr["std"].append(m.std)
    tr["median"].append(med.value)
    tr["MAD"].append(mad.value)
    tr["IQR"].append(iqr.value)

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(x, color="0.75", lw=0.8, label="stream")
for k, v in tr.items():
    ax.plot(v, lw=1.5, ls="--" if k in ("mean", "std") else "-", label=f"running {k}" if k in ("mean", "std") else f"rolling {k}")
ax.set_ylim(-4, 6)
ax.set_xlabel("sample")
ax.legend(loc="upper left", fontsize=8, ncol=3)
fig.tight_layout()
fig.savefig(Path(__file__).with_name("robust_stream.png"), dpi=130)
