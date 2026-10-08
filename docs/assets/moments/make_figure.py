from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.utility.stats import EWStats, RunningMoments

rng = np.random.default_rng(0)
x = np.concatenate([rng.normal(0, 1, 200), rng.normal(3, 1, 200)])
m, ew = RunningMoments(), EWStats(alpha=0.05)
mean, std, ewm = [], [], []
for v in x:
    m.learn_one({"x": float(v)})
    ew.learn_one({"x": float(v)})
    mean.append(m.mean)
    std.append(m.std)
    ewm.append(ew.mean)

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(x, color="0.75", lw=0.8, label="stream")
ax.plot(mean, lw=1.8, label="running mean")
ax.plot(std, lw=1.4, label="running std")
ax.plot(ewm, "--", lw=1.8, label="EW mean (alpha = 0.05)")
ax.axvline(200, ls=":", color="k")
ax.set_xlabel("sample")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(Path(__file__).with_name("moments_stream.png"), dpi=130)
