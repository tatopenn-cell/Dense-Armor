from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.utility.stats import RollingCorrelation, RunningCorrelation

rng = np.random.default_rng(0)
f = rng.normal(size=250)
a = np.concatenate([f + 0.33 * rng.normal(size=250), rng.normal(size=250)])
b = np.concatenate([f + 0.33 * rng.normal(size=250), rng.normal(size=250)])
run, rol = RunningCorrelation(), RollingCorrelation(window=50)
cr, cw = [], []
for x, y in zip(a, b):
    s = {"x": float(x), "y": float(y)}
    run.learn_one(s)
    rol.learn_one(s)
    cr.append(run.corr)
    cw.append(rol.corr)

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(cr, lw=1.8, label="running correlation")
ax.plot(cw, lw=1.8, label="rolling correlation (window = 50)")
ax.axvline(250, ls=":", color="k")
ax.set_ylim(-0.5, 1.05)
ax.set_xlabel("sample")
ax.set_ylabel("Pearson r")
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout()
fig.savefig(Path(__file__).with_name("dependence_stream.png"), dpi=130)
