import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.utility.datasets import DriftStream
from dense_armor.utility.forest import OnlineExtraTreesRegressor
from dense_armor.utility.rules import AMRulesRegressor
from dense_armor.utility.tree import MondrianForestRegressor

out = Path(__file__).parent
N, D = 2000, 4

LEARNERS = {
    "OXT": lambda s: OnlineExtraTreesRegressor(seed=s),
    "MondrianForest": lambda s: MondrianForestRegressor(n_trees=10, seed=s),
    "AMRules": lambda s: AMRulesRegressor(seed=s),
}
COLORS = {"OXT": "tab:blue", "MondrianForest": "tab:green", "AMRules": "tab:orange"}

curves = {k: [] for k in LEARNERS}
for seed in range(5):
    X, y = DriftStream(n_samples=N, n_features=D, n_drifts=1,
                       kind="sudden", seed=seed).data()
    names = [f"x{i}" for i in range(D)]
    for name, make in LEARNERS.items():
        est = make(seed)
        err = np.empty(N)
        for i, (xi, yi) in enumerate(zip(X, y)):
            d = dict(zip(names, map(float, xi)))
            err[i] = abs(float(est.predict_one(d)) - float(yi))
            est.learn_one(d, float(yi))
        curves[name].append(err)

win = 50
fig, ax = plt.subplots(figsize=(8, 3.4))
for name, errs in curves.items():
    smooth = np.median(
        [np.convolve(e, np.ones(win) / win, mode="valid") for e in errs],
        axis=0,
    )
    ax.plot(np.arange(len(smooth)) + win // 2, smooth,
            color=COLORS[name], lw=1.3, label=name)

ax.axvline(N // 2, color="0.5", lw=0.8, ls="--")
ax.text(N // 2 + 8, ax.get_ylim()[1] * 0.92, "rule changes",
        fontsize=8, color="0.4")
ax.set_xlabel("sample")
ax.set_ylabel("median |error| over five seeds (50-sample window)")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(out / "oxt_recovery.png", dpi=130)
print("scritto:", out / "oxt_recovery.png")
