from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.utility.stats import DDSketch, TDigest

rng = np.random.default_rng(0)
x = rng.pareto(1.5, 20000) + 1.0
dd, td = DDSketch(alpha=0.01), TDigest(delta=200)
for v in x:
    dd.learn_one({"x": float(v)})
    td.learn_one({"x": float(v)})
qs = [0.5, 0.9, 0.99, 0.999]
true = np.quantile(x, qs)
edd = [abs(dd.quantile(q) - t) / t for q, t in zip(qs, true)]
etd = [abs(td.quantile(q) - t) / t for q, t in zip(qs, true)]

fig, ax = plt.subplots(figsize=(8, 3.5))
pos = np.arange(len(qs))
ax.bar(pos - 0.18, edd, 0.36, label="DDSketch (alpha = 0.01)")
ax.bar(pos + 0.18, etd, 0.36, label="TDigest (delta = 200)")
ax.axhline(0.01, ls="--", color="k", label="alpha tolerance")
ax.set_xticks(pos, [str(q) for q in qs])
ax.set_xlabel("quantile")
ax.set_ylabel("relative error")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(Path(__file__).with_name("quantiles_stream.png"), dpi=130)
print([round(e, 4) for e in edd], [round(e, 4) for e in etd])
