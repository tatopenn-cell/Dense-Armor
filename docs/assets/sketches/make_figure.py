from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.utility.stats import BloomFilter, CountMinSketch, HyperLogLog, SpaceSaving

rng = np.random.default_rng(0)
lengths = [10**3, 10**4, 10**5, 10**6]
err = {"CountMinSketch": [], "HyperLogLog": [], "BloomFilter": [], "SpaceSaving": []}
for n in lengths:
    items = rng.zipf(1.3, n) % 5000
    cms, hll, bf, ss = CountMinSketch(), HyperLogLog(), BloomFilter(n_expected=5000), SpaceSaving(k=20)
    for i in items:
        s = {"x": str(i)}
        for e in (cms, hll, bf, ss):
            e.learn_one(s)
    u, c = np.unique(items, return_counts=True)
    top = u[np.argsort(-c)[:10]]
    cnt = dict(zip(u, c))
    err["CountMinSketch"].append(np.mean([abs(cms.estimate(str(k)) - cnt[k]) / cnt[k] for k in top]))
    err["HyperLogLog"].append(abs(hll.count() - len(u)) / len(u))
    err["BloomFilter"].append(np.mean([bf.check(f"unseen{j}") for j in range(2000)]))
    err["SpaceSaving"].append(np.mean([abs(ss.estimate(str(k)) - cnt[k]) / cnt[k] for k in top]))

fig, ax = plt.subplots(figsize=(8, 3.5))
for k, v in err.items():
    ax.plot(lengths, v, "o-", lw=1.6, label=k)
ax.set_xscale("log")
ax.set_xlabel("stream length")
ax.set_ylabel("average error")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(Path(__file__).with_name("sketches_stream.png"), dpi=130)
print({k: [round(float(x), 4) for x in v] for k, v in err.items()})
