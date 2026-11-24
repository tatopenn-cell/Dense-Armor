from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.core.signals.stats.moments import EWStats, RunningMoments, RunningMomentsVector

OUT = Path(__file__).parent
rng = np.random.default_rng(0)

x = rng.gamma(4.0, 3.0, 500)
m = RunningMoments()
rows = {"mean": [], "std": [], "skewness": [], "kurtosis": []}
for v in x:
    m.learn_one({"x": float(v)})
    for k in rows:
        rows[k].append(getattr(m, k))
true = {"mean": 12.0, "std": 6.0, "skewness": 1.0, "kurtosis": 1.5}
fig, ax = plt.subplots(1, 4, figsize=(11, 2.8))
for a, k in zip(ax, rows):
    a.plot(rows[k], lw=1.6)
    a.axhline(true[k], ls="--", color="k", lw=1)
    a.set_title(k)
    a.set_xlabel("payment")
fig.tight_layout()
fig.savefig(OUT / "moments_running.png", dpi=130)

y = (90.0 + 0.1 * rng.standard_normal(20000)).astype(np.float32)
f32 = RunningMoments(dtype=np.float32)
s1 = np.float32(0.0)
s2 = np.float32(0.0)
naive, welford, exact = [], [], []
for i, v in enumerate(y, 1):
    f32.learn_one({"x": float(v)})
    s1 += v
    s2 += v * v
    if i >= 2 and i % 50 == 0:
        naive.append(float((s2 - s1 * s1 / np.float32(i)) / np.float32(i - 1)))
        welford.append(f32.var)
        exact.append(float(np.var(y[:i].astype(np.float64), ddof=1)))
n = np.arange(50, 20001, 50)
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(n, naive, lw=1, label="sum of squares, float32")
ax.plot(n, welford, lw=1.8, label="RunningMoments, float32")
ax.plot(n, exact, "k--", lw=1, label="exact, float64")
ax.axhline(0, color="0.6", lw=0.8)
ax.set_xlabel("sample")
ax.set_ylabel("variance")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "moments_cancellation.png", dpi=130)
print("naive range", min(naive), max(naive), "welford", welford[-1], "exact", exact[-1])

parts = np.array_split(rng.gamma(4.0, 3.0, 2000), 4)
ws = []
for p in parts:
    w = RunningMoments()
    for v in p:
        w.learn_one({"x": float(v)})
    ws.append(w)
g = ws[0]
for w in ws[1:]:
    g = g.merge(w)
one = RunningMoments()
for v in np.concatenate(parts):
    one.learn_one({"x": float(v)})
t = np.arange(600) * 0.01
q = np.stack([np.sin(2 * np.pi * 0.5 * t), 0.5 * np.cos(2 * np.pi * 0.3 * t)], 1)
q = q + 0.05 * rng.standard_normal(q.shape)
vec = RunningMomentsVector()
mm = [[], []]
ss = [[], []]
for r in q:
    vec.learn_one({"q0": float(r[0]), "q1": float(r[1])})
    s = vec.transform_one({})
    for j, k in enumerate(("q0", "q1")):
        mm[j].append(s[k]["mean"])
        ss[j].append(s[k]["std"])
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for i, w in enumerate(ws):
    ax[0].errorbar(i, w.mean, yerr=w.std, fmt="o", capsize=4, color="C0")
ax[0].errorbar(4, g.mean, yerr=g.std, fmt="s", capsize=4, color="C3", label="merged")
ax[0].errorbar(5, one.mean, yerr=one.std, fmt="^", capsize=4, color="k", label="one accumulator")
ax[0].set_xticks(range(6), ["w1", "w2", "w3", "w4", "merged", "single"])
ax[0].set_title("four workers, merged and single: mean and std")
ax[0].set_ylim(0, 26)
ax[0].legend(fontsize=8, loc="upper center", ncol=2)
ax[1].plot(t, q[:, 0], color="0.8", lw=0.8)
ax[1].plot(t, q[:, 1], color="0.8", lw=0.8)
for j in range(2):
    ax[1].plot(t, mm[j], color=f"C{j}", lw=1.8, label=f"q{j} mean")
    ax[1].plot(t, ss[j], "--", color=f"C{j}", lw=1.4, label=f"q{j} std")
ax[1].set_xlabel("time (s)")
ax[1].set_title("RunningMomentsVector on two joints")
ax[1].legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(OUT / "moments_merge.png", dpi=130)
print("merge", g.count, g.mean, one.mean, g.var, one.var)

s = np.concatenate([np.full(300, 1.0), np.full(300, 5.0)]) + 0.3 * rng.standard_normal(600)
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(s, color="0.8", lw=0.8, label="stream")
rm = RunningMoments()
rr = []
for v in s:
    rm.learn_one({"x": float(v)})
    rr.append(rm.mean)
ax.plot(rr, "k", lw=1.6, label="RunningMoments")
for a in (0.5, 0.05, 0.005):
    e = EWStats(alpha=a)
    ee = []
    for v in s:
        e.learn_one({"x": float(v)})
        ee.append(e.mean)
    ax.plot(ee, lw=1.6, label=f"EWStats alpha = {a}")
ax.axvline(300, ls=":", color="k")
ax.set_xlabel("sample")
ax.legend(fontsize=8, loc="center right")
fig.tight_layout()
fig.savefig(OUT / "moments_alpha.png", dpi=130)
