"""Generate docs/assets/neural/continuous.png.

Left: the progressive error of OnlineNetClassifier and of
HedgeNetClassifier on a synthetic stream whose rule changes at sample
500 (the change is marked). Centre: the alphas of the Hedge network
against the sample. Right: the time of learn_one called directly and
called through BackgroundLearner on the same estimator. All three from
the library classes, labelled synthetic.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import jax
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.utility.neural.background import BackgroundLearner
from dense_armor.utility.neural.hedge import HedgeNetClassifier
from dense_armor.utility.neural.losses import BCEWithLogits
from dense_armor.utility.neural.models import mlp
from dense_armor.utility.neural.online import OnlineNetClassifier
from dense_armor.utility.optim import SGD

HERE = Path(__file__).resolve().parents[2]
OUT = HERE / "docs" / "assets" / "neural" / "continuous.png"

rng = np.random.default_rng(0)
n = 1000
change = 500
X = rng.normal(0.0, 1.0, (n, 2)).astype(np.float32)
y_pre = (X[:, 0] + X[:, 1] > 0.0).astype(int)
y_post = (X[:, 0] - X[:, 1] > 0.0).astype(int)
y = np.where(np.arange(n) < change, y_pre, y_post)

net = OnlineNetClassifier(
    mlp([2, 8, 1], jax.random.PRNGKey(0)),
    BCEWithLogits(),
    SGD(0.05),
    classes=[0, 1],
    feature_keys=["a", "b"],
)
hedge = HedgeNetClassifier(
    [2, 8, 8, 8],
    jax.random.PRNGKey(0),
    classes=[0, 1],
    beta=0.99,
    lr=0.05,
    feature_keys=["a", "b"],
)

net_err, hedge_err = [], []
alphas_snapshots = []
for i in range(n):
    d = {"a": float(X[i, 0]), "b": float(X[i, 1])}
    net_pred = net.predict_one(d)
    hedge_pred = hedge.predict_one(d)
    net_err.append(0.0 if net_pred == int(y[i]) else 1.0)
    hedge_err.append(0.0 if hedge_pred == int(y[i]) else 1.0)
    net.learn_one(d, int(y[i]))
    hedge.learn_one(d, int(y[i]))
    if i % 10 == 0:
        alphas_snapshots.append(np.asarray(hedge.alphas_).copy())

window = 50


def _roll(x):
    return np.convolve(x, np.ones(window) / window, mode="valid")


net_roll = _roll(net_err)
hedge_roll = _roll(hedge_err)
alphas_arr = np.array(alphas_snapshots)

direct_times = []
for i in range(200):
    s = time.perf_counter()
    net.learn_one({"a": float(X[i, 0]), "b": float(X[i, 1])}, int(y[i]))
    direct_times.append(time.perf_counter() - s)

m2 = mlp([2, 8, 1], jax.random.PRNGKey(0))
inner = OnlineNetClassifier(
    m2, BCEWithLogits(), SGD(0.05), classes=[0, 1], feature_keys=["a", "b"]
)
bg = BackgroundLearner(inner, maxsize=1000)
bg_times = []
for i in range(200):
    s = time.perf_counter()
    bg.learn_one({"a": float(X[i, 0]), "b": float(X[i, 1])}, int(y[i]))
    bg_times.append(time.perf_counter() - s)
bg.flush()
bg.close()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.2))
ax1.plot(net_roll, label="OnlineNetClassifier")
ax1.plot(hedge_roll, label="HedgeNetClassifier")
ax1.axvline(change, color="k", linestyle="--", alpha=0.5, label="rule change")
ax1.set_xlabel("sample")
ax1.set_ylabel(f"error (rolling mean, w={window})")
ax1.set_title("progressive error (synthetic)", weight="bold", fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=8)

for k in range(alphas_arr.shape[1]):
    ax2.plot(np.arange(0, n, 10), alphas_arr[:, k], label=f"depth {k}")
ax2.axvline(change, color="k", linestyle="--", alpha=0.5)
ax2.set_xlabel("sample")
ax2.set_ylabel("alpha(l)")
ax2.set_title("Hedge alphas (synthetic)", weight="bold", fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8)

ax3.plot(np.array(direct_times) * 1e3, label="learn_one direct")
ax3.plot(np.array(bg_times) * 1e3, label="learn_one via BackgroundLearner")
ax3.set_xlabel("sample")
ax3.set_ylabel("time (ms)")
ax3.set_title("learn_one time (synthetic)", weight="bold", fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=8)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=140)
print(f"written: {OUT} ({OUT.stat().st_size} B)")
print(
    f"net rolling err: pre={net_roll[:100].mean():.3f} post={net_roll[-100:].mean():.3f}"
)
print(
    f"hedge rolling err: pre={hedge_roll[:100].mean():.3f} post={hedge_roll[-100:].mean():.3f}"
)
print(f"alphas final: {np.round(alphas_arr[-1], 3).tolist()}")
print(
    f"direct p50={np.median(direct_times) * 1e3:.3f} ms p99={np.percentile(direct_times, 99) * 1e3:.3f} ms"
)
print(
    f"bg p50={np.median(bg_times) * 1e3:.3f} ms p99={np.percentile(bg_times, 99) * 1e3:.3f} ms"
)
