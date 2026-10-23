"""Generate docs/assets/neural/deep.png.

Three panels on synthetic data, from the library classes:

1. the training loss of SGD, SGD with momentum, SGD with Nesterov,
   RMSProp and Adam on the same synthetic regression with a small MLP,
   100 epochs, the median over 20 seeds. The script prints the final
   median loss of each optimizer with the 25th and 75th percentiles;
2. the curves of GELU, SiLU, ELU, LeakyReLU, Softplus and ReLU on
   [-4, 4];
3. the learning rate of NoamSchedule(64, 20) and of
   LinearWarmup(0.001, 0.01, 20) over 100 epochs.

The labels say ``synthetic``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.utility.neural.activations import Relu
from dense_armor.utility.neural.activations_more import (
    ELU,
    GELU,
    LeakyReLU,
    SiLU,
    Softplus,
)
from dense_armor.utility.neural.chain import Sequential
from dense_armor.utility.neural.layers import Dense
from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.schedules_more import LinearWarmup, NoamSchedule
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import SGD, Adam, RMSProp

HERE = Path(__file__).resolve().parents[2]
OUT = HERE / "docs" / "assets" / "neural" / "deep.png"


def _mlp(key):
    return Sequential(
        [
            Dense(2, 16, key),
            GELU(),
            Dense(16, 16, jax.random.PRNGKey(1)),
            GELU(),
            Dense(16, 1, jax.random.PRNGKey(2)),
        ]
    )


def _data(seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, (100, 2)).astype(np.float32)
    y = (np.sin(X[:, 0:1]) + X[:, 1:2] ** 2).astype(np.float32)
    return X, y


def _run(seed, opt_factory):
    X, y = _data(seed)
    m = _mlp(jax.random.PRNGKey(seed))
    t = Trainer(m, MSE(), opt_factory())
    t.fit(X, y, epochs=100, batch_size=20, key=jax.random.PRNGKey(seed + 1))
    return t.history_["train_loss"]


OPTS = [
    ("SGD", lambda: SGD(0.05)),
    ("SGD+momentum", lambda: SGD(0.05, momentum=0.9)),
    ("SGD+Nesterov", lambda: SGD(0.05, momentum=0.9, nesterov=True)),
    ("RMSProp", lambda: RMSProp(0.005)),
    ("Adam", lambda: Adam(0.01)),
]

curves = {name: [] for name, _ in OPTS}
for seed in range(20):
    for name, factory in OPTS:
        h = _run(seed, factory)
        curves[name].append(h)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

final_medians = {}
n_diverged = {}
for name, _ in OPTS:
    arr = np.array(curves[name])
    n_div = int(np.sum(~np.isfinite(arr[:, -1])))
    n_diverged[name] = n_div
    if n_div == arr.shape[0]:
        continue
    med = np.nanmedian(arr, axis=0)
    q1 = np.nanpercentile(arr, 25, axis=0)
    q3 = np.nanpercentile(arr, 75, axis=0)
    epochs = np.arange(1, arr.shape[1] + 1)
    axes[0].plot(epochs, med, label=name)
    axes[0].fill_between(epochs, q1, q3, alpha=0.15)
    final_medians[name] = (med[-1], q1[-1], q3[-1])

axes[0].set_xlabel("epoch")
axes[0].set_ylabel("train loss (MSE)")
axes[0].set_title(
    "Optimizers on synthetic regression (median, 20 seeds)", weight="bold", fontsize=10
)
axes[0].set_yscale("log")
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=8)

xs = jnp.linspace(-4.0, 4.0, 401)
acts = [
    ("ReLU", Relu().forward(xs)),
    ("GELU", GELU().forward(xs)),
    ("SiLU", SiLU().forward(xs)),
    ("ELU", ELU().forward(xs)),
    ("LeakyReLU", LeakyReLU(0.01).forward(xs)),
    ("Softplus", Softplus().forward(xs)),
]
for name, y in acts:
    axes[1].plot(np.asarray(xs), np.asarray(y), label=name)
axes[1].set_xlabel("x")
axes[1].set_ylabel("activation(x)")
axes[1].set_title("Activations on [-4, 4] (synthetic)", weight="bold", fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=8)

epochs_lr = np.arange(0, 101)
noam = [NoamSchedule(64, 20)(float(e)) for e in epochs_lr]
linw = [LinearWarmup(0.001, 0.01, 20)(float(e)) for e in epochs_lr]
axes[2].plot(epochs_lr, noam, label="NoamSchedule(64, 20)")
axes[2].plot(epochs_lr, linw, label="LinearWarmup(0.001, 0.01, 20)")
axes[2].set_xlabel("epoch")
axes[2].set_ylabel("learning rate")
axes[2].set_title("Two warmup schedules (synthetic)", weight="bold", fontsize=10)
axes[2].grid(True, alpha=0.3)
axes[2].legend(fontsize=8)

fig.suptitle("deep neural-network on synthetic data", weight="bold", fontsize=12)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=140)
print(f"written: {OUT} ({OUT.stat().st_size} B)")

print("\nfinal median loss (25th, 75th) after 100 epochs:")
for name in final_medians:
    m, q1, q3 = final_medians[name]
    n_div = n_diverged[name]
    extra = f"  [{n_div}/20 seeds diverged]" if n_div else ""
    print(f"  {name:15s}: median={m:.4f}, q1={q1:.4f}, q3={q3:.4f}{extra}")
