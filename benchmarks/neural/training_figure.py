"""Generate docs/assets/neural/training.png.

Left: train loss and val loss against epoch on the synthetic early-
stopping set, with the best epoch marked. Right: the learning rate of
CosineWarmRestarts(0.05, 0.0, 10) with t_mult=1 and t_mult=2. Both
from the library classes, labelled synthetic.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import jax
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.models import mlp
from dense_armor.utility.neural.schedules import CosineWarmRestarts
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import Adam

HERE = Path(__file__).resolve().parents[2]
OUT = HERE / "docs" / "assets" / "neural" / "training.png"

rng = np.random.default_rng(0)
n = 60
X = rng.normal(0.0, 1.0, (n, 2)).astype(np.float32)
y = (X[:, 0:1] + 0.5 * X[:, 1:2]).astype(np.float32)
X_val = rng.normal(0.0, 1.0, (30, 2)).astype(np.float32)
y_val = (X_val[:, 0:1] + 0.5 * X_val[:, 1:2]).astype(np.float32)
y_val = y_val + rng.normal(0.0, 0.4, y_val.shape).astype(np.float32)

model = mlp([2, 16, 16, 1], jax.random.PRNGKey(0))
tr = Trainer(model, MSE(), Adam(0.01))
tr.fit(
    X,
    y,
    epochs=150,
    batch_size=10,
    key=jax.random.PRNGKey(1),
    X_val=X_val,
    y_val=y_val,
    patience=3,
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

epochs = list(range(len(tr.history_["train_loss"])))
ax1.plot(epochs, tr.history_["train_loss"], "o-", label="train loss", ms=3)
ax1.plot(epochs, tr.history_["val_loss"], "s-", label="val loss", ms=3)
if tr.best_epoch_ is not None:
    best_ep = tr.best_epoch_
    ax1.axvline(
        best_ep, color="k", linestyle="--", alpha=0.5, label=f"best epoch ({best_ep})"
    )
ax1.set_xlabel("epoch")
ax1.set_ylabel("MSE loss")
ax1.set_title("early stopping on the synthetic set", weight="bold", fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=8)

c1 = CosineWarmRestarts(0.05, 0.0, 10)
c2 = CosineWarmRestarts(0.05, 0.0, 10, t_mult=2)
es = np.linspace(0, 50, 500)
ax2.plot(es, [c1(float(e)) for e in es], label="t_mult=1")
ax2.plot(es, [c2(float(e)) for e in es], label="t_mult=2")
ax2.set_xlabel("epoch")
ax2.set_ylabel("learning rate")
ax2.set_title("CosineWarmRestarts(0.05, 0.0, 10)", weight="bold", fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.legend()

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=140)
print(f"written: {OUT} ({OUT.stat().st_size} B)")
print(f"best_epoch_: {tr.best_epoch_}, stopped_epoch_: {tr.stopped_epoch_}")
if tr.best_epoch_ is not None:
    best_ep = tr.best_epoch_
    print(f"val_loss at best: {tr.history_['val_loss'][best_ep]:.4f}")
