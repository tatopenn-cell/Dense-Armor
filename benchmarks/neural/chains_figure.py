"""Generate docs/assets/neural/chains.png.

Left: the decision regions of mlp([2, 16, 16, 1]) trained on a
synthetic two-moons set made by a function of this script. Right: the
loss against the step. Both from the library classes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.utility.neural.losses import (
    bce_with_logits_backward,
    bce_with_logits_forward,
)
from dense_armor.utility.neural.models import mlp
from dense_armor.utility.optim.adam import Adam

HERE = Path(__file__).resolve().parents[2]
OUT = HERE / "docs" / "assets" / "neural" / "chains.png"


def two_moons(n=300, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal([-0.5, 0.5], 0.2, (n // 2, 2))
    b = rng.normal([0.5, -0.5], 0.2, (n // 2, 2))
    X = np.vstack([a, b]).astype(np.float32)
    y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(np.float32)
    return X, y


X, y = two_moons(300, seed=0)
model = mlp([2, 16, 16, 1], jax.random.PRNGKey(0))
opts = {id(p): Adam(lr=0.01) for p in model.params()}
rng = np.random.default_rng(0)
losses = []
for step in range(200):
    model.zero_grad()
    perm = rng.permutation(len(X))
    total = 0.0
    for i in perm:
        logit = model.forward(jnp.asarray(X[i]))[0]
        total += float(bce_with_logits_forward(jnp.array([logit]), jnp.array([y[i]])))
        g = bce_with_logits_backward(jnp.array([logit]), jnp.array([y[i]]))[0]
        model.backward(jnp.array([g]))
    for p in model.params():
        p.update(opts[id(p)])
    losses.append(total / len(X))

xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 80), np.linspace(-1.5, 1.5, 80))
grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
Z = np.array([float(model.forward(jnp.asarray(g))[0]) for g in grid])
Z = Z.reshape(xx.shape)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
ax1.contourf(xx, yy, Z > 0, levels=[-0.5, 0.5, 1.5], colors=["#f4c2a1", "#a1c2f4"])
ax1.scatter(X[y == 0, 0], X[y == 0, 1], s=8, c="#a63c2c", label="class 0")
ax1.scatter(X[y == 1, 0], X[y == 1, 1], s=8, c="#2c6fa6", label="class 1")
ax1.set_title("mlp([2,16,16,1]), synthetic two moons", weight="bold", fontsize=11)
ax1.legend(fontsize=8)
ax2.plot(losses, color="#2c6fa6")
ax2.set_xlabel("step")
ax2.set_ylabel("mean BCE loss")
ax2.set_title("training loss (synthetic)", weight="bold", fontsize=11)
ax2.grid(True, alpha=0.3)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=140)
print(f"written: {OUT} ({OUT.stat().st_size} B)")
print(f"final loss: {losses[-1]:.4f}")
