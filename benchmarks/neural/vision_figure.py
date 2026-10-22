"""Generate docs/assets/neural/vision.png.

A synthetic 8x8 image with a vertical bar, the output of a Conv2D whose
kernel is set to a vertical-edge detector, and the output of a
MaxPool2D(2) on it. All three from the library classes, labelled
synthetic.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.utility.neural.conv import Conv2D, MaxPool2D

HERE = Path(__file__).resolve().parents[2]
OUT = HERE / "docs" / "assets" / "neural" / "vision.png"

rng = np.random.default_rng(0)
img = rng.normal(0.0, 0.05, (8, 8, 1)).astype(np.float32)
img[:, 3, 0] += 1.0

c = Conv2D(1, 1, 3, jax.random.PRNGKey(0))
vertical = np.array(
    [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]], dtype=np.float32
)
c.W.value = jnp.asarray(vertical).reshape(3, 3, 1, 1)
c.b.value = jnp.zeros(1)
y = np.asarray(c.forward(img)).reshape(6, 6)

p = MaxPool2D(2)
y_pool = np.asarray(p.forward(jnp.asarray(y).reshape(6, 6, 1))).reshape(3, 3)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
axes[0].imshow(img[:, :, 0], cmap="gray")
axes[0].set_title("input: 8x8, vertical bar (synthetic)", weight="bold", fontsize=10)
axes[1].imshow(y, cmap="RdBu_r")
axes[1].set_title("Conv2D 3x3, vertical-edge kernel", weight="bold", fontsize=10)
axes[2].imshow(y_pool, cmap="RdBu_r")
axes[2].set_title("MaxPool2D(2)", weight="bold", fontsize=10)
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=140)
print(f"written: {OUT} ({OUT.stat().st_size} B)")
print(f"conv output shape: {y.shape}, max: {float(y.max()):.4f}")
print(f"pool output shape: {y_pool.shape}, max: {float(y_pool.max()):.4f}")
