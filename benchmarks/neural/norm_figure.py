"""Generate docs/assets/neural/norm.png.

Three histograms of the first feature of a skewed synthetic batch: the
raw batch, the same batch after BatchNorm, and the same batch after
LayerNorm. Both normalisations come from the library classes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.utility.neural.norm import BatchNorm, LayerNorm

HERE = Path(__file__).resolve().parents[2]
OUT = HERE / "docs" / "assets" / "neural" / "norm.png"

rng = np.random.default_rng(0)
n = 512
x = np.empty((n, 2), dtype=np.float32)
x[:, 0] = rng.exponential(2.0, n) + 5.0
x[:, 1] = rng.normal(0.0, 1.0, n)

bn = BatchNorm(2)
y_bn = np.asarray(bn.forward(jnp.asarray(x), training=True))

ln = LayerNorm(2)
y_ln = np.asarray(ln.forward(jnp.asarray(x)))

fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
axes[0].hist(x[:, 0], bins=40, color="#444")
axes[0].set_title("raw batch (synthetic)", weight="bold", fontsize=11)
axes[0].set_xlabel("x[:, 0]")
axes[1].hist(y_bn[:, 0], bins=40, color="#2c6fa6")
axes[1].set_title("after BatchNorm", weight="bold", fontsize=11)
axes[1].set_xlabel("y[:, 0]")
axes[2].hist(y_ln[:, 0], bins=40, color="#a63c2c")
axes[2].set_title("after LayerNorm", weight="bold", fontsize=11)
axes[2].set_xlabel("y[:, 0]")
for ax in axes:
    ax.grid(True, alpha=0.3)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=140)
print(f"written: {OUT} ({OUT.stat().st_size} B)")
print(f"raw mean {float(np.mean(x[:, 0])):.3f} median {float(np.median(x[:, 0])):.3f}")
print(
    f"bn  mean {float(np.mean(y_bn[:, 0])):.3f} median {float(np.median(y_bn[:, 0])):.3f}"
)
print(
    f"ln  mean {float(np.mean(y_ln[:, 0])):.3f} median {float(np.median(y_ln[:, 0])):.3f}"
)
