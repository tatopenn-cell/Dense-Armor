"""Generate docs/assets/neural/activations.png.

Two rows: the five activations on x in [-4, 4], then their derivatives
from the code backward. The figure comes from the library functions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.utility.neural.activations import (
    prelu_backward,
    prelu_forward,
    relu_backward,
    relu_forward,
    sigmoid_backward,
    sigmoid_forward,
    softmax_forward,
    tanh_backward,
    tanh_forward,
)

HERE = Path(__file__).resolve().parents[2]
OUT = HERE / "docs" / "assets" / "neural" / "activations.png"

x = jnp.linspace(-4.0, 4.0, 400)
g = jnp.ones_like(x)

y_relu = np.asarray(relu_forward(x))
d_relu = np.asarray(relu_backward(x, y_relu, g))

y_prelu = np.asarray(prelu_forward(x, 0.25))
d_prelu = np.asarray(prelu_backward(x, y_prelu, g, 0.25))

y_sig = np.asarray(sigmoid_forward(x))
d_sig = np.asarray(sigmoid_backward(x, y_sig, g))

y_tanh = np.asarray(tanh_forward(x))
d_tanh = np.asarray(tanh_backward(x, y_tanh, g))

y_softmax_1d = np.asarray(softmax_forward(jnp.stack([x, jnp.zeros_like(x)], axis=-1)))[
    :, 0
]
y_softmax_1d_neg = 1.0 - y_softmax_1d

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

ax1.plot(np.asarray(x), y_relu, label="relu")
ax1.plot(np.asarray(x), y_prelu, label="prelu(a=0.25)")
ax1.plot(np.asarray(x), y_sig, label="sigmoid")
ax1.plot(np.asarray(x), y_tanh, label="tanh")
ax1.plot(np.asarray(x), y_softmax_1d, label="softmax[0] vs 0")
ax1.set_xlabel("x")
ax1.set_ylabel("f(x)")
ax1.set_title("activations", weight="bold", fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=8)

ax2.plot(np.asarray(x), d_relu, label="relu'")
ax2.plot(np.asarray(x), d_prelu, label="prelu'(a=0.25)")
ax2.plot(np.asarray(x), d_sig, label="sigmoid'")
ax2.plot(np.asarray(x), d_tanh, label="tanh'")
ax2.set_xlabel("x")
ax2.set_ylabel("f'(x)")
ax2.set_title("derivatives from backward", weight="bold", fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=140)
print(f"written: {OUT} ({OUT.stat().st_size} B)")
