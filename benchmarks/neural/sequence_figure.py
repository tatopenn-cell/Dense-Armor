"""Generate docs/assets/neural/sequence.png.

Left: the attention weights of a MultiHeadSelfAttention on a
synthetic sequence of 8 steps, averaged over the heads. Right: the
norm of the hidden state of a GRU and of an LSTM along the same
synthetic sequence. All three from the library classes, labelled
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

from dense_armor.utility.neural.attention import MultiHeadSelfAttention
from dense_armor.utility.neural.recurrent import GRU, LSTM

HERE = Path(__file__).resolve().parents[2]
OUT = HERE / "docs" / "assets" / "neural" / "sequence.png"

rng = np.random.default_rng(0)
seq = rng.normal(0.0, 1.0, (8, 4)).astype(np.float32)
seq[3] += 2.0

mhsa = MultiHeadSelfAttention(4, 2, jax.random.PRNGKey(0))
mhsa.forward(jnp.asarray(seq))
w = np.asarray(mhsa._cache["Q"] @ jnp.swapaxes(mhsa._cache["K"], -1, -2))[0]
d_k = w.shape[-1]
scores = w / np.sqrt(d_k)
weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
weights = weights / weights.sum(axis=-1, keepdims=True)
weights_avg = weights.mean(axis=0)

gru = GRU(4, 4, jax.random.PRNGKey(1))
y_gru = np.asarray(gru.forward(jnp.asarray(seq)))
n_gru = np.linalg.norm(y_gru, axis=-1)

lstm = LSTM(4, 4, jax.random.PRNGKey(2))
y_lstm = np.asarray(lstm.forward(jnp.asarray(seq)))
n_lstm = np.linalg.norm(y_lstm, axis=-1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
im = ax1.imshow(weights_avg, cmap="viridis", aspect="equal")
ax1.set_title(
    "attention weights, mean over 2 heads (synthetic)", weight="bold", fontsize=10
)
ax1.set_xlabel("key position")
ax1.set_ylabel("query position")
fig.colorbar(im, ax=ax1, fraction=0.046)

ax2.plot(n_gru, "o-", label="GRU")
ax2.plot(n_lstm, "s-", label="LSTM")
ax2.set_title(
    "hidden-state norm along the sequence (synthetic)", weight="bold", fontsize=10
)
ax2.set_xlabel("step")
ax2.set_ylabel("||h_t||")
ax2.grid(True, alpha=0.3)
ax2.legend()

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=140)
print(f"written: {OUT} ({OUT.stat().st_size} B)")
print(f"attention row 0 sums to: {float(weights_avg[0].sum()):.4f}")
print(f"GRU final norm: {float(n_gru[-1]):.4f}, LSTM: {float(n_lstm[-1]):.4f}")
