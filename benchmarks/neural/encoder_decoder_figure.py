"""Generate docs/assets/neural/encoder_decoder.png.

Five panels on synthetic data, from the library classes:

1. the reconstruction error of an Autoencoder and of PCA against the
   code size, both as the mean over the last axis of the squared error;
2. the code usage of a VectorQuantizer;
3. the accuracy of the Matryoshka heads against m: the model is
   trained on the first 60 of the 200 samples and evaluated on all 200,
   so 140 of the 200 points were never used in training. The three
   accuracies ``0.99, 0.82, 0.935`` are computed on all 200 points, of
   which 60 are train and 140 are held out;
4. a forecast of a Seq2Seq against the truth on a synthetic sinusoid:
   the printed MSE is measured on the training batch, not on held-out
   data, and the panel is a fit, not a generalisation measure;
5. the score of an OnlineAutoencoder along a stream with a fault
   injected at sample 300.

The labels say ``synthetic``. The classes are the ones of
``dense_armor.utility.neural`` and ``dense_armor.utility.vision.reduce``.
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

from dense_armor.ml.vision import PCA
from dense_armor.utility.neural.autoencoder import Autoencoder
from dense_armor.utility.neural.chain import Sequential
from dense_armor.utility.neural.layers import Dense
from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.matryoshka import MatryoshkaLoss
from dense_armor.utility.neural.online_autoencoder import OnlineAutoencoder
from dense_armor.utility.neural.quantize import VectorQuantizer
from dense_armor.utility.neural.recurrent import LSTM
from dense_armor.utility.neural.seq2seq import Seq2Seq
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import Adam

HERE = Path(__file__).resolve().parents[2]
OUT = HERE / "docs" / "assets" / "neural" / "encoder_decoder.png"

rng = np.random.default_rng(0)

# 1. reconstruction error vs code size
A = rng.normal(0.0, 1.0, (2, 6)).astype(np.float32)
Z = rng.normal(0.0, 1.0, (300, 2)).astype(np.float32)
X6 = Z @ A + 0.05 * rng.normal(0.0, 1.0, (300, 6)).astype(np.float32)

sizes = [1, 2, 3, 4]
ae_errs = []
pca_errs = []
for m in sizes:
    enc = Sequential([Dense(6, m, jax.random.PRNGKey(0))])
    dec = Sequential([Dense(m, 6, jax.random.PRNGKey(1))])
    ae = Autoencoder(enc, dec)
    t = Trainer(ae, MSE(), Adam(0.01))
    t.fit(X6, X6, epochs=30, batch_size=30, key=jax.random.PRNGKey(2))
    ae_errs.append(float(np.mean(np.asarray(ae.reconstruction_error(X6)))))

    pca = PCA(m)
    pca.fit(X6)
    rec = pca.transform(X6) @ np.asarray(pca.components_) + np.asarray(pca.mean_)
    pca_errs.append(float(np.mean(np.mean((X6 - rec) ** 2, axis=-1))))

# 2. vector quantizer code usage
enc = Sequential([Dense(6, 2, jax.random.PRNGKey(0))])
dec2 = Sequential([Dense(2, 6, jax.random.PRNGKey(3))])
vq = VectorQuantizer(4, 2, jax.random.PRNGKey(1))


class AE:
    def __init__(self):
        self.enc = enc
        self.vq = vq
        self.dec2 = dec2

    def forward(self, x, training=True):
        z = self.enc.forward(x, training=training)
        zq = self.vq.forward(z, training=training)
        return self.dec2.forward(zq, training=training)

    def backward(self, g):
        d_zq = self.dec2.backward(g)
        d_z = self.vq.backward(d_zq)
        return self.enc.backward(d_z)

    def params(self):
        return (
            list(self.enc.params()) + list(self.vq.params()) + list(self.dec2.params())
        )


ae_vq = AE()
t = Trainer(ae_vq, MSE(), Adam(0.01))
t.fit(X6, X6, epochs=50, batch_size=30, key=jax.random.PRNGKey(2))
indices = np.asarray(vq.encode_indices(enc.forward(jnp.asarray(X6), training=False)))
counts = np.array([int((indices == k).sum()) for k in range(4)])

# 3. Matryoshka accuracy vs m
means = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=np.float32)
y_m = rng.integers(0, 4, 200)
X_m = np.zeros((200, 8), dtype=np.float32)
X_m[:, :2] = means[y_m] + 0.3 * rng.normal(0.0, 1.0, (200, 2)).astype(np.float32)
X_m[:, 2:] = 0.3 * rng.normal(0.0, 1.0, (200, 6)).astype(np.float32)
ml = MatryoshkaLoss([2, 4, 8], 8, 4, jax.random.PRNGKey(0))
opts = {id(h): Adam(0.02) for h in ml.params()}
for _ in range(60):
    for h in ml.params():
        h.zero_grad()
    for i in range(60):
        ml.forward(X_m[i], y_m[i])
        ml.backward()
    for h in ml.params():
        h.update(opts[id(h)])
accs = []
for m in [2, 4, 8]:
    ok = 0
    for i in range(200):
        pred = int(jnp.argmax(ml.logits(X_m[i], m)))
        if pred == int(y_m[i]):
            ok += 1
    accs.append(ok / 200)

# 4. seq2seq forecast vs truth
n_s = 30
T = 8
H = 4
phase = rng.uniform(0.0, 2.0 * np.pi, n_s)
t_src = np.arange(T, dtype=np.float32)
t_tgt = np.arange(T, T + H, dtype=np.float32)
X_s = np.sin(t_src[None, :, None] + phase[:, None, None]) + 0.05 * rng.normal(
    0.0, 1.0, (n_s, T, 1)
).astype(np.float32)
Y_s = np.sin(t_tgt[None, :, None] + phase[:, None, None]).astype(np.float32)
s = Seq2Seq(
    LSTM(1, 4, jax.random.PRNGKey(0)),
    LSTM(4, 4, jax.random.PRNGKey(1)),
    Dense(4, 1, jax.random.PRNGKey(2)),
    horizon=H,
)
t = Trainer(s, MSE(), Adam(0.01))
t.fit(X_s, Y_s, epochs=30, batch_size=10, key=jax.random.PRNGKey(0))
pred = np.asarray(s.forward(jnp.asarray(X_s), training=False))

# 5. online autoencoder score along a stream with a fault
A2 = rng.normal(0.0, 1.0, (2, 6)).astype(np.float32)
Z2 = rng.normal(0.0, 1.0, (400, 2)).astype(np.float32)
X2 = Z2 @ A2
fault_at = 300
X2[fault_at:] += 3.0
enc2 = Sequential([Dense(6, 2, jax.random.PRNGKey(0))])
dec2 = Sequential([Dense(2, 6, jax.random.PRNGKey(1))])
oa = OnlineAutoencoder(
    Autoencoder(enc2, dec2),
    Adam(0.01),
    feature_keys=[f"p{i}" for i in range(6)],
)
scores = []
for row in X2:
    d = {f"p{i}": float(row[i]) for i in range(6)}
    scores.append(oa.score_one(d))
    oa.learn_one(d)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes[0, 0].plot(sizes, ae_errs, "o-", label="Autoencoder")
axes[0, 0].plot(sizes, pca_errs, "s--", label="PCA")
axes[0, 0].set_xlabel("code size")
axes[0, 0].set_ylabel("reconstruction error (MSE)")
axes[0, 0].set_title("Autoencoder vs PCA (synthetic)", weight="bold", fontsize=10)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend(fontsize=8)

axes[0, 1].bar(range(4), counts, color="#2c6fa6")
axes[0, 1].set_xlabel("code index")
axes[0, 1].set_ylabel("count")
axes[0, 1].set_title(
    "VectorQuantizer code usage (synthetic)", weight="bold", fontsize=10
)
axes[0, 1].grid(True, alpha=0.3, axis="y")

axes[0, 2].bar(["m=2", "m=4", "m=8"], accs, color="#a63c2c")
axes[0, 2].set_ylabel("accuracy")
axes[0, 2].set_title("Matryoshka accuracy vs m (synthetic)", weight="bold", fontsize=10)
axes[0, 2].grid(True, alpha=0.3, axis="y")

i = 0
axes[1, 0].plot(range(T, T + H), Y_s[i, :, 0], "o-", label="truth")
axes[1, 0].plot(range(T, T + H), pred[i, :, 0], "s--", label="Seq2Seq")
axes[1, 0].set_xlabel("step")
axes[1, 0].set_ylabel("value")
axes[1, 0].set_title("Seq2Seq forecast (synthetic)", weight="bold", fontsize=10)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend(fontsize=8)

axes[1, 1].plot(scores, color="#444")
axes[1, 1].axvline(fault_at, color="r", linestyle="--", label=f"fault at {fault_at}")
axes[1, 1].set_xlabel("sample")
axes[1, 1].set_ylabel("score (reconstruction error)")
axes[1, 1].set_title("OnlineAutoencoder score (synthetic)", weight="bold", fontsize=10)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend(fontsize=8)

axes[1, 2].axis("off")
axes[1, 2].text(
    0.05,
    0.5,
    f"median before fault: {np.median(scores[:fault_at]):.3f}\n"
    f"score at fault:      {scores[fault_at]:.3f}\n"
    f"median after fault:  {np.median(scores[fault_at + 20 :]):.3f}",
    fontsize=11,
    verticalalignment="center",
    family="monospace",
)

fig.suptitle("encoders and decoders on synthetic data", weight="bold", fontsize=13)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=140)
print(f"written: {OUT} ({OUT.stat().st_size} B)")
print(f"AE errs: {[round(e, 3) for e in ae_errs]}")
print(f"PCA errs: {[round(e, 3) for e in pca_errs]}")
print(f"VQ counts: {counts.tolist()}")
print(f"Matryoshka accs: {[round(a, 3) for a in accs]}")
print(f"Seq2Seq MSE forecast: {float(np.mean((pred - Y_s) ** 2)):.4f}")
print(f"median before fault: {float(np.median(scores[:fault_at])):.4f}")
print(f"score at fault: {scores[fault_at]:.4f}")
print(f"median after fault: {float(np.median(scores[fault_at + 20 :])):.4f}")
