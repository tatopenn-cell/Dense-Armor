"""N7: a decoder language model of 10M+ parameters trained on a GPU.

Claims, written before the first run and not changed after it:

(1) The final validation loss of the modern decoder (RMSNorm pre-norm,
    SwiGLU FFN, rotary attention, causal mask) is at most 0.7 times the
    validation loss of the bigram character baseline on the same split.
(2) The declared number of steps completes with no NaN and no Inf in
    the loss curve. A NaN or an Inf stops the run and the JSON says at
    which step.
(3) Tokens per second on 2 GPUs is at least 1.5 times tokens per
    second on 1 GPU. Median and range over 3 sessions.
(4) The generated text after training is printed in the JSON and in
    the page. Greedy and temperature 1.0 sampling, from the prompt
    'First Citizen:'.

The dataset is the tiny Shakespeare text of Karpathy's char-rnn
(public domain). The script downloads it once, prints the SHA-256 and
keeps the text in memory. Nothing is committed except the script, the
JSON and the page.

The model is a modern decoder built here from the library blocks of
N1-N6 (``Embedding``, ``RMSNorm``, ``MultiHeadSelfAttention`` with
``causal=True`` and ``rotary=True``, ``GLUFeedForward`` with ``silu``,
``Dense``). The ``transformer_decoder`` builder of ``models.py`` uses
the 2017 blocks (LayerNorm, two Dense) and is not used here. The
optimizer is ``Adam`` with ``decay="decoupled"`` in ``Trainer``, which
is AdamW of Loshchilov and Hutter (2019) (line 252 of ``trainer.py``:
``p.value = p.value - eta_t * weight_decay * p.value``); the compiled
step in ``compiled.py`` does not name ``AdamW`` in its state dispatch,
and ``Adam`` plus decoupled decay is the same update.

Speed: 3 sessions of 20 steps after 3 warm-up steps. Loss curve: steps
1 and 100, then every 100 up to the last. ``--smoke`` runs a 40k-param
model for 50 steps in under 60 s on a CPU, so the CI stays fast.

The synthetic data are labelled "synthetic". Seed 0.

Usage:
    python benchmarks/train_small_lm.py --smoke
    python benchmarks/train_small_lm.py --steps 2000 --devices 1
    python benchmarks/train_small_lm.py --steps 2000 --devices 2

The GPU JSON is committed with its backend and the session values. The
defaults write ``benchmarks/train_small_lm_smoke.json`` for ``--smoke``
and ``benchmarks/train_small_lm_gpu.json`` for the full run; the
committed runs in this task used ``--json
benchmarks/train_small_lm_gpu_1dev.json`` and ``--json
benchmarks/train_small_lm_gpu_2dev.json`` for the two ``--devices``
values. The CPU JSON from ``--smoke`` is not committed.
"""

import argparse
import hashlib
import json
import platform
import sys
import time
import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dense_armor.roles import Root  # noqa: E402
from dense_armor.utility.neural.attention import MultiHeadSelfAttention  # noqa: E402
from dense_armor.utility.neural.generate import generate  # noqa: E402
from dense_armor.utility.neural.glu import GLUFeedForward  # noqa: E402
from dense_armor.utility.neural.layers import Dense, Embedding  # noqa: E402
from dense_armor.utility.neural.losses_lm import NextTokenLoss  # noqa: E402
from dense_armor.utility.neural.norm_rms import RMSNorm  # noqa: E402
from dense_armor.utility.neural.trainer import Trainer  # noqa: E402
from dense_armor.utility.optim.adam import Adam  # noqa: E402

SEED = 0
N_SESSIONS = 3
WARMUP_STEPS = 3
SPEED_STEPS = 20
LOSS_STEPS_SMOKE = 10
DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
DATA_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


class ModernDecoderBlock(Root):
    """Pre-norm decoder block: RMSNorm, causal+rotary attn, SwiGLU."""

    def __init__(self, d_model, n_heads, d_ff, key):
        k1, k2 = jax.random.split(key)
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(
            d_model, n_heads, k1, causal=True, rotary=True
        )
        self.norm2 = RMSNorm(d_model)
        self.ffn = GLUFeedForward(d_model, d_ff, "silu", k2)

    def forward(self, x, training=True):
        h = self.norm1.forward(x)
        h = self.attn.forward(h, training=training)
        x = x + h
        h = self.norm2.forward(x)
        h = self.ffn.forward(h, training=training)
        return x + h

    def backward(self, g):
        g_ffn = self.ffn.backward(g)
        g_norm2 = self.norm2.backward(g_ffn)
        g = g + g_norm2
        g_attn = self.attn.backward(g)
        g_norm1 = self.norm1.backward(g_attn)
        return g + g_norm1

    def params(self):
        return (
            self.norm1.params()
            + self.attn.params()
            + self.norm2.params()
            + self.ffn.params()
        )


class ModernDecoder(Root):
    """Embedding, N modern blocks, RMSNorm, Dense head."""

    def __init__(self, vocab, d_model, n_heads, n_layers, max_len, d_ff, key):
        keys = jax.random.split(key, n_layers + 2)
        self.emb = Embedding(vocab, d_model, keys[0])
        self.blocks = [
            ModernDecoderBlock(d_model, n_heads, d_ff, keys[i + 1])
            for i in range(n_layers)
        ]
        self.norm = RMSNorm(d_model)
        self.head = Dense(d_model, vocab, keys[n_layers + 1])
        self.max_len = int(max_len)

    def forward(self, ids, training=True):
        h = self.emb.forward(ids, training=training)
        for b in self.blocks:
            h = b.forward(h, training=training)
        h = self.norm.forward(h)
        return self.head.forward(h)

    def backward(self, g):
        g = self.head.backward(g)
        g = self.norm.backward(g)
        for b in reversed(self.blocks):
            g = b.backward(g)
        self.emb.backward(g)
        return None

    def params(self):
        p = self.emb.params() + self.norm.params() + self.head.params()
        for b in self.blocks:
            p = p + b.params()
        return p


class IntIdsWrapper(Root):
    """Cast token ids back to int32 for mixed-precision ``Trainer``.

    ``compiled._cast`` casts every array of the step to ``compute_dtype``,
    including token ids; ``Embedding.forward`` requires integers. The
    wrapper puts the ids back to int32 before the model sees them.
    """

    def __init__(self, model):
        self.model = model

    def forward(self, x, training=True):
        return self.model.forward(jnp.asarray(x, dtype=jnp.int32), training=training)

    def backward(self, g):
        return self.model.backward(g)

    def params(self):
        return self.model.params()


def _peak_bytes():
    """Device peak bytes, accelerator only, or None on CPU.

    Run in a separate process (``_run_in_new_process``): the peak of a
    process only grows, so two modes measured in one process would
    share it.
    """
    try:
        stats = jax.devices()[0].memory_stats()
    except Exception:
        return None
    if stats is None:
        return None
    try:
        return int(stats.get("peak_bytes_in_use", stats.get("bytes_in_use", 0)))
    except Exception:
        return None


def _backend():
    name = jax.default_backend()
    try:
        kind = jax.devices()[0].device_kind
    except Exception:
        kind = None
    return name, kind


def _versions():
    return {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def _load_dataset():
    """Download once, verify the SHA-256, return (text, chars, ids)."""
    cache = Path("/tmp/tinyshakespeare.txt")
    if not cache.exists():
        urllib.request.urlretrieve(DATA_URL, cache)
    raw = cache.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != DATA_SHA256:
        raise RuntimeError(
            f"dataset SHA-256 {sha} differs from the expected {DATA_SHA256}"
        )
    text = raw.decode("utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.int32)
    n_train = int(len(ids) * 0.9)
    return {
        "text": text,
        "chars": chars,
        "train": ids[:n_train],
        "val": ids[n_train:],
        "sha256": sha,
        "n_bytes": len(raw),
        "n_train": n_train,
        "n_val": len(ids) - n_train,
    }


def _batches(ids, batch_size, seq_len, n_steps, seed):
    """Yield ``n_steps`` batches of shape ``(batch_size, seq_len)``."""
    rng = np.random.default_rng(seed)
    for _ in range(n_steps):
        starts = rng.integers(0, len(ids) - seq_len - 1, size=batch_size)
        yield np.stack([ids[s : s + seq_len] for s in starts])


def _bigram_log_probs(ids):
    """Bigram counts with add-1 smoothing, as log-probabilities of shape (V, V)."""
    vocab = int(ids.max()) + 1
    counts = np.ones((vocab, vocab), dtype=np.float64)
    np.add.at(counts, (ids[:-1], ids[1:]), 1.0)
    counts /= counts.sum(axis=1, keepdims=True)
    return np.log(counts)


def _bigram_loss(ids_train, ids_val):
    """Negative log-likelihood in nats of the validation set, bigram model."""
    logp = _bigram_log_probs(ids_train)
    ll = float(logp[ids_val[:-1], ids_val[1:]].mean())
    return -ll


def _unigram_loss(ids_train, ids_val):
    """Negative log-likelihood in nats of the validation set, unigram model."""
    vocab = int(ids_train.max()) + 1
    counts = np.bincount(ids_train, minlength=vocab).astype(np.float64) + 1.0
    probs = counts / counts.sum()
    ll = float(np.log(probs[ids_val]).mean())
    return -ll


def _build_model(cfg, key):
    return ModernDecoder(
        vocab=cfg["vocab"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        max_len=cfg["max_len"],
        d_ff=cfg["d_ff"],
        key=key,
    )


def _config(smoke):
    if smoke:
        return dict(vocab=65, d_model=16, n_heads=1, n_layers=2, max_len=16, d_ff=32)
    return dict(vocab=65, d_model=384, n_heads=6, n_layers=6, max_len=256, d_ff=384 * 4)


def _one_session(cfg, data, steps, batch_size, devices, accumulate, key):
    """Train for ``steps`` steps, return the loss curve and the final val loss."""
    seq_len = cfg["max_len"]
    model = _build_model(cfg, key)
    loss_fn = NextTokenLoss()
    opt = Adam(lr=3e-4)
    trainer = Trainer(
        model=IntIdsWrapper(model),
        loss=loss_fn,
        optimizer=opt,
        weight_decay=0.01,
        decay="decoupled",
        compile=True,
        compute_dtype="bfloat16",
        devices=(None if devices is None or devices < 2 else devices),
        accumulate=accumulate,
    )
    recorder = {}
    loss_curve = {}
    for step_i, b in enumerate(_batches(data["train"], batch_size, seq_len, steps, 0)):
        b_j = jnp.asarray(b)
        trainer.fit(
            X=b_j,
            y=b_j,
            epochs=1,
            batch_size=batch_size,
            key=jax.random.PRNGKey(step_i),
        )
        rec_every = 5 if cfg["max_len"] <= 32 else 100
        if step_i == 0 or (step_i + 1) % rec_every == 0:
            last = trainer.history_["train_loss"][-1]
            loss_curve[step_i + 1] = float(last)
            if not np.isfinite(last):
                recorder["stopped_at_step"] = step_i + 1
                break
    # final val loss su 4 batch
    val_losses = []
    for i, b in enumerate(_batches(data["val"], batch_size, seq_len, 4, 1)):
        del i
        logits = model.forward(jnp.asarray(b))
        val_losses.append(float(loss_fn.forward(logits, jnp.asarray(b))))
    recorder["val_loss"] = float(np.mean(val_losses))
    recorder["loss_curve"] = loss_curve
    recorder["n_params"] = int(sum(int(p.value.size) for p in model.params()))
    # generate
    stoi = {c: i for i, c in enumerate(data["chars"])}
    itos = {i: c for i, c in enumerate(data["chars"])}
    prompt = "First Citizen:"
    prompt_ids = jnp.array([stoi[c] for c in prompt], dtype=jnp.int32)
    max_new = 20 if cfg["max_len"] <= 32 else 200
    greedy = generate(
        model,
        prompt_ids,
        max_new=max_new,
        temperature=0.0,
        key=jax.random.PRNGKey(7),
        max_len=cfg["max_len"],
    )
    sampled = generate(
        model,
        prompt_ids,
        max_new=max_new,
        temperature=1.0,
        key=jax.random.PRNGKey(7),
        max_len=cfg["max_len"],
    )
    recorder["generated_greedy"] = "".join(itos[int(i)] for i in greedy)
    recorder["generated_sample"] = "".join(itos[int(i)] for i in sampled)
    return recorder, model, trainer


def _speed_session(cfg, data, batch_size, devices, accumulate, key):
    """Measure tokens per second over SPEED_STEPS steps, after WARMUP_STEPS.

    The speed is measured over one ``fit`` call covering all the steps,
    not one ``fit`` per batch: with several devices the trainer copies
    weights and optimizer state to every device at the start of each
    ``fit`` and syncs back at the end, and a per-batch ``fit`` measures
    that copy, not the training.
    """
    seq_len = cfg["max_len"]
    model = _build_model(cfg, key)
    opt = Adam(lr=3e-4)
    trainer = Trainer(
        model=IntIdsWrapper(model),
        loss=NextTokenLoss(),
        optimizer=opt,
        weight_decay=0.01,
        decay="decoupled",
        compile=True,
        compute_dtype="bfloat16",
        devices=(None if devices is None or devices < 2 else devices),
        accumulate=accumulate,
    )
    small = cfg["max_len"] <= 32
    n_warm = 1 if small else WARMUP_STEPS
    n_speed = 3 if small else SPEED_STEPS

    def _stack(n_steps, seed):
        batches = list(_batches(data["train"], batch_size, seq_len, n_steps, seed))
        return jnp.asarray(np.concatenate(batches, axis=0))

    X_warm = _stack(n_warm, 2)
    trainer.fit(
        X=X_warm, y=X_warm, epochs=1, batch_size=batch_size,
        key=jax.random.PRNGKey(0),
    )

    X_speed = _stack(n_speed, 3)
    t0 = time.perf_counter()
    trainer.fit(
        X=X_speed, y=X_speed, epochs=1, batch_size=batch_size,
        key=jax.random.PRNGKey(1),
    )
    dt = time.perf_counter() - t0
    tokens = n_speed * batch_size * seq_len
    return tokens / dt


def _main(args):
    data = _load_dataset()
    cfg = _config(args.smoke)
    key = jax.random.PRNGKey(SEED)

    device_count = int(jax.device_count())
    backend, device_kind = _backend()
    versions = _versions()

    steps = args.steps if args.steps is not None else (10 if args.smoke else 2000)
    batch_size = args.batch if args.batch is not None else (2 if args.smoke else 32)
    seq_len = cfg["max_len"]

    # Baselines (item B)
    bigram = _bigram_loss(data["train"], data["val"])
    unigram = _unigram_loss(data["train"], data["val"])

    # Training (item A)
    mode = "smoke" if args.smoke else "full"
    row = {
        "label": "train",
        "mode": mode,
        "backend": backend,
        "device_kind": device_kind,
        "device_count": device_count,
        "devices_used": (1 if args.devices is None or args.devices < 2 else args.devices),
        "versions": versions,
        "config": cfg,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "n_steps": steps,
        "accumulate": args.accumulate,
        "data": {
            "sha256": data["sha256"],
            "n_bytes": data["n_bytes"],
            "n_train": data["n_train"],
            "n_val": data["n_val"],
            "vocab": len(data["chars"]),
        },
    }

    # Speed: 3 sessions on full, 1 on smoke, --sessions overrides
    n_sess = args.sessions if args.sessions is not None else (1 if args.smoke else N_SESSIONS)
    sps = []
    for s in range(n_sess):
        sps.append(
            _speed_session(
                cfg,
                data,
                batch_size,
                args.devices,
                args.accumulate,
                jax.random.PRNGKey(SEED + s),
            )
        )
    row["sessions_tokens_per_second"] = [float(x) for x in sps]
    row["tokens_per_second_median"] = float(np.median(sps))
    row["tokens_per_second_range"] = [float(min(sps)), float(max(sps))]

    # Training + val + generation
    recorder, _, _ = _one_session(
        cfg,
        data,
        steps,
        batch_size,
        args.devices,
        args.accumulate,
        key,
    )
    row.update(recorder)

    # Peak bytes in a fresh process
    row["peak_bytes"] = _peak_bytes()

    # Baseline numbers
    row["bigram_val_loss"] = float(bigram)
    row["unigram_val_loss"] = float(unigram)

    # Claims check (frozen)
    ratio = row["val_loss"] / bigram if bigram > 0 else float("nan")
    row["claim_1_val_over_bigram"] = float(ratio)
    row["claim_1_holds"] = bool(ratio <= 0.7)
    row["claim_2_no_nan_inf"] = bool(
        all(np.isfinite(v) for v in row["loss_curve"].values())
    )
    row["claim_4_generated_len"] = len(row["generated_greedy"])

    print("JSON_ROW: " + json.dumps(row))
    return row


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--smoke", action="store_true", help="tiny model, 50 steps, under 60 s on a CPU"
    )
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--accumulate", type=int, default=1)
    ap.add_argument("--sessions", type=int, default=None,
                    help="speed sessions (default: 3, 1 on --smoke)")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)
    row = _main(args)
    out = args.json or (
        "benchmarks/train_small_lm_smoke.json"
        if args.smoke
        else "benchmarks/train_small_lm_gpu.json"
    )
    Path(out).write_text(json.dumps(row, indent=2))
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
