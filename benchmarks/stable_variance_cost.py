"""The cost of stable_var in a network step: a stack of complete transformer blocks with two layer normalisations, on a GPU.

Each block is ``x <- x + Attn(LN(x))`` then ``x <- x + gelu(LN(x) @ W1) @ W2``: a causal multi-head self-attention with the projections
``[D, 3D]`` and ``[D, D]``, and the feed-forward part with ``W1`` of shape ``[D, 4D]`` and ``W2`` of shape ``[4D, D]``. Two layer normalisations are compared in float32: the usual one, with ``jnp.mean`` and
``jnp.var``, and the one with ``stable_var`` (the variance of ``stable_var``, the mean subtracted on the data shifted by the first
element). What is timed is the forward and backward pass of ``mean(x_L ** 2)`` with respect to all the weights, which is a training
step, the median of the runs after a warm-up, with ``block_until_ready``. The normalisation alone, forward and backward on one
``[batch, seq, D]`` array, is timed too, so that the share of the normalisation in a step can be read: ``2 L`` times the time of one
normalisation (there are two in a block) over the time of the step.

Run:
    python benchmarks/stable_variance_cost.py            full run, writes the json next to this file
    python benchmarks/stable_variance_cost.py --small    small shapes
"""

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dense_armor.core.signals.stats.stable import stable_var  # noqa: E402

EPS = 1e-6


def ln_usual(x):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + EPS)


def ln_stable(x):
    y = x - x[..., :1]
    d = y - jnp.mean(y, axis=-1, keepdims=True)
    return d / jnp.sqrt(stable_var(x, axis=-1, keepdims=True) + EPS)


def make_step(ln, heads):
    def loss(params, x0):
        B, T, D = x0.shape
        dh = D // heads
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))

        def block(x, w):
            wqkv, wo, w1, w2 = w
            q, k, v = jnp.split(ln(x) @ wqkv, 3, axis=-1)
            q, k, v = (a.reshape(B, T, heads, dh) for a in (q, k, v))
            sc = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(jnp.float32(dh))
            att = jax.nn.softmax(jnp.where(mask, sc, -1e9), axis=-1)
            ctx = jnp.einsum("bhts,bshd->bthd", att, v).reshape(B, T, D)
            x = x + ctx @ wo
            return x + jax.nn.gelu(ln(x) @ w1) @ w2, None

        x, _ = jax.lax.scan(block, x0, params)
        return jnp.mean(x**2)

    return jax.jit(jax.grad(loss))


def make_ln_only(ln):
    return jax.jit(jax.grad(lambda x: jnp.sum(ln(x) ** 2)))


def timeit(f, args, runs):
    jax.block_until_ready(f(*args))
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        jax.block_until_ready(f(*args))
        ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    args = ap.parse_args()
    if args.small:
        B, T, D, H, L, runs = 2, 8, 32, 4, 2, 3
    else:
        B, T, D, H, L, runs = 8, 512, 1024, 16, 8, 7
    print(
        f"jax {jax.__version__}  backend {jax.default_backend()}  batch {B} seq {T} D {D} heads {H} L {L}",
        flush=True,
    )
    rng = np.random.default_rng(0)
    x0 = jnp.asarray(rng.normal(0.0, 1.0, (B, T, D)).astype(np.float32))
    wqkv = jnp.asarray(rng.normal(0.0, D**-0.5, (L, D, 3 * D)).astype(np.float32))
    wo = jnp.asarray(rng.normal(0.0, D**-0.5, (L, D, D)).astype(np.float32))
    w1 = jnp.asarray(rng.normal(0.0, D**-0.5, (L, D, 4 * D)).astype(np.float32))
    w2 = jnp.asarray(rng.normal(0.0, (4 * D) ** -0.5, (L, 4 * D, D)).astype(np.float32))
    params = (wqkv, wo, w1, w2)
    out = {}
    for name, ln in (("usual", ln_usual), ("stable", ln_stable)):
        step = make_step(ln, H)
        g = step(params, x0)
        ok = bool(all(np.all(np.isfinite(np.asarray(a))) for a in g))
        out[name] = {
            "step_ms": timeit(step, (params, x0), runs),
            "ln_ms": timeit(make_ln_only(ln), (x0,), runs),
            "finite_gradient": ok,
        }
        print(name, out[name], flush=True)
    for name in out:
        out[name]["ln_share_of_step"] = (
            2 * L * out[name]["ln_ms"] / out[name]["step_ms"]
        )
    out["ratio_step_stable_over_usual"] = (
        out["stable"]["step_ms"] / out["usual"]["step_ms"]
    )
    out["ratio_ln_stable_over_usual"] = out["stable"]["ln_ms"] / out["usual"]["ln_ms"]
    payload = {
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "python": sys.version.split()[0],
        "batch": B,
        "seq": T,
        "heads": H,
        "width": D,
        "hidden": 4 * D,
        "layers": L,
        "runs": runs,
        "result": out,
    }
    print(json.dumps(out, indent=1))
    name = (
        "stable_variance_cost_small.json" if args.small else "stable_variance_cost.json"
    )
    (Path(__file__).resolve().parent / name).write_text(json.dumps(payload, indent=2))
    print("written", name)


if __name__ == "__main__":
    main()
