"""stable_var with bfloat16 and float8 values: the variance of what is stored, and the time of a step.

Part 1, the variance. Values ``N(offset, spread)`` are stored in a low precision type (bfloat16, float8_e4m3fn, float8_e5m2: the float8
types are only a storage type here, the matrix products are not done in float8). The reference is the float64 variance of the stored
values. Three ways to compute the variance of the stored values are compared, all returned in float32 and compared with the reference:
``low`` (``jnp.var`` on the low precision array), ``f32`` (the usual practice: the array is cast to float32 and ``jnp.var`` is applied),
and ``stable`` (``stable_var`` on the array cast to float32 first: stable_var itself raises a TypeError on bfloat16 and float8_e4m3fn). The relative error is the median over the rows.
What cannot be done by any of them is to give back a wiggle that the storage has already lost: the reference is the variance of the
stored values, so that case is not counted against the methods.

Part 2, the time. A stack of complete transformer blocks (attention and feed-forward) with the weights and the activations in
bfloat16, the forward and backward pass with respect to the weights, with the usual layer normalisation and with the one with
``stable_var``, the two run in turn in each round, the median of the runs. On a GPU without native bfloat16 the time is not the one
of a GPU or a TPU that has it.

Run:
    python benchmarks/stable_variance_lowprec.py            full run, writes the json next to this file
    python benchmarks/stable_variance_lowprec.py --small    small shapes
"""

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dense_armor.core.signals.stats.stable import stable_var  # noqa: E402

EPS = 1e-6
TYPES = {
    "bfloat16": ml_dtypes.bfloat16,
    "float8_e4m3fn": ml_dtypes.float8_e4m3fn,
    "float8_e5m2": ml_dtypes.float8_e5m2,
}


def variance_part(rows, D, offsets, spread):
    out = []
    rng = np.random.default_rng(0)
    for name, dt in TYPES.items():
        for offset in offsets:
            x64 = rng.normal(offset, spread, (rows, D))
            stored = x64.astype(dt)
            ref = np.var(stored.astype(np.float64), axis=1)
            xs = jnp.asarray(stored)
            res = {
                "low": np.asarray(jnp.var(xs, axis=1), np.float64),
                "f32": np.asarray(jnp.var(xs.astype(jnp.float32), axis=1), np.float64),
                "stable": np.asarray(
                    stable_var(xs.astype(jnp.float32), axis=1), np.float64
                ),
            }
            row = {
                "type": name,
                "offset": offset,
                "spread": spread,
                "distinct_values": int(len(np.unique(stored[0]))),
            }
            for k, v in res.items():
                ok = ref > 0
                err = (
                    np.abs(v[ok] - ref[ok]) / ref[ok]
                    if ok.any()
                    else np.array([np.nan])
                )
                row[k] = float(np.median(err))
            row["reference_variance_median"] = float(np.median(ref))
            out.append(row)
            print(row, flush=True)
    return out


def ln_usual(x):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + EPS)


def ln_stable(x):
    y = x - x[..., :1]
    d = y - jnp.mean(y, axis=-1, keepdims=True)
    return d / jnp.sqrt(
        stable_var(x.astype(jnp.float32), axis=-1, keepdims=True).astype(x.dtype) + EPS
    )


def make_step(ln, heads):
    def loss(params, x0):
        B, T, D = x0.shape
        dh = D // heads
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))

        def block(x, w):
            wqkv, wo, w1, w2 = w
            q, k, v = jnp.split(ln(x) @ wqkv, 3, axis=-1)
            q, k, v = (a.reshape(B, T, heads, dh) for a in (q, k, v))
            sc = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(jnp.float32(dh)).astype(
                x.dtype
            )
            att = jax.nn.softmax(jnp.where(mask, sc, -1e9), axis=-1)
            ctx = jnp.einsum("bhts,bshd->bthd", att, v).reshape(B, T, D)
            x = x + ctx @ wo
            return x + jax.nn.gelu(ln(x) @ w1) @ w2, None

        x, _ = jax.lax.scan(block, x0, params)
        return jnp.mean(x.astype(jnp.float32) ** 2)

    return jax.jit(jax.grad(loss))


def timeit_pair(fa, fb, args, runs):
    jax.block_until_ready(fa(*args))
    jax.block_until_ready(fb(*args))
    ta, tb = [], []
    for _ in range(runs):
        t = time.perf_counter()
        jax.block_until_ready(fa(*args))
        ta.append((time.perf_counter() - t) * 1e3)
        t = time.perf_counter()
        jax.block_until_ready(fb(*args))
        tb.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ta)), float(np.median(tb))


def time_part(B, T, D, H, L, runs):
    rng = np.random.default_rng(0)
    bf = jnp.bfloat16
    x0 = jnp.asarray(rng.normal(0.0, 1.0, (B, T, D)), bf)
    params = (
        jnp.asarray(rng.normal(0.0, D**-0.5, (L, D, 3 * D)), bf),
        jnp.asarray(rng.normal(0.0, D**-0.5, (L, D, D)), bf),
        jnp.asarray(rng.normal(0.0, D**-0.5, (L, D, 4 * D)), bf),
        jnp.asarray(rng.normal(0.0, (4 * D) ** -0.5, (L, 4 * D, D)), bf),
    )
    fu, fs = make_step(ln_usual, H), make_step(ln_stable, H)
    ok = {
        n: bool(
            all(np.all(np.isfinite(np.asarray(a, np.float32))) for a in f(params, x0))
        )
        for n, f in (("usual", fu), ("stable", fs))
    }
    tu, ts = timeit_pair(fu, fs, (params, x0), runs)
    res = {
        "step_ms_usual": tu,
        "step_ms_stable": ts,
        "ratio": ts / tu,
        "finite_gradient": ok,
    }
    print(res, flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    args = ap.parse_args()
    print(
        f"jax {jax.__version__}  backend {jax.default_backend()}  devices {jax.devices()}",
        flush=True,
    )
    if args.small:
        rows, D, offsets, tshape = 8, 128, [0.0, 100.0], (2, 8, 32, 4, 2, 3)
    else:
        rows, D, offsets, tshape = (
            64,
            4096,
            [0.0, 1.0, 10.0, 100.0, 1000.0],
            (2, 512, 1024, 16, 2, 9),
        )
    payload = {
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "python": sys.version.split()[0],
        "variance": variance_part(rows, D, offsets, 1.0),
        "time": time_part(*tshape),
    }
    name = (
        "stable_variance_lowprec_small.json"
        if args.small
        else "stable_variance_lowprec.json"
    )
    (Path(__file__).resolve().parent / name).write_text(json.dumps(payload, indent=2))
    print("written", name)


if __name__ == "__main__":
    main()
