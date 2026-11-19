"""How many normalised layers can be stacked in float32 before the gradient is wrong, with and without stable_var.

Every layer receives activations around a large offset with a small spread, which is the case where a layer
normalisation loses the digits of the spread: ``x <- OFFSET + SPREAD * tanh(LN(x) @ W)``, with ``W`` orthogonal times
``WSCALE`` = 0.7 (the gradient still grows with depth, by a factor of about 10^15 over 1024 layers, which float32 holds). Two layer normalisations are compared in float32:

- ``usual``: ``(x - mean) / sqrt(var + eps)`` with ``jnp.mean`` and ``jnp.var``;
- ``stable``: the variance from ``stable_var`` and the mean subtracted on the data shifted by the first element.

For each depth ``L`` the forward pass runs in float32 and records the input of every layer; the gradient of
``sum(c * x_L)`` with respect to ``x_0`` is computed in float32 by ``jax.grad`` and, as a reference, in float64 by
replaying the backward pass layer by layer on the same float32 inputs. So the comparison isolates the error of
the backward pass from the rounding of the activations themselves. The relative error ``|g32 - g64| / |g64|``,
the largest error of the output of a layer normalisation (against float64 on the same input) and the time of one
forward and backward step (median of the runs, after a warm-up, ``block_until_ready``) are recorded for both. The
matrix products use the highest precision, so that the layer normalisation is the only difference.

Run:
    python benchmarks/stable_variance_depth.py            full run, writes the json next to this file
    python benchmarks/stable_variance_depth.py --small    small shapes
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

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

EPS = 1e-6
OFFSET, SPREAD = 1000.0, 0.004
WSCALE = 0.7


def ln_usual(x):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + EPS)


def ln_stable(x):
    y = x - x[..., :1]
    d = y - jnp.mean(y, axis=-1, keepdims=True)
    return d / jnp.sqrt(stable_var(x, axis=-1, keepdims=True) + EPS)


def block(x, w, ln):
    return OFFSET + SPREAD * jnp.tanh(ln(x) @ w)


def make_trajectory(ln):
    def trajectory(x0, W):
        def step(x, w):
            return block(x, w, ln), x

        return jax.lax.scan(step, x0, W)

    return trajectory


def make_grad(ln):
    traj = make_trajectory(ln)
    return jax.jit(jax.grad(lambda x0, W, c: jnp.sum(c * traj(x0, W)[0])))


def make_inputs(ln):
    traj = make_trajectory(ln)
    return jax.jit(lambda x0, W: traj(x0, W)[1])


@jax.jit
def reference_grad(xs, W, c):
    def step(g, inp):
        x, w = inp
        return jax.vjp(lambda a: block(a, w, ln_usual), x)[1](g)[0], None

    return jax.lax.scan(step, c, (xs, W), reverse=True)[0]


def orthogonal(rng, L, D):
    q = np.linalg.qr(rng.normal(0.0, 1.0, (L, D, D)))[0]
    return (WSCALE * q).astype(np.float32)


def timeit(f, args, runs):
    f(*args).block_until_ready()
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        f(*args).block_until_ready()
        ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    args = ap.parse_args()
    if args.small:
        B, D, depths, runs = 16, 64, [1, 2, 4, 8, 16, 32], 3
    else:
        B, D, depths, runs = 128, 512, [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], 11
    print(f"jax {jax.__version__}  backend {jax.default_backend()}  B {B} D {D}")
    rng = np.random.default_rng(0)
    y0 = rng.normal(OFFSET, SPREAD, (B, D)).astype(np.float32)
    c = rng.normal(0.0, 1.0, (B, D)).astype(np.float32)
    lns = {"usual": ln_usual, "stable": ln_stable}
    grads = {name: make_grad(ln) for name, ln in lns.items()}
    inputs = {name: make_inputs(ln) for name, ln in lns.items()}
    rows = []
    for L in depths:
        W = jnp.asarray(orthogonal(rng, L, D))
        a = (jnp.asarray(y0), W, jnp.asarray(c))
        row = {"depth": L}
        for name in lns:
            xs = inputs[name](a[0], W)
            g64 = np.asarray(
                reference_grad(
                    xs.astype(jnp.float64),
                    W.astype(jnp.float64),
                    a[2].astype(jnp.float64),
                )
            )
            g32 = np.asarray(grads[name](*a), np.float64)
            fwd = np.asarray(jax.vmap(lns[name])(xs), np.float64)
            fwd64 = np.asarray(jax.vmap(ln_usual)(xs.astype(jnp.float64)))
            row[name] = {
                "grad_rel_err": float(np.linalg.norm(g32 - g64) / np.linalg.norm(g64)),
                "grad_norm_reference": float(np.linalg.norm(g64)),
                "layer_output_max_err": float(np.max(np.abs(fwd - fwd64))),
                "nan_or_inf": int(np.sum(~np.isfinite(g32))),
                "ms": timeit(grads[name], a, runs),
            }
        row["ratio_time_stable_over_usual"] = row["stable"]["ms"] / row["usual"]["ms"]
        rows.append(row)
        u, s = row["usual"], row["stable"]
        print(
            L,
            f"usual {u['grad_rel_err']:.2e} out {u['layer_output_max_err']:.1e} {u['ms']:.2f} ms"
            f" | stable {s['grad_rel_err']:.2e} out {s['layer_output_max_err']:.1e} {s['ms']:.2f} ms"
            f" | x{row['ratio_time_stable_over_usual']:.2f} | |g| {u['grad_norm_reference']:.2e}",
            flush=True,
        )
    payload = {
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "python": sys.version.split()[0],
        "batch": B,
        "width": D,
        "offset": OFFSET,
        "spread": SPREAD,
        "rows": rows,
    }
    name = (
        "stable_variance_depth_small.json"
        if args.small
        else "stable_variance_depth.json"
    )
    (Path(__file__).resolve().parent / name).write_text(json.dumps(payload, indent=2))
    print("written", name)


if __name__ == "__main__":
    main()
