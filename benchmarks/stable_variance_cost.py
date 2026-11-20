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
    python benchmarks/stable_variance_cost.py --scaling  the widths 512, 1024, 2048 and 4096: the share of the normalisation
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


def make_step(ln, heads, grad=True):
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

    return jax.jit(jax.grad(loss) if grad else loss)


def make_ln_only(ln):
    return jax.jit(jax.grad(lambda x: jnp.sum(ln(x) ** 2)))


def timeit_pair(fa, fb, args_a, args_b, runs):
    """Median time of two functions, run one after the other in every round, so that a drift of the GPU hits both."""
    jax.block_until_ready(fa(*args_a))
    jax.block_until_ready(fb(*args_b))
    ta, tb = [], []
    for _ in range(runs):
        t = time.perf_counter()
        jax.block_until_ready(fa(*args_a))
        ta.append((time.perf_counter() - t) * 1e3)
        t = time.perf_counter()
        jax.block_until_ready(fb(*args_b))
        tb.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ta)), float(np.median(tb))


def measure(B, T, D, H, L, runs):
    rng = np.random.default_rng(0)
    x0 = jnp.asarray(rng.normal(0.0, 1.0, (B, T, D)).astype(np.float32))
    wqkv = jnp.asarray(rng.normal(0.0, D**-0.5, (L, D, 3 * D)).astype(np.float32))
    wo = jnp.asarray(rng.normal(0.0, D**-0.5, (L, D, D)).astype(np.float32))
    w1 = jnp.asarray(rng.normal(0.0, D**-0.5, (L, D, 4 * D)).astype(np.float32))
    w2 = jnp.asarray(rng.normal(0.0, (4 * D) ** -0.5, (L, 4 * D, D)).astype(np.float32))
    params = (wqkv, wo, w1, w2)
    lns = {"usual": ln_usual, "stable": ln_stable}
    step = {n: make_step(ln, H) for n, ln in lns.items()}
    fwd = {n: make_step(ln, H, grad=False) for n, ln in lns.items()}
    lno = {n: make_ln_only(ln) for n, ln in lns.items()}
    ok = {
        n: bool(all(np.all(np.isfinite(np.asarray(a))) for a in step[n](params, x0)))
        for n in lns
    }
    s_u, s_s = timeit_pair(
        step["usual"], step["stable"], (params, x0), (params, x0), runs
    )
    f_u, f_s = timeit_pair(
        fwd["usual"], fwd["stable"], (params, x0), (params, x0), runs
    )
    l_u, l_s = timeit_pair(lno["usual"], lno["stable"], (x0,), (x0,), runs)
    out = {}
    for n, st, fw, ln_ms in (("usual", s_u, f_u, l_u), ("stable", s_s, f_s, l_s)):
        out[n] = {
            "step_ms": st,
            "forward_ms": fw,
            "backward_ms": st - fw,
            "ln_ms": ln_ms,
            "ln_share_of_step": 2 * L * ln_ms / st,
            "finite_gradient": ok[n],
        }
        print(n, out[n], flush=True)
    out["ratio_step_stable_over_usual"] = s_s / s_u
    out["ratio_forward_stable_over_usual"] = f_s / f_u
    out["ratio_backward_stable_over_usual"] = (s_s - f_s) / (s_u - f_u)
    out["ratio_ln_stable_over_usual"] = l_s / l_u
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    ap.add_argument(
        "--scaling",
        action="store_true",
        help="four widths, 512 to 4096, to see the share of the normalisation fall",
    )
    args = ap.parse_args()
    if args.small:
        B, T, D, H, L, runs = 2, 8, 32, 4, 2, 3
    else:
        B, T, D, H, L, runs = 8, 512, 1024, 16, 8, 7
    print(
        f"jax {jax.__version__}  backend {jax.default_backend()}  batch {B} seq {T} D {D} heads {H} L {L}",
        flush=True,
    )
    if args.scaling:
        widths = [32, 64] if args.small else [512, 1024, 2048, 4096]
        rows = []
        for w in widths:
            res = measure(
                2,
                512 if not args.small else 8,
                w,
                max(w // 64, 2),
                2,
                9 if not args.small else 3,
            )
            rows.append({"width": w, "heads": max(w // 64, 2), "result": res})
            print(
                "width",
                w,
                "step ratio",
                res["ratio_step_stable_over_usual"],
                "share",
                res["usual"]["ln_share_of_step"],
                flush=True,
            )
        payload = {
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(d) for d in jax.devices()],
            "python": sys.version.split()[0],
            "scaling": rows,
        }
        name = (
            "stable_variance_cost_scaling_small.json"
            if args.small
            else "stable_variance_cost_scaling.json"
        )
        (Path(__file__).resolve().parent / name).write_text(
            json.dumps(payload, indent=2)
        )
        print("written", name)
        return
    out = measure(B, T, D, H, L, runs)
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
