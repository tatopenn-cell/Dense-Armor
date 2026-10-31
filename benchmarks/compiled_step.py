"""Benchmark of the compiled training step.

Claims, written before the first run and not changed after it:

(1) The loss of the compiled step is within ``1e-4`` relative of the
    loop at steps 1, 10, 50 and 200, on both models.
(2) On the GPU the compiled step has at least 3.0 times the steps per
    second of the loop on both models.
(3) On the CPU at least 1.5 times.

A claim is checked at the end of the run on the hardware that produced
the JSON. A claim that does not hold is printed as ``does not hold``
with the numbers. A claim that was not measured on that hardware is
printed as ``not measured`` with the reason.

The two models of item C: a 4-layer MLP of width 512, batch 256; a
transformer encoder of 4 layers, ``d_model`` 256, sequence 128, batch
32. ``--smoke`` shrinks both so the script stays under a minute on a
CPU.

Speed: 20 steps per session, 3 sessions, after a warm-up step. The
compile time of the compiled mode is reported apart. The median and
the range of steps per second over the sessions are printed and
written in the JSON with the backend.

Loss comparison: ``--loss-steps N`` runs N steps of the loop and N
steps of the compiled mode and records the loss at steps 1, 10, 50
and N (the last one if smaller). ``--loss-steps 0`` skips the loss
comparison and declares it ``not measured`` with the reason. The CPU
run of the transformer uses 0 because 200 steps at 8 s per step would
take 27 minutes on this CPU; the GPU run uses 200.

Synthetic data, seed 0. The JSON is ``benchmarks/compiled_step_cpu.json``
or ``benchmarks/compiled_step_gpu.json`` according to the backend.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dense_armor.utility.neural.compiled import (  # noqa: E402
    init_state,
    make_step,
    read_values,
    write_values,
)
from dense_armor.utility.neural.losses import MSE  # noqa: E402
from dense_armor.utility.neural.models import (  # noqa: E402
    mlp,
    transformer_encoder,
)
from dense_armor.utility.neural.schedules import ConstantLR  # noqa: E402
from dense_armor.utility.neural.trainer import Trainer  # noqa: E402
from dense_armor.utility.optim import SGD  # noqa: E402

SEED = 0
N_SESSIONS = 3
N_STEPS = 20
LOSS_CHECKPOINTS = [1, 10, 50, 200]


def _models(smoke):
    if smoke:
        return {
            "mlp": (
                lambda: mlp([64, 64, 64], jax.random.PRNGKey(SEED)),
                (16, 64),
                (16, 64),
            ),
            "transformer_encoder": (
                lambda: transformer_encoder(
                    n_in=32, d_model=32, n_heads=4, d_ff=64, depth=1,
                    n_out=32, key=jax.random.PRNGKey(SEED), max_len=8,
                ),
                (4, 8, 32),
                (4, 32),
            ),
        }
    return {
        "mlp": (
            lambda: mlp([512, 512, 512, 512, 512], jax.random.PRNGKey(SEED)),
            (256, 512),
            (256, 512),
        ),
        "transformer_encoder": (
            lambda: transformer_encoder(
                n_in=128, d_model=256, n_heads=8, d_ff=1024, depth=4,
                n_out=128, key=jax.random.PRNGKey(SEED), max_len=128,
            ),
            (32, 128, 128),
            (32, 128),
        ),
    }


def _data(shape_x, shape_y, seed):
    kx, ky = jax.random.split(jax.random.PRNGKey(seed))
    return jax.random.normal(kx, shape_x), jax.random.normal(ky, shape_y)


def _sps_loop(model_make, X, y, batch_size, n_steps):
    m = model_make()
    t = Trainer(m, MSE(), SGD(0.01))
    n = len(X)
    n_batches = max(1, n // batch_size)
    # warm-up: one step, then time n_steps
    for p in m.params():
        p.zero_grad()
    pred = m.forward(X[0:batch_size], training=True)
    _ = t.loss.forward(pred, y[0:batch_size])
    m.backward(t.loss.backward())
    for p in m.params():
        p.update(t.optimizer_prototype)
    t0 = time.perf_counter()
    for s in range(n_steps):
        idx_start = (s % n_batches) * batch_size
        idx = slice(idx_start, idx_start + batch_size)
        for p in m.params():
            p.zero_grad()
        pred = m.forward(X[idx], training=True)
        _ = t.loss.forward(pred, y[idx])
        m.backward(t.loss.backward())
        for p in m.params():
            p.update(t.optimizer_prototype)
    dt = time.perf_counter() - t0
    return n_steps / dt


def _sps_compiled(model_make, X, y, batch_size, n_steps):
    m = model_make()
    values = read_values(m)
    write_values(m, values)
    opt = SGD(0.01)
    st = init_state(opt, values)
    step = make_step(m, MSE(), opt, ConstantLR(0.01), 0.0, "l2", None)
    n = len(X)
    n_batches = max(1, n // batch_size)
    # compile with the first batch, then time
    tc = time.perf_counter()
    values, st, _ = step(values, st, X[0:batch_size], y[0:batch_size], 0.0)
    compile_time = time.perf_counter() - tc
    write_values(m, values)
    t0 = time.perf_counter()
    for s in range(n_steps):
        idx_start = (s % n_batches) * batch_size
        idx = slice(idx_start, idx_start + batch_size)
        values, st, _ = step(values, st, X[idx], y[idx], float(s))
        write_values(m, values)
    dt = time.perf_counter() - t0
    return n_steps / dt, compile_time


def _loss_comparison(model_make, X, y, batch_size, n_steps):
    checkpoints = sorted(set(c for c in LOSS_CHECKPOINTS if c <= n_steps) | {n_steps})
    n = len(X)
    n_batches = max(1, n // batch_size)

    m1 = model_make()
    t1 = Trainer(m1, MSE(), SGD(0.01))
    losses_loop = {}
    for s in range(n_steps):
        idx_start = (s % n_batches) * batch_size
        idx = slice(idx_start, idx_start + batch_size)
        for p in m1.params():
            p.zero_grad()
        pred = m1.forward(X[idx], training=True)
        l = float(t1.loss.forward(pred, y[idx]))
        m1.backward(t1.loss.backward())
        for p in m1.params():
            p.update(t1.optimizer_prototype)
        if (s + 1) in checkpoints:
            losses_loop[s + 1] = l

    m2 = model_make()
    values = read_values(m2)
    write_values(m2, values)
    opt = SGD(0.01)
    st = init_state(opt, values)
    step = make_step(m2, MSE(), opt, ConstantLR(0.01), 0.0, "l2", None)
    losses_comp = {}
    for s in range(n_steps):
        idx_start = (s % n_batches) * batch_size
        idx = slice(idx_start, idx_start + batch_size)
        values, st, l = step(values, st, X[idx], y[idx], float(s))
        write_values(m2, values)
        if (s + 1) in checkpoints:
            losses_comp[s + 1] = float(l)

    rel = {k: abs(losses_loop[k] - losses_comp[k]) / max(abs(losses_loop[k]), 1e-12)
           for k in checkpoints}
    return losses_loop, losses_comp, rel, checkpoints


def _speed(model_make, X, y, batch_size):
    loop_sps = []
    comp_sps = []
    compile_time = 0.0
    for _ in range(N_SESSIONS):
        loop_sps.append(_sps_loop(model_make, X, y, batch_size, N_STEPS))
        sps, ct = _sps_compiled(model_make, X, y, batch_size, N_STEPS)
        comp_sps.append(sps)
        compile_time = ct
    return loop_sps, comp_sps, compile_time


def _backend_label():
    b = str(jax.default_backend())
    if b in ("gpu", "cuda"):
        return "gpu"
    if b == "cpu":
        return "cpu"
    return b


def main(smoke, models_to_run, loss_steps, out_path=None):
    label = _backend_label()
    print(f"backend: {label} (jax {jax.__version__}, "
          f"python {platform.python_version()}, {platform.processor()})")
    print(f"n_steps={N_STEPS}, n_sessions={N_SESSIONS}, seed={SEED}, "
          f"smoke={smoke}, loss_steps={loss_steps}")
    print()
    print(f"{'model':22s} {'mode':10s} {'steps/s median':>15s} "
          f"{'range':>22s} {'compile s':>10s}")
    result = {
        "backend": label,
        "jax": jax.__version__,
        "python": platform.python_version(),
        "processor": platform.processor(),
        "platform": platform.platform(),
        "n_steps": N_STEPS,
        "n_sessions": N_SESSIONS,
        "seed": SEED,
        "smoke": smoke,
        "loss_steps": loss_steps,
        "claims": {
            "1_loss_within_1e-4": None,
            "2_gpu_at_least_3x": None,
            "3_cpu_at_least_1_5x": None,
        },
        "rows": [],
    }
    all_loss_ok = True
    any_loss_measured = False
    all_gpu_ok = True
    any_gpu = False
    all_cpu_ok = True
    any_cpu = False

    for name, (make, sx, sy) in _models(smoke).items():
        if models_to_run and name not in models_to_run:
            continue
        X, y = _data(sx, sy, SEED)
        batch_size = sx[0]
        loop_sps, comp_sps, compile_time = _speed(make, X, y, batch_size)
        med_loop = float(np.median(loop_sps))
        med_comp = float(np.median(comp_sps))
        rng_loop = (float(min(loop_sps)), float(max(loop_sps)))
        rng_comp = (float(min(comp_sps)), float(max(comp_sps)))
        ratio = med_comp / med_loop if med_loop > 0 else float("inf")
        print(f"{name:22s} {'loop':10s} {med_loop:15.3f} "
              f"[{rng_loop[0]:8.2f}, {rng_loop[1]:8.2f}] {'-':>10s}")
        print(f"{'':22s} {'compiled':10s} {med_comp:15.3f} "
              f"[{rng_comp[0]:8.2f}, {rng_comp[1]:8.2f}] {compile_time:10.3f}")
        print(f"{'':22s} ratio={ratio:.3f}")
        row = {
            "model": name,
            "loop_sps_median": med_loop,
            "loop_sps_range": list(rng_loop),
            "compiled_sps_median": med_comp,
            "compiled_sps_range": list(rng_comp),
            "ratio": ratio,
            "compile_time_s": compile_time,
        }
        if label == "gpu":
            any_gpu = True
            if ratio < 3.0:
                all_gpu_ok = False
        if label == "cpu":
            any_cpu = True
            if ratio < 1.5:
                all_cpu_ok = False

        if loss_steps > 0:
            losses_loop, losses_comp, rel, checkpoints = _loss_comparison(
                make, X, y, batch_size, loss_steps
            )
            max_rel = max(rel.values())
            print(f"{'':22s} loss_rel_max={max_rel:.2e} "
                  f"at {checkpoints}")
            row["loss_loop"] = losses_loop
            row["loss_compiled"] = losses_comp
            row["loss_rel"] = rel
            row["loss_max_rel"] = max_rel
            any_loss_measured = True
            if max_rel > 1e-4:
                all_loss_ok = False
        else:
            row["loss_loop"] = None
            row["loss_compiled"] = None
            row["loss_rel"] = None
            row["loss_max_rel"] = None
            row["loss_note"] = "not measured: loss_steps=0"
        print()
        result["rows"].append(row)

    if any_loss_measured:
        result["claims"]["1_loss_within_1e-4"] = "holds" if all_loss_ok else "does not hold"
    else:
        result["claims"]["1_loss_within_1e-4"] = "not measured: loss_steps=0"
    if any_gpu:
        result["claims"]["2_gpu_at_least_3x"] = "holds" if all_gpu_ok else "does not hold"
    else:
        result["claims"]["2_gpu_at_least_3x"] = "not measured: backend is not gpu"
    if any_cpu:
        result["claims"]["3_cpu_at_least_1_5x"] = "holds" if all_cpu_ok else "does not hold"
    else:
        result["claims"]["3_cpu_at_least_1_5x"] = "not measured: backend is not cpu"

    print("claims:")
    for k, v in result["claims"].items():
        print(f"  {k}: {v}")

    if out_path is not None:
        out = Path(out_path)
    else:
        out = Path("benchmarks") / f"compiled_step_{label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"JSON: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--loss-steps", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    main(args.smoke, args.models, args.loss_steps, args.out)
