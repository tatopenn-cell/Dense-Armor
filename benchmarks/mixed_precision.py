"""Benchmark of mixed-precision training on the compiled step.

Claims, written before the first run and not changed after it:

(1) The loss of bfloat16 at step 200 is within 5% relative of
    float32 on the same stream.
(2) The peak memory of the lower precision is at most 0.7 times that
    of float32 (GPU only: ``memory_stats()`` of the device, read in a
    fresh process per mode because the peak of a process only grows; on
    the CPU the value is ``None``).
(3) The faster of bfloat16 and float16 has at least 1.3 times the
    steps per second of float32 (the script prints which one).

The model of item C: a ``transformer_encoder`` of 6 layers,
``d_model`` 512, sequence 256, batch 32. ``--smoke`` shrinks it so the
script stays under a minute on a CPU.

Speed: 20 steps per session, 3 sessions, after a warm-up step. The
loss comparison runs 200 steps of each mode and records the loss at
step 200. The JSON is ``benchmarks/mixed_precision_cpu.json`` or
``benchmarks/mixed_precision_gpu.json`` according to the backend.

The synthetic data are labelled "synthetic". Seed 0.
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dense_armor.utility.neural.losses import MSE  # noqa: E402
from dense_armor.utility.neural.models import transformer_encoder  # noqa: E402
from dense_armor.utility.neural.trainer import Trainer  # noqa: E402
from dense_armor.utility.optim import SGD  # noqa: E402

SEED = 0
N_SESSIONS = 3
N_STEPS = 20
LOSS_STEPS = 200


def _model(smoke):
    if smoke:
        return transformer_encoder(
            n_in=32,
            d_model=64,
            n_heads=4,
            d_ff=128,
            depth=2,
            n_out=8,
            key=jax.random.PRNGKey(SEED),
            max_len=32,
        ), 8
    return transformer_encoder(
        n_in=512,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        depth=6,
        n_out=512,
        key=jax.random.PRNGKey(SEED),
        max_len=256,
    ), 32


def _data(seq, batch, seed=SEED):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(batch, seq, 512 if not seq < 64 else 32)).astype(np.float32),
        rng.normal(size=(batch, 512 if not seq < 64 else 8)).astype(np.float32),
    )


def _peak_bytes():
    try:
        stats = jax.devices()[0].memory_stats()
        if stats is None:
            return None
        return int(stats.get("peak_bytes_in_use", stats.get("bytes_in_use", 0)))
    except Exception:
        return None


def _backend():
    return "gpu" if jax.default_backend() == "gpu" else "cpu"


def _make_stepper(trainer, dt, batch):
    """Return a function ``step(values, opt_state, x, y) -> (values, opt_state, loss)``.

    For ``dt is None`` uses the plain jitted step (``trainer._compiled_step.plain``).
    For ``dt`` in ``("bfloat16", "float16")`` uses the mixed step plus the same
    bookkeeping as ``Trainer.fit`` (overflow check, halving, growth,
    ``_apply_update``), returning the updated optimizer state as well. The
    unscale is inside the mixed step.
    """
    from dense_armor.utility.neural.compiled import _apply_update

    if dt is None:
        plain = trainer._compiled_step.plain

        def step(values, opt_state, x, y):
            return plain(values, opt_state, x, y, 0.0)

        return step

    mixed = trainer._compiled_step.mixed

    def step(values, opt_state, x, y):
        grads_fp16, grads_fp32, loss = mixed(values, x, y, trainer._scale)
        overflow = any(bool(jnp.any(~jnp.isfinite(g))) for g in grads_fp16)
        if overflow:
            if trainer.loss_scale == "dynamic":
                trainer._scale = max(trainer._scale / 2.0, 1.0)
            trainer._good_steps = 0
        else:
            if trainer.loss_scale == "dynamic":
                trainer._good_steps += 1
                if trainer._good_steps >= 2000:
                    trainer._scale = trainer._scale * 2.0
                    trainer._good_steps = 0
            values, opt_state = _apply_update(
                trainer.optimizer_prototype,
                values,
                grads_fp32,
                opt_state,
                trainer.schedule(0.0),
                trainer.schedule.multiplier(0.0),
                trainer.weight_decay,
                trainer.decay,
            )
        return values, opt_state, loss

    return step


def _run_one(label, compute_dtype, loss_scale, smoke, loss_steps):
    model, batch = _model(smoke)
    seq = 32 if smoke else 256
    X, y = _data(seq, batch)
    t = Trainer(
        model,
        MSE(),
        SGD(0.01),
        compile=True,
        compute_dtype=compute_dtype,
        loss_scale=loss_scale,
    )
    t.fit(X, y, epochs=1, batch_size=batch, key=jax.random.PRNGKey(SEED))
    values = t._compiled_values
    opt_state = t._compiled_opt_state
    step = _make_stepper(t, compute_dtype, batch)
    xb = jnp.asarray(X)
    yb = jnp.asarray(y)
    loss = None
    first_loss = None
    for _ in range(max(1, loss_steps)):
        values, opt_state, loss = step(values, opt_state, xb, yb)
        if first_loss is None:
            first_loss = float(loss)
    peak = _peak_bytes()
    steps = []
    for _ in range(N_SESSIONS):
        values_s = t._compiled_values
        opt_state_s = t._compiled_opt_state
        for _ in range(3):
            values_s, opt_state_s, _ = step(values_s, opt_state_s, xb, yb)
        jax.block_until_ready(values_s[0])
        t0 = time.perf_counter()
        for _ in range(N_STEPS):
            values_s, opt_state_s, _ = step(values_s, opt_state_s, xb, yb)
        jax.block_until_ready(values_s[0])
        steps.append(N_STEPS / (time.perf_counter() - t0))
    med = float(np.median(steps))
    return {
        "mode": label,
        "compute_dtype": compute_dtype,
        "loss_scale": loss_scale,
        "loss_at_step": int(loss_steps),
        "initial_loss": first_loss,
        "final_loss": float(loss),
        "peak_bytes": peak,
        "steps_per_second_median": med,
        "steps_per_second_range": [min(steps), max(steps)],
        "steps_per_second_sessions": steps,
    }


MODES = [
    ("float32", None, None),
    ("bfloat16", "bfloat16", None),
    ("float16", "float16", "dynamic"),
]


def _run_in_new_process(label, smoke, loss_steps):
    """Run one mode in a fresh process: the peak memory of a process only grows."""
    cmd = [sys.executable, __file__, "--mode", label, "--loss-steps", str(loss_steps)]
    if smoke:
        cmd.append("--smoke")
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    line = next(ln for ln in out.splitlines() if ln.startswith("ROW "))
    return json.loads(line[4:])


def main(smoke, loss_steps):
    print(f"backend: {_backend()}")
    rows = []
    for label, _dt, _ls in MODES:
        print(f"  running {label} ...")
        rows.append(_run_in_new_process(label, smoke, loss_steps))
        r = rows[-1]
        print(
            f"    loss={r['final_loss']:.6f}  "
            f"steps/s={r['steps_per_second_median']:.2f}  "
            f"peak={r['peak_bytes']}"
        )
    claims = {
        "1_bf16_loss_within_5pct": None,
        "2_memory_0.7x": None,
        "3_faster_of_bf16_fp16_1.3x": None,
    }
    rows_by = {r["mode"]: r for r in rows}
    ref = rows_by["float32"]
    if ref["final_loss"] > 0:
        bf = rows_by["bfloat16"]["final_loss"]
        claims["1_bf16_loss_within_5pct"] = {
            "holds": abs(bf - ref["final_loss"]) / ref["final_loss"] < 0.05,
            "rel_diff": abs(bf - ref["final_loss"]) / ref["final_loss"],
        }
    if ref["peak_bytes"] is not None:
        best_peak = min(
            (rows_by["bfloat16"]["peak_bytes"] or 1e18),
            (rows_by["float16"]["peak_bytes"] or 1e18),
        )
        claims["2_memory_0.7x"] = {
            "holds": best_peak / ref["peak_bytes"] <= 0.7,
            "ratio": best_peak / ref["peak_bytes"],
        }
    else:
        claims["2_memory_0.7x"] = {"holds": None, "reason": "no GPU peak memory"}
    fastest = max(
        ("bfloat16", rows_by["bfloat16"]["steps_per_second_median"]),
        ("float16", rows_by["float16"]["steps_per_second_median"]),
        key=lambda kv: kv[1],
    )
    if _backend() == "cpu":
        claims["3_faster_of_bf16_fp16_1.3x"] = {
            "holds": None,
            "reason": "GPU only: no bfloat16 acceleration on CPU",
        }
    else:
        claims["3_faster_of_bf16_fp16_1.3x"] = {
            "which": fastest[0],
            "ratio": fastest[1] / ref["steps_per_second_median"],
            "holds": fastest[1] / ref["steps_per_second_median"] >= 1.3,
        }
    payload = {
        "backend": _backend(),
        "jax": jax.__version__,
        "python": platform.python_version(),
        "processor": platform.processor(),
        "platform": platform.platform(),
        "n_sessions": N_SESSIONS,
        "n_steps_per_session": N_STEPS,
        "loss_steps": loss_steps,
        "smoke": smoke,
        "seed": SEED,
        "claims": claims,
        "rows": rows,
    }
    out = Path(__file__).parent / f"mixed_precision_{_backend()}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"JSON: {out}")
    print(f"claims: {json.dumps(claims, indent=2)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--loss-steps", type=int, default=LOSS_STEPS)
    ap.add_argument("--mode", choices=[m[0] for m in MODES])
    args = ap.parse_args()
    if args.mode is None:
        main(args.smoke, args.loss_steps)
    else:
        label, dt, ls = next(m for m in MODES if m[0] == args.mode)
        row = _run_one(label, dt, ls, args.smoke, args.loss_steps)
        print("ROW " + json.dumps(row))
