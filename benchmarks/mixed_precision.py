"""Benchmark of mixed-precision training on the compiled step.

Claims, written before the first run and not changed after it:

(1) The loss of bfloat16 at step 200 is within 5% relative of
    float32 on the same stream.
(2) The peak memory of the lower precision is at most 0.7 times that
    of float32 (GPU only: ``memory_stats()`` of the device; on the CPU
    the value is ``None``).
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


def _step_fn(trainer):
    """A function ``step(values, x, y)`` that does one full trainer step.

    It mirrors ``Trainer.fit``: overflow check, unscale, schedule,
    ``_apply_update``. Returns the new values and the loss.
    """
    from dense_armor.utility.neural.compiled import _apply_update

    def step(values, x, y):
        grads, loss = trainer._compiled_mixed(values, x, y)
        overflow = any(bool(jnp.any(~jnp.isfinite(g))) for g in grads)
        if not overflow:
            if trainer.loss_scale is not None and trainer._scale != 1.0:
                grads = [g / trainer._scale for g in grads]
            values, _ = _apply_update(
                trainer.optimizer_prototype,
                values,
                grads,
                trainer._compiled_opt_state,
                trainer.schedule(0.0),
                trainer.schedule.multiplier(0.0),
                trainer.weight_decay,
                trainer.decay,
            )
        return values, loss

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
    step = _step_fn(t)
    xb = jnp.asarray(X)
    yb = jnp.asarray(y)
    loss = None
    for _ in range(max(1, loss_steps)):
        values, loss = step(values, xb, yb)
    peak = _peak_bytes()
    steps = []
    for _ in range(N_SESSIONS):
        values_s = t._compiled_values
        for _ in range(3):
            values_s, _ = step(values_s, xb, yb)
        jax.block_until_ready(values_s[0])
        t0 = time.perf_counter()
        for _ in range(N_STEPS):
            values_s, _ = step(values_s, xb, yb)
        jax.block_until_ready(values_s[0])
        steps.append(N_STEPS / (time.perf_counter() - t0))
    med = float(np.median(steps))
    return {
        "mode": label,
        "compute_dtype": compute_dtype,
        "loss_scale": loss_scale,
        "loss_at_step": int(loss_steps),
        "final_loss": float(loss),
        "peak_bytes": peak,
        "steps_per_second_median": med,
        "steps_per_second_range": [min(steps), max(steps)],
        "steps_per_second_sessions": steps,
    }


def main(smoke, loss_steps):
    print(f"backend: {_backend()}")
    rows = []
    for label, dt, ls in [
        ("float32", "float32", 1.0),
        ("bfloat16", "bfloat16", 1.0),
        ("float16", "float16", "dynamic"),
    ]:
        print(f"  running {label} ...")
        rows.append(_run_one(label, dt, ls, smoke, loss_steps))
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
    args = ap.parse_args()
    main(args.smoke, args.loss_steps)
