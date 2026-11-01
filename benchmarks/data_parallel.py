"""Benchmark of data-parallel training on the compiled step.

Claims, written before the first run and not changed after it:

(1) The loss on 2 GPUs is within 1e-4 relative of the loss on 1 GPU at
    steps 1, 10, 50 and 200 (same global batch, same seed).
(2) 2 GPUs have at least 1.5 times the steps per second of 1 GPU.

The model: a ``transformer_encoder`` of 6 layers, ``d_model`` 512, sequence
256, global batch 64, on synthetic data (seed 0), with SGD at learning rate
``0.01``. One step per epoch, so the loss of an epoch is the loss of a step.
``--smoke`` shrinks the model so the script stays under a minute on a CPU,
where the devices are virtual.

Speed: 20 steps per session, 3 sessions, after the run of the loss steps
that compiles the step. Each device count runs in a process of its own and
the parent never touches JAX: the devices a process sees are fixed when
JAX starts. The JSON is ``benchmarks/data_parallel_<backend>.json``.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

SEED = 0
N_SESSIONS = 3
N_STEPS = 20
LOSS_STEPS = 200
GLOBAL_BATCH = 64
CHECK_STEPS = (1, 10, 50, 200)
DEVICE_COUNTS = (1, 2)


def _config(smoke):
    if smoke:
        return dict(
            n_in=32, d_model=64, n_heads=4, d_ff=128, depth=2, n_out=8, seq=32, batch=8
        )
    return dict(
        n_in=512,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        depth=6,
        n_out=512,
        seq=256,
        batch=GLOBAL_BATCH,
    )


def _peak_bytes(jax):
    try:
        stats = jax.devices()[0].memory_stats()
        if stats is None:
            return None
        return int(stats.get("peak_bytes_in_use", stats.get("bytes_in_use", 0)))
    except Exception:
        return None


def _run_one(n_devices, smoke, loss_steps):
    import jax
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dense_armor.utility.neural.losses import MSE
    from dense_armor.utility.neural.models import transformer_encoder
    from dense_armor.utility.neural.trainer import Trainer
    from dense_armor.utility.optim import SGD

    cfg = _config(smoke)
    model = transformer_encoder(
        n_in=cfg["n_in"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        depth=cfg["depth"],
        n_out=cfg["n_out"],
        key=jax.random.PRNGKey(SEED),
        max_len=cfg["seq"],
    )
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(cfg["batch"], cfg["seq"], cfg["n_in"])).astype(np.float32)
    y = rng.normal(size=(cfg["batch"], cfg["n_out"])).astype(np.float32)
    devices = None if n_devices == 1 else n_devices
    trainer = Trainer(model, MSE(), SGD(0.01), compile=True, devices=devices)
    key = jax.random.PRNGKey(SEED)
    trainer.fit(
        X, y, epochs=loss_steps, batch_size=cfg["batch"], key=key, shuffle=False
    )
    losses = trainer.history_["train_loss"]
    at = {str(s): float(losses[s - 1]) for s in CHECK_STEPS if s <= len(losses)}
    peak = _peak_bytes(jax)
    steps = []
    for _ in range(N_SESSIONS):
        t0 = time.perf_counter()
        trainer.fit(
            X, y, epochs=N_STEPS, batch_size=cfg["batch"], key=key, shuffle=False
        )
        steps.append(N_STEPS / (time.perf_counter() - t0))
    return {
        "n_devices": n_devices,
        "backend": "gpu"
        if jax.default_backend() in ("gpu", "cuda", "rocm")
        else jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "device_count_seen": jax.device_count(),
        "jax": jax.__version__,
        "global_batch": cfg["batch"],
        "loss_at_step": at,
        "peak_bytes_device0": peak,
        "steps_per_second_median": float(sorted(steps)[len(steps) // 2]),
        "steps_per_second_range": [min(steps), max(steps)],
        "steps_per_second_sessions": steps,
    }


def _run_in_new_process(n_devices, smoke, loss_steps):
    cmd = [
        sys.executable,
        __file__,
        "--devices",
        str(n_devices),
        "--loss-steps",
        str(loss_steps),
    ]
    if smoke:
        cmd.append("--smoke")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(n_devices))
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(
        max(DEVICE_COUNTS)
    )
    done = subprocess.run(cmd, capture_output=True, text=True, env=env)
    lines = [ln for ln in done.stdout.splitlines() if ln.startswith("ROW ")]
    if done.returncode != 0 or not lines:
        return {"n_devices": n_devices, "error": (done.stderr or done.stdout)[-400:]}
    return json.loads(lines[0][4:])


def _claims(rows):
    ok = {r["n_devices"]: r for r in rows if "error" not in r}
    claims = {
        "1_loss_within_1e-4": {"holds": None, "reason": "a run failed"},
        "2_speedup_1.5x": {"holds": None, "reason": "a run failed"},
    }
    if 1 not in ok or 2 not in ok:
        return claims
    one, two = ok[1], ok[2]
    rels = {}
    for step, base in one["loss_at_step"].items():
        if step in two["loss_at_step"]:
            rels[step] = abs(two["loss_at_step"][step] - base) / abs(base)
    claims["1_loss_within_1e-4"] = {
        "holds": max(rels.values()) < 1e-4,
        "rel_diff_by_step": rels,
    }
    if two["backend"] == "cpu":
        claims["2_speedup_1.5x"] = {
            "holds": None,
            "reason": "accelerator only: the virtual CPU devices share the same cores",
        }
    else:
        ratio = two["steps_per_second_median"] / one["steps_per_second_median"]
        claims["2_speedup_1.5x"] = {"ratio": ratio, "holds": ratio >= 1.5}
    return claims


def main(smoke, loss_steps):
    rows = []
    for n in DEVICE_COUNTS:
        print(f"  running {n} device(s) ...")
        rows.append(_run_in_new_process(n, smoke, loss_steps))
        r = rows[-1]
        if "error" in r:
            print(f"    FAILED: {r['error']}")
            continue
        print(
            f"    steps/s={r['steps_per_second_median']:.3f}  "
            f"loss={r['loss_at_step']}  seen={r['device_count_seen']}"
        )
    done = [r for r in rows if "error" not in r]
    if not done:
        raise SystemExit("every run failed")
    backend = done[0]["backend"]
    claims = _claims(rows)
    payload = {
        "backend": backend,
        "device_kind": done[0]["device_kind"],
        "jax": done[0]["jax"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "n_sessions": N_SESSIONS,
        "n_steps_per_session": N_STEPS,
        "loss_steps": loss_steps,
        "smoke": smoke,
        "seed": SEED,
        "synthetic": True,
        "claims": claims,
        "rows": rows,
    }
    out = Path(__file__).parent / f"data_parallel_{backend}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"JSON: {out}")
    print(f"claims: {json.dumps(claims, indent=2)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--loss-steps", type=int, default=LOSS_STEPS)
    ap.add_argument("--devices", type=int)
    args = ap.parse_args()
    if args.devices is None:
        main(args.smoke, args.loss_steps)
    else:
        print("ROW " + json.dumps(_run_one(args.devices, args.smoke, args.loss_steps)))
