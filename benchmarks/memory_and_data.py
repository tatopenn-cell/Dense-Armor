"""Benchmark of gradient checkpointing, gradient accumulation and the data stream.

Claims, written before the first run and not changed after it:

(1) The peak memory with ``remat=True`` is at most 0.7 times the peak
    without it.
(2) The time per step with ``remat=True`` is at most 1.5 times the time
    without it.
(3) The accumulation of 4 micro-batches of 8 equals one batch of 32 on the
    GPU too: the weights after 10 updates agree within 1e-5 relative to the
    largest weight of each array.

The model of claims (1) and (2): a ``transformer_encoder`` of 12 layers,
``d_model`` 512, sequence 256, batch 32, on synthetic data (seed 0), with
SGD at learning rate ``0.01``. One step per epoch. Each configuration runs
in a process of its own, because the peak memory of a process only grows,
and the parent never touches JAX. The speed is the median of 3 sessions of
20 steps after the steps that compile. The throughput of the stream is
measured on a memory-mapped file of synthetic samples and compared with the
slices of the same array in memory. ``--smoke`` shrinks everything so the
script stays under a minute on a CPU. The JSON is
``benchmarks/memory_and_data_<backend>.json``.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SEED = 0
N_SESSIONS = 3
N_STEPS = 20
WARM_STEPS = 3
UPDATES = 10


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
        depth=12,
        n_out=512,
        seq=256,
        batch=32,
    )


def _peak_bytes(jax):
    try:
        stats = jax.devices()[0].memory_stats()
        if stats is None:
            return None
        return int(stats.get("peak_bytes_in_use", stats.get("bytes_in_use", 0)))
    except Exception:
        return None


def _backend(jax):
    name = jax.default_backend()
    return "gpu" if name in ("gpu", "cuda", "rocm") else name


def _imports():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import jax
    import numpy as np

    from dense_armor.core.deep.neural.losses import MSE
    from dense_armor.core.deep.neural.models import mlp, transformer_encoder
    from dense_armor.core.deep.neural.stream import DataStream
    from dense_armor.core.deep.neural.trainer import Trainer
    from dense_armor.core.online.optim import SGD

    return jax, np, MSE, mlp, transformer_encoder, DataStream, Trainer, SGD


def _run_remat(remat, smoke):
    jax, np, MSE, _mlp, transformer_encoder, _stream, Trainer, SGD = _imports()
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
        remat=remat,
    )
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(cfg["batch"], cfg["seq"], cfg["n_in"])).astype(np.float32)
    y = rng.normal(size=(cfg["batch"], cfg["n_out"])).astype(np.float32)
    trainer = Trainer(model, MSE(), SGD(0.01), compile=True)
    key = jax.random.PRNGKey(SEED)
    trainer.fit(
        X, y, epochs=WARM_STEPS, batch_size=cfg["batch"], key=key, shuffle=False
    )
    peak = _peak_bytes(jax)
    sessions = []
    for _ in range(N_SESSIONS):
        t0 = time.perf_counter()
        trainer.fit(
            X, y, epochs=N_STEPS, batch_size=cfg["batch"], key=key, shuffle=False
        )
        sessions.append(N_STEPS / (time.perf_counter() - t0))
    return {
        "kind": "remat",
        "remat": remat,
        "backend": _backend(jax),
        "device_kind": jax.devices()[0].device_kind,
        "jax": jax.__version__,
        "peak_bytes": peak,
        "steps_per_second_median": float(sorted(sessions)[len(sessions) // 2]),
        "steps_per_second_range": [min(sessions), max(sessions)],
        "steps_per_second_sessions": sessions,
    }


def _run_stream(smoke):
    jax, np, _MSE, _mlp, _enc, DataStream, _Trainer, _SGD = _imports()
    n, shape, batch = (512, (8, 16), 32) if smoke else (6400, (256, 64), 32)
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(n,) + shape).astype(np.float32)
    y = rng.normal(size=(n, 1)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        np.save(Path(tmp) / "x.npy", X)
        np.save(Path(tmp) / "y.npy", y)

        def count(batches):
            t0 = time.perf_counter()
            served = sum(len(yb) for _, yb in batches)
            return served / (time.perf_counter() - t0)

        def in_memory():
            return ((X[i : i + batch], y[i : i + batch]) for i in range(0, n, batch))

        paths = (Path(tmp) / "x.npy", Path(tmp) / "y.npy")
        rates = {}
        for name, make in (
            ("array_in_memory", in_memory),
            ("stream_memory_map", lambda: iter(DataStream(paths, batch))),
            (
                "stream_memory_map_shuffled",
                lambda: iter(DataStream(paths, batch, 512, jax.random.PRNGKey(SEED))),
            ),
        ):
            runs = [count(make()) for _ in range(N_SESSIONS)]
            rates[name] = {
                "samples_per_second_median": float(sorted(runs)[len(runs) // 2]),
                "samples_per_second_range": [min(runs), max(runs)],
            }
    return {
        "kind": "stream",
        "backend": _backend(jax),
        "n_samples": n,
        "sample_shape": list(shape),
        "batch": batch,
        "rates": rates,
    }


def _run_accumulation():
    jax, np, MSE, mlp, _enc, _stream, Trainer, SGD = _imports()
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(64, 8)).astype(np.float32)
    y = rng.normal(size=(64, 1)).astype(np.float32)

    def weights(batch, accumulate):
        model = mlp([8, 16, 1], jax.random.PRNGKey(SEED))
        trainer = Trainer(model, MSE(), SGD(0.05), compile=True, accumulate=accumulate)
        trainer.fit(
            X,
            y,
            epochs=UPDATES // 2,
            batch_size=batch,
            key=jax.random.PRNGKey(SEED),
            shuffle=False,
        )
        return [np.asarray(p.value) for p in model.params()]

    whole, split = weights(32, 1), weights(8, 4)
    worst = max(
        float(np.max(np.abs(a - b)) / np.max(np.abs(a))) for a, b in zip(whole, split)
    )
    return {
        "kind": "accumulation",
        "backend": _backend(jax),
        "updates": UPDATES,
        "max_relative_difference": worst,
    }


def _child(mode, smoke):
    if mode == "remat_off":
        return _run_remat(False, smoke)
    if mode == "remat_on":
        return _run_remat(True, smoke)
    if mode == "stream":
        return _run_stream(smoke)
    return _run_accumulation()


MODES = ("remat_off", "remat_on", "stream", "accumulation")


def _run_in_new_process(mode, smoke):
    cmd = [sys.executable, __file__, "--mode", mode]
    if smoke:
        cmd.append("--smoke")
    done = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ))
    lines = [ln for ln in done.stdout.splitlines() if ln.startswith("ROW ")]
    if done.returncode != 0 or not lines:
        return {"mode": mode, "error": (done.stderr or done.stdout)[-400:]}
    row = json.loads(lines[0][4:])
    row["mode"] = mode
    return row


def _claims(rows):
    ok = {r["mode"]: r for r in rows if "error" not in r}
    claims = {
        "1_peak_memory_0.7x": {"holds": None, "reason": "a run failed"},
        "2_time_per_step_1.5x": {"holds": None, "reason": "a run failed"},
        "3_accumulation_1e-5": {"holds": None, "reason": "a run failed"},
    }
    off, on = ok.get("remat_off"), ok.get("remat_on")
    if off and on:
        if off["peak_bytes"] is not None and on["peak_bytes"] is not None:
            ratio = on["peak_bytes"] / off["peak_bytes"]
            claims["1_peak_memory_0.7x"] = {"ratio": ratio, "holds": ratio <= 0.7}
        else:
            claims["1_peak_memory_0.7x"] = {
                "holds": None,
                "reason": "no peak memory on this device",
            }
        ratio = off["steps_per_second_median"] / on["steps_per_second_median"]
        claims["2_time_per_step_1.5x"] = {"ratio": ratio, "holds": ratio <= 1.5}
    acc = ok.get("accumulation")
    if acc:
        worst = acc["max_relative_difference"]
        claims["3_accumulation_1e-5"] = {
            "max_relative_difference": worst,
            "holds": worst <= 1e-5,
        }
    return claims


def main(smoke):
    rows = []
    for mode in MODES:
        print(f"  running {mode} ...")
        rows.append(_run_in_new_process(mode, smoke))
        if "error" in rows[-1]:
            print(f"    FAILED: {rows[-1]['error']}")
    done = [r for r in rows if "error" not in r]
    if not done:
        raise SystemExit("every run failed")
    backend = done[0]["backend"]
    payload = {
        "backend": backend,
        "device_kind": next(
            (r["device_kind"] for r in done if "device_kind" in r), None
        ),
        "jax": next((r["jax"] for r in done if "jax" in r), None),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "n_sessions": N_SESSIONS,
        "n_steps_per_session": N_STEPS,
        "smoke": smoke,
        "seed": SEED,
        "synthetic": True,
        "claims": _claims(rows),
        "rows": rows,
    }
    out = Path(__file__).parent / f"memory_and_data_{backend}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"JSON: {out}")
    print(f"claims: {json.dumps(payload['claims'], indent=2)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--mode", choices=MODES)
    args = ap.parse_args()
    if args.mode is None:
        main(args.smoke)
    else:
        print("ROW " + json.dumps(_child(args.mode, args.smoke)))
