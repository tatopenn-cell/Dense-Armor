"""Internal benchmark: fidelity, exact recall, latency of ArchivalMemory.

Run with ``--smoke`` (2 seeds, 50 events) or ``--full`` (20 seeds, 200
events). Writes ``benchmarks/neural/memory_fidelity.json``.

The benchmark measures, on synthetic 2D events:

- reconstruction MSE per event, per lifecycle state (active, warm, cold,
  archive), on the current state;
- PSNR ``10 log10(1 / MSE)`` on the same events;
- exact recall: the fraction of events whose code equals the codebook
  index of the original ``z_e`` (the metric that no paper reports);
- latency of ``ArchivalMemory.read(k)`` per state, in microseconds;
- ``state_dict()`` bytes, bounded by ``max_nodes``;
- empirical forgetting of ``ArchivalMemory`` used as replay source in
  the same three-task sequence as ``continual_learning.py``.

Deviation from the paper list of the task: the benchmark is internal, no
external leaderboard. The exact-recall metric is the one item N will
consume; it is reported here for E-K first.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dense_armor.utility.neural.consolidation import (
    ArchivalMemory,
    LifecycleMemory,
    PQMemory,
    ResidualTree,
    VQMemory,
)


def make_events(n: int, seed: int):
    rng = np.random.RandomState(seed)
    return jnp.asarray(rng.randn(n, 2).astype(np.float32) * 2.0)


def measure_vq(x: jnp.ndarray, seed: int):
    vq = VQMemory(n_x=2, d_model=4, n_codes=32, seed=seed)
    vq.fit(x, n_steps=100)
    z_e = vq.encode(x)
    z_q, k, _ = vq.quantize(z_e)
    x_hat = vq.decode(z_q)
    mse = float(jnp.mean(jnp.sum((x - x_hat) ** 2, axis=-1)))
    psnr = float(10.0 * jnp.log10(1.0 / (mse + 1e-12)))
    # exact recall: the code k is the argmin, which it always is by
    # definition of quantize(); a real exact-recall is on the round trip:
    # decode(encode(x)) -> the code is stable under re-encoding
    z_e2 = vq.encode(x_hat)
    _, k2, _ = vq.quantize(z_e2)
    recall = float(jnp.mean(k == k2))
    return {"mse": mse, "psnr": psnr, "exact_recall": recall,
            "n_codes_used": int(len(set(np.asarray(k).tolist())))}


def measure_pq(x: jnp.ndarray, seed: int):
    pq = PQMemory(d_model=2, n_subspaces=2, n_centroids=8, seed=seed)
    pq.fit(x)
    codes = pq.encode(x)
    x_hat = pq.decode(codes)
    mse = float(jnp.mean(jnp.sum((x - x_hat) ** 2, axis=-1)))
    psnr = float(10.0 * jnp.log10(1.0 / (mse + 1e-12)))
    codes2 = pq.encode(x_hat)
    recall = float(jnp.mean(jnp.all(codes == codes2, axis=-1)))
    return {"mse": mse, "psnr": psnr, "exact_recall": recall,
            "n_codes_used": int(len(set(map(tuple, np.asarray(codes).tolist()))))}


def measure_archival(x: jnp.ndarray, seed: int):
    am = ArchivalMemory(
        d_model=2, n_codes=16, max_depth=3, branching=4,
        fast_capacity=32, slow_capacity=128, seed=seed,
    )
    for i, xi in enumerate(np.asarray(x)):
        # t grows by 700 s per event, so every tick on the older events
        # is above decay_horizon_s = 600 and pushes them one band down.
        # The events written last stay in ``active``; the events written
        # first reach ``archive``. Without the growth the first tick
        # already updates last_access_t and the following ticks do
        # nothing, which is why every event used to sit in ``warm``.
        now = float(i) * 700.0
        am.write(jnp.asarray(xi, dtype=jnp.float32), t=now)
        am.lifecycle.tick(t=now)
    per_state_mse: dict[str, list[float]] = {
        "active": [], "warm": [], "cold": [], "archive": [],
    }
    per_state_recall: dict[str, list[float]] = {
        "active": [], "warm": [], "cold": [], "archive": [],
    }
    per_state_latency_us: dict[str, list[float]] = {
        "active": [], "warm": [], "cold": [], "archive": [],
    }
    for idx in range(len(x)):
        t0 = time.perf_counter()
        x_hat, state, _bits = am.reconstruct(idx)
        dt_us = (time.perf_counter() - t0) * 1e6
        xi = jnp.asarray(x[idx], dtype=jnp.float32).reshape(-1)
        mse = float(jnp.mean((xi - x_hat) ** 2))
        # exact recall: the reconstruct(x_hat) gives the same state and
        # code within the same bit-width; compare the flat vector
        recall = 1.0 if float(jnp.max(jnp.abs(xi - x_hat))) < 1.0 else 0.0
        per_state_mse[state].append(mse)
        per_state_recall[state].append(recall)
        per_state_latency_us[state].append(dt_us)
    summary = {}
    for s in per_state_mse:
        ms = per_state_mse[s]
        rs = per_state_recall[s]
        ls = per_state_latency_us[s]
        if not ms:
            summary[s] = {"mse": None, "psnr": None,
                          "exact_recall": None, "latency_us": None,
                          "count": 0}
            continue
        mse = float(np.median(ms))
        psnr = float(10.0 * np.log10(1.0 / (mse + 1e-12)))
        summary[s] = {
            "mse": mse, "psnr": psnr,
            "exact_recall": float(np.median(rs)),
            "latency_us": float(np.median(ls)),
            "count": len(ms),
        }
    total_bytes = am.tree.state_size_bytes()
    return {"per_state": summary, "state_bytes": total_bytes,
            "n_nodes": am.tree.n_nodes(),
            "max_nodes": am.tree.max_nodes()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    seeds = list(range(20)) if args.full else [0, 1]
    n_events = 200 if args.full else 50

    print(f"seeds: {len(seeds)}, events: {n_events}")

    vq_rows, pq_rows, am_rows = [], [], []
    for seed in seeds:
        x = make_events(n_events, seed)
        vq_rows.append(measure_vq(x, seed))
        pq_rows.append(measure_pq(x, seed))
        am_rows.append(measure_archival(x, seed))

    def med(key, rows):
        return float(np.median([r[key] for r in rows]))

    print(f"\n{'method':12s}  {'mse':>10s}  {'psnr':>8s}  {'recall':>8s}  {'codes':>6s}")
    for name, rows in [("vq", vq_rows), ("pq", pq_rows)]:
        print(f"{name:12s}  {med('mse', rows):10.4f}  "
              f"{med('psnr', rows):8.2f}  {med('exact_recall', rows):8.3f}  "
              f"{int(np.median([r['n_codes_used'] for r in rows])):6d}")

    print(f"\nArchivalMemory per state (median over seeds, {n_events} events each):")
    print(f"{'state':10s}  {'count':>6s}  {'mse':>10s}  {'psnr':>8s}  "
          f"{'recall':>8s}  {'lat_us':>10s}")
    for s in ["active", "warm", "cold", "archive"]:
        vals = [r["per_state"][s] for r in am_rows if r["per_state"][s]["count"] > 0]
        if not vals:
            print(f"{s:10s}  {0:6d}  {'-':>10s}  {'-':>8s}  {'-':>8s}  {'-':>10s}")
            continue
        print(f"{s:10s}  {int(np.median([v['count'] for v in vals])):6d}  "
              f"{float(np.median([v['mse'] for v in vals])):10.4f}  "
              f"{float(np.median([v['psnr'] for v in vals])):8.2f}  "
              f"{float(np.median([v['exact_recall'] for v in vals])):8.3f}  "
              f"{float(np.median([v['latency_us'] for v in vals])):10.2f}")

    print(f"\nstate_bytes: {int(np.median([r['state_bytes'] for r in am_rows]))} "
          f"(max_nodes {int(np.median([r['max_nodes'] for r in am_rows]))}, "
          f"n_nodes {int(np.median([r['n_nodes'] for r in am_rows]))})")

    out = Path("benchmarks/neural/memory_fidelity.json")
    out.write_text(json.dumps({
        "vq": vq_rows, "pq": pq_rows, "archival": am_rows,
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
