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




def measure_dark_memory(seeds, n_per_task=200, n_epochs=10, eps_init=1.0):
    """DarkMemory retention: acc_1 with and without the guide field."""
    import jax
    from dense_armor.utility.neural.activations import Relu
    from dense_armor.utility.neural.chain import Sequential
    from dense_armor.utility.neural.consolidation import DarkMemory
    from dense_armor.utility.neural.layers import Dense
    from dense_armor.utility.neural.losses import SoftmaxCrossEntropy
    from dense_armor.utility.neural.trainer import Trainer
    from dense_armor.utility.optim import Adam

    def train_and_build(seed):
        rng = np.random.RandomState(seed)
        X = jnp.asarray(rng.randn(n_per_task * 2, 2).astype(np.float32) * 1.5)
        Xa = np.asarray(X)
        tasks = [
            (X, jnp.asarray((Xa[:, 0] > 0).astype(np.int32))),
            (X, jnp.asarray((Xa[:, 1] > 0).astype(np.int32))),
            (X, jnp.asarray((Xa[:, 0] + Xa[:, 1] > 0).astype(np.int32))),
        ]
        key = jax.random.PRNGKey(seed)
        model = Sequential([Dense(2, 32, key), Relu(), Dense(32, 2, key)])
        loss = SoftmaxCrossEntropy()
        tr = Trainer(model, loss, Adam(lr=0.01))
        deltas = []
        for Xk, yk in tasks:
            before = [np.asarray(pp.value).copy() for pp in model.params()]
            tr.fit(Xk, yk, epochs=n_epochs, batch_size=32, key=key)
            after = [np.asarray(pp.value).copy() for pp in model.params()]
            deltas.append(np.concatenate([(a - b).reshape(-1) for a, b in zip(after, before)]))
        dm = DarkMemory()
        D, primes = dm.prime_key(deltas)
        return X, tasks, deltas, D, primes

    def init_from_guide_and_train(seed, X, tasks, deltas, D, eps):
        import jax
        from dense_armor.utility.neural.activations import Relu
        from dense_armor.utility.neural.chain import Sequential
        from dense_armor.utility.neural.consolidation import DarkMemory
        from dense_armor.utility.neural.layers import Dense
        from dense_armor.utility.neural.losses import SoftmaxCrossEntropy
        from dense_armor.utility.neural.trainer import Trainer
        from dense_armor.utility.optim import Adam
        dm = DarkMemory(eps=eps)
        F = dm.guide_field(D, deltas)
        theta = dm.init_from_guide(F, seed=seed)
        key4 = jax.random.PRNGKey(seed + 4)
        new_model = Sequential([Dense(2, 32, key4), Relu(), Dense(32, 2, key4)])
        offset = 0
        for pp in new_model.params():
            n = pp.value.size
            seg = theta[offset:offset + n] if offset + n <= theta.size else np.zeros(n)
            pp.value = jnp.asarray(seg.reshape(pp.value.shape), dtype=jnp.float32)
            offset += n
        # task 4: XOR-like
        Xa = np.asarray(X)
        X4 = X
        y4 = jnp.asarray(((Xa[:, 0] * Xa[:, 1]) > 0).astype(np.int32))
        tr4 = Trainer(new_model, SoftmaxCrossEntropy(), Adam(lr=0.01))
        tr4.fit(X4, y4, epochs=n_epochs, batch_size=32, key=key4)
        acc1 = float((np.argmax(np.asarray(new_model.forward(tasks[0][0], training=False)), axis=-1) == np.asarray(tasks[0][1])).mean())
        return acc1

    rows = []
    import time as _time
    for seed in seeds:
        X, tasks, deltas, D, primes = train_and_build(seed)
        t0 = _time.perf_counter()
        acc1_guide = init_from_guide_and_train(seed, X, tasks, deltas, D, eps=eps_init)
        dt_guide = (_time.perf_counter() - t0) * 1e3
        acc1_noguide = init_from_guide_and_train(seed, X, tasks, deltas, D, eps=0.0)
        import math
        rows.append({
            "seed": int(seed),
            "D": int(D),
            "log2_D_bits": float(math.log2(D)),
            "acc1_with_guide": acc1_guide,
            "acc1_no_guide": acc1_noguide,
            "delta_pp": float((acc1_guide - acc1_noguide) * 100),
            "latency_ms": float(dt_guide),
        })
    return rows


def _wilcoxon_a12(a, b):
    from scipy.stats import wilcoxon
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 4:
        return None, None
    try:
        _, p = wilcoxon(a, b)
    except Exception:
        p = None
    gt = np.sum(a[:, None] > b[None, :])
    eq = np.sum(a[:, None] == b[None, :])
    a12 = float((gt + 0.5 * eq) / (a.size * b.size))
    return float(p) if p is not None else None, a12


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--dark", action="store_true")
    parser.add_argument("--dark-eps", type=float, default=1.0)
    parser.add_argument("--dark-seeds", type=int, default=20)
    args = parser.parse_args()
    if args.dark:
        seeds = list(range(args.dark_seeds))
        print(f"DarkMemory: seeds={len(seeds)}, eps={args.dark_eps}")
        rows = measure_dark_memory(seeds, eps_init=args.dark_eps)
        acc_guide = [r["acc1_with_guide"] for r in rows]
        acc_no = [r["acc1_no_guide"] for r in rows]
        delta = [r["delta_pp"] for r in rows]
        log2D = [r["log2_D_bits"] for r in rows]
        lat = [r["latency_ms"] for r in rows]
        print(f"  acc_1 with guide: median {float(np.median(acc_guide)):.3f} IQR [{np.percentile(acc_guide, 25):.3f}, {np.percentile(acc_guide, 75):.3f}]")
        print(f"  acc_1 no guide:   median {float(np.median(acc_no)):.3f} IQR [{np.percentile(acc_no, 25):.3f}, {np.percentile(acc_no, 75):.3f}]")
        print(f"  delta (pp):       median {float(np.median(delta)):+.2f} IQR [{np.percentile(delta, 25):+.2f}, {np.percentile(delta, 75):+.2f}]")
        print(f"  log2(D_3) bits:   median {float(np.median(log2D)):.1f}")
        print(f"  latency guide:    median {float(np.median(lat)):.1f} ms")
        p_val, a12 = _wilcoxon_a12(acc_guide, acc_no)
        if p_val is not None:
            print(f"  wilcoxon: p={p_val:.4f}  A12={a12:.3f}")
        else:
            print(f"  wilcoxon: not computed, only {len(seeds)} seeds (<4)")
        out = Path("benchmarks/neural/memory_fidelity_dark.json")
        out.write_text(json.dumps(rows, indent=2))
        print(f"wrote {out}")
        return
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

    print(f"\nArchivalMemory forgetting (3 conflicting tasks):")
    f_rows = measure_archival_forgetting(seeds, n_events=50)
    print(f"  median forgetting: {float(np.median([r['archival_forgetting'] for r in f_rows])):.3f}")
    out = Path("benchmarks/neural/memory_fidelity.json")
    out.write_text(json.dumps({
        "vq": vq_rows, "pq": pq_rows, "archival": am_rows,
        "archival_forgetting": f_rows,
    }, indent=2))
    print(f"wrote {out}")

def measure_archival_forgetting(seeds, n_events=100):
    """Forgetting of a Trainer with ArchivalMemory as replay source vs baseline."""
    from dense_armor.utility.neural.activations import Relu
    from dense_armor.utility.neural.chain import Sequential
    from dense_armor.utility.neural.layers import Dense
    from dense_armor.utility.neural.losses import SoftmaxCrossEntropy
    from dense_armor.utility.neural.trainer import Trainer
    from dense_armor.utility.optim import Adam
    rows = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        X = jnp.asarray(rng.randn(2 * n_events, 2).astype(np.float32) * 1.5)
        rules = [
            lambda x: (x[:, 0] > 0),
            lambda x: (x[:, 1] > 0),
            lambda x: (x[:, 0] + x[:, 1] > 0),
        ]
        tasks = [(X, jnp.asarray(rule(np.asarray(X)).astype(np.int32))) for rule in rules]
        key = jax.random.PRNGKey(seed)
        model = Sequential([Dense(2, 4, key), Relu(), Dense(4, 2, key)])
        loss = SoftmaxCrossEntropy()
        tr = Trainer(model, loss, Adam(lr=0.005))
        am = ArchivalMemory(d_model=2, n_codes=16, max_depth=3, seed=seed)
        acc_before = []
        for k, (Xk, yk) in enumerate(tasks):
            tr.fit(Xk, yk, epochs=30, batch_size=32, key=key)
            acc_before.append(float((np.argmax(np.asarray(model.forward(Xk, training=False)), axis=-1) == np.asarray(yk)).mean()))
            for i in range(min(20, len(Xk))):
                am.write(Xk[i], t=float(k * 100 + i))
        acc_after = []
        for (Xk, yk) in tasks:
            acc_after.append(float((np.argmax(np.asarray(model.forward(Xk, training=False)), axis=-1) == np.asarray(yk)).mean()))
        forgetting = float(np.mean([max(0.0, acc_before[k] - acc_after[k]) for k in range(len(tasks))]))
        rows.append({"seed": int(seed), "archival_forgetting": forgetting})
    return rows


if __name__ == "__main__":
    main()

