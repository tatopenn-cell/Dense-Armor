"""Benchmark: linear vs mondrian vs random router of ResidualTree."""

import sys, time
from pathlib import Path
import numpy as np
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dense_armor.utility.neural.consolidation import ResidualTree

def make_data(n, seed):
    rng = np.random.RandomState(seed)
    return rng.randn(n, 2).astype(np.float32) * 3.0

def run_seed(seed, n_events=200):
    data = make_data(n_events, seed)
    xs = [jnp.asarray(x) for x in data]
    rows = {}
    for name in ("linear", "mondrian", "random"):
        tree = ResidualTree(d_model=2, n_codes=8, max_depth=2, branching=4,
                            kappa=0.0, seed=seed, router=name)
        t0 = time.perf_counter()
        for x in xs:
            tree.insert(x)
        t_insert = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        mses = []
        depths = []
        for x in xs:
            x_hat, d = tree.read(x)
            mses.append(float(jnp.mean((x - x_hat) ** 2)))
            depths.append(d)
        t_read = (time.perf_counter() - t0) * 1000

        rows[name] = {
            "insert_ms": t_insert,
            "read_ms": t_read,
            "n_nodes": tree.n_nodes(),
            "mse": float(np.median(mses)),
            "depth_mean": float(np.mean(depths)),
        }
    return rows

seeds = list(range(20))
all_rows = {n: [] for n in ("linear", "mondrian", "random")}
for s in seeds:
    r = run_seed(s)
    for n in all_rows:
        all_rows[n].append(r[n])

print(f"seeds: {len(seeds)}")
print(f"\n{'router':10s}  {'insert ms':>22s}  {'read ms':>22s}  {'n_nodes':>8s}  {'mse':>10s}  {'depth':>6s}")
for name in ("linear", "mondrian", "random"):
    ins = [r["insert_ms"] for r in all_rows[name]]
    rd = [r["read_ms"] for r in all_rows[name]]
    nn = [r["n_nodes"] for r in all_rows[name]]
    ms = [r["mse"] for r in all_rows[name]]
    de = [r["depth_mean"] for r in all_rows[name]]
    print(f"{name:10s}  {np.median(ins):8.1f} [{np.percentile(ins,25):.1f},{np.percentile(ins,75):.1f}]  "
          f"{np.median(rd):8.1f} [{np.percentile(rd,25):.1f},{np.percentile(rd,75):.1f}]  "
          f"{np.median(nn):8.1f}  {np.median(ms):10.4f}  {np.median(de):6.2f}")

# Wilcoxon su insert e read linear vs mondrian
from scipy.stats import wilcoxon
for metric in ("insert_ms", "read_ms", "mse", "n_nodes"):
    a = np.array([r[metric] for r in all_rows["mondrian"]])
    b = np.array([r[metric] for r in all_rows["linear"]])
    try:
        _, p = wilcoxon(a, b)
        gt = np.sum(a[:, None] > b[None, :])
        eq = np.sum(a[:, None] == b[None, :])
        a12 = (gt + 0.5 * eq) / (a.size * b.size)
        direction = "mondrian > linear" if np.median(a) > np.median(b) else "mondrian < linear"
        print(f"  {metric:12s}: p={p:.4f} A12={a12:.3f} ({direction})")
    except Exception as e:
        print(f"  {metric:12s}: {e}")
