"""MoE vs dense, and test-time compute, on a micro 2D domain, with seeds.

Two experiments on a synthetic 2D domain, repeated over a list of seeds,
with mean and standard deviation:

1. MoE vs dense. A `SparseMoE(n_experts=4, top_k=2, hidden)` and a dense
   `Sequential([Dense(2, 32), Relu(), Dense(32, 2)])` with matched
   parameter count, trained on a 2-task sequence (half-plane x > 0,
   half-plane y > 0). For each seed: accuracy before and after task 2,
   forgetting, expert usage, dropped fraction, wall time, parameter count.

2. Test-time compute vs single pass. A
   `Sequential([Dense(2, 16), Relu(), Dense(16, 2)])` with and without a
   `TestTimeCompute(n_candidates, noise)` wrapper, at three noise levels
   (0.1, 0.3, 0.5). For each seed and each (n_candidates, noise) pair:
   single-pass accuracy, test-time accuracy, delta.

Writes `benchmarks/neural/moe_testtime.json` with per-seed values and
the aggregate (mean, std over seeds) of every metric. `--smoke` uses 2
seeds, 2 epochs, 50 samples per task; without `--smoke`, 5 seeds, 5
epochs, 100 samples per task.

Run: python benchmarks/neural/moe_testtime.py [--smoke] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import numpy as np

from dense_armor.utility.neural.activations import Relu
from dense_armor.utility.neural.chain import Sequential
from dense_armor.utility.neural.layers import Dense
from dense_armor.utility.neural.losses import SoftmaxCrossEntropy
from dense_armor.utility.neural.moe import SparseMoE
from dense_armor.utility.neural.test_time import TestTimeCompute
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import Adam


def make_tasks(n, seed, noise=0.0):
    rng = np.random.RandomState(seed)
    X = rng.randn(2 * n, 2).astype(np.float32)
    if noise > 0:
        X = X + noise * rng.randn(*X.shape).astype(np.float32)
    y1 = (X[:n, 0] > 0).astype(np.int32)
    y2 = (X[n:, 1] > 0).astype(np.int32)
    return [
        (jnp.asarray(X[:n]), jnp.asarray(y1)),
        (jnp.asarray(X[n:]), jnp.asarray(y2)),
    ]


def accuracy(model, X, y):
    try:
        logits = model.forward(X, training=False)
    except TypeError:
        logits = model.forward(X)
    pred = np.argmax(np.asarray(logits), axis=-1)
    return float((pred == np.asarray(y)).mean())


def train_sequence(model, tasks, key, n_epochs, batch=32):
    loss = SoftmaxCrossEntropy()
    trainer = Trainer(model, loss, Adam(lr=0.01))
    acc_before = []
    for X, y in tasks:
        trainer.fit(X, y, epochs=n_epochs, batch_size=batch, key=key)
        acc_before.append(accuracy(model, X, y))
    acc_after = [accuracy(model, X, y) for X, y in tasks]
    forgetting = float(
        np.mean([max(0.0, acc_before[k] - acc_after[k]) for k in range(len(tasks))])
    )
    return acc_before, acc_after, forgetting


def count_params(model):
    return int(sum(int(p.value.size) for p in model.params()))


def _agg(values):
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "n": int(arr.size),
    }


def run_moe_vs_dense(seeds, smoke):
    n = 100 if not smoke else 50
    epochs = 5 if not smoke else 2
    per_seed = []
    for seed in seeds:
        tasks = make_tasks(n, seed)
        key = jax.random.PRNGKey(seed)

        dense = Sequential([Dense(2, 32, key), Relu(), Dense(32, 2, key)])
        n_dense = count_params(dense)
        t0 = time.time()
        acc_b, acc_a, for_g = train_sequence(dense, tasks, key, epochs)
        t_dense = time.time() - t0

        n_experts = 4
        gating_params = 2 * n_experts + n_experts
        hidden = max(
            2, int(round((n_dense - gating_params - n_experts * 2) / (n_experts * 5)))
        )
        moe = SparseMoE(
            n_x=2,
            n_y=2,
            n_experts=n_experts,
            top_k=2,
            hidden=hidden,
            seed=seed,
            capacity_factor=1.25,
        )
        n_moe = count_params(moe)
        t0 = time.time()
        acc_b_m, acc_a_m, for_m = train_sequence(moe, tasks, key, epochs)
        t_moe = time.time() - t0

        moe.forward(tasks[0][0])
        usage = np.asarray(moe.expert_usage()).tolist()
        drop = float(moe.dropped_fraction())

        per_seed.append(
            {
                "seed": seed,
                "dense": {
                    "n_params": n_dense,
                    "acc_before": acc_b,
                    "acc_after": acc_a,
                    "forgetting": for_g,
                    "wall_time_s": t_dense,
                },
                "moe": {
                    "n_params": n_moe,
                    "hidden": hidden,
                    "acc_before": acc_b_m,
                    "acc_after": acc_a_m,
                    "forgetting": for_m,
                    "wall_time_s": t_moe,
                    "expert_usage": usage,
                    "dropped_fraction": drop,
                },
            }
        )

    def _mean(key, sub):
        return _agg([d[sub][key] for d in per_seed])

    return {
        "n_samples_per_task": n,
        "n_epochs": epochs,
        "seeds": list(seeds),
        "per_seed": per_seed,
        "aggregate": {
            "dense_acc_after_task1": _agg(
                [d["dense"]["acc_after"][0] for d in per_seed]
            ),
            "dense_acc_after_task2": _agg(
                [d["dense"]["acc_after"][1] for d in per_seed]
            ),
            "dense_forgetting": _agg([d["dense"]["forgetting"] for d in per_seed]),
            "dense_wall_time_s": _agg([d["dense"]["wall_time_s"] for d in per_seed]),
            "moe_acc_after_task1": _agg([d["moe"]["acc_after"][0] for d in per_seed]),
            "moe_acc_after_task2": _agg([d["moe"]["acc_after"][1] for d in per_seed]),
            "moe_forgetting": _agg([d["moe"]["forgetting"] for d in per_seed]),
            "moe_wall_time_s": _agg([d["moe"]["wall_time_s"] for d in per_seed]),
            "moe_dropped_fraction": _agg(
                [d["moe"]["dropped_fraction"] for d in per_seed]
            ),
            "moe_expert_usage_mean": [
                float(np.mean([d["moe"]["expert_usage"][i] for d in per_seed]))
                for i in range(4)
            ],
        },
    }


def run_testtime(seeds, smoke):
    n = 100 if not smoke else 50
    epochs = 5 if not smoke else 2
    verifiers = ("confidence", "margin", "entropy", "ensemble")
    configs = [(8, 0.1)]
    noises = (0.1, 0.3, 0.5)
    grid = []
    for noise in noises:
        for n_candidates, ttc_noise in configs:
            for verifier in verifiers:
                rows = []
                for seed in seeds:
                    tasks = make_tasks(n, seed, noise=noise)
                    key = jax.random.PRNGKey(seed)
                    base = Sequential([Dense(2, 16, key), Relu(), Dense(16, 2, key)])
                    train_sequence(base, tasks, key, epochs)
                    acc_single = float(
                        np.mean([accuracy(base, X, y) for X, y in tasks])
                    )
                    ttc = TestTimeCompute(
                        base,
                        n_candidates=n_candidates,
                        noise=ttc_noise,
                        seed=seed,
                        verifier=verifier,
                    )
                    acc_ttc = float(np.mean([accuracy(ttc, X, y) for X, y in tasks]))
                    rows.append(
                        {
                            "seed": seed,
                            "single_pass": acc_single,
                            "test_time": acc_ttc,
                            "delta": acc_ttc - acc_single,
                        }
                    )
                grid.append(
                    {
                        "domain_noise": noise,
                        "n_candidates": n_candidates,
                        "ttc_noise": ttc_noise,
                        "verifier": verifier,
                        "single_pass_acc": _agg([r["single_pass"] for r in rows]),
                        "test_time_acc": _agg([r["test_time"] for r in rows]),
                        "delta": _agg([r["delta"] for r in rows]),
                    }
                )
    return {
        "n_samples_per_task": n,
        "n_epochs": epochs,
        "seeds": list(seeds),
        "configs": grid,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_suffix(".json"),
    )
    args = parser.parse_args()

    smoke = bool(args.smoke)
    seeds = [0, 1] if smoke else [0, 1, 2, 3, 4]
    out = {
        "smoke": smoke,
        "moe_vs_dense": run_moe_vs_dense(seeds, smoke),
        "test_time": run_testtime(seeds, smoke),
    }
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"written: {args.out}")
    m = out["moe_vs_dense"]["aggregate"]
    print("MoE vs dense (mean ± std over seeds):")
    print(
        f"  dense acc task1 = {m['dense_acc_after_task1']['mean']:.3f} ± {m['dense_acc_after_task1']['std']:.3f}"
    )
    print(
        f"  dense acc task2 = {m['dense_acc_after_task2']['mean']:.3f} ± {m['dense_acc_after_task2']['std']:.3f}"
    )
    print(
        f"  dense forgetting = {m['dense_forgetting']['mean']:.3f} ± {m['dense_forgetting']['std']:.3f}"
    )
    print(
        f"  MoE acc task1 = {m['moe_acc_after_task1']['mean']:.3f} ± {m['moe_acc_after_task1']['std']:.3f}"
    )
    print(
        f"  MoE acc task2 = {m['moe_acc_after_task2']['mean']:.3f} ± {m['moe_acc_after_task2']['std']:.3f}"
    )
    print(
        f"  MoE forgetting = {m['moe_forgetting']['mean']:.3f} ± {m['moe_forgetting']['std']:.3f}"
    )
    print(
        f"  MoE dropped = {m['moe_dropped_fraction']['mean']:.3f} ± {m['moe_dropped_fraction']['std']:.3f}"
    )
    print(f"  MoE usage mean = {[round(v, 3) for v in m['moe_expert_usage_mean']]}")
    print("Test-time compute (delta over single pass):")
    for row in out["test_time"]["configs"]:
        d = row["delta"]
        print(
            f"  noise={row['domain_noise']} verifier={row['verifier']}: "
            f"delta = {d['mean']:.3f} ± {d['std']:.3f}"
        )


if __name__ == "__main__":
    main()
