"""Internal benchmark: forgetting of five learners on three synthetic tasks.

Run with ``--smoke`` (2 seeds) or ``--full`` (20 seeds). Writes
``benchmarks/neural/continual_learning.json``.

The synthetic tasks are 2-class Gaussian clusters, rotated by 30 degrees
from one task to the next, 100 samples per task, seed printed. The model
is a 1-hidden-layer MLP (8 units). Baseline and DGR use ``Trainer``; EWC,
GEM, A-GEM use a manual loop that applies the constraint per step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import wilcoxon

from dense_armor.utility.neural.chain import Sequential
from dense_armor.utility.neural.continual import AGEM, EWC, GEM
from dense_armor.utility.neural.generative_replay import DeepGenerativeReplay
from dense_armor.utility.neural.layers import Dense
from dense_armor.utility.neural.activations import Relu
from dense_armor.utility.neural.losses import SoftmaxCrossEntropy
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import Adam


def make_tasks(n_tasks: int, n_per: int, seed: int):
    """Three tasks in conflict, so that forgetting is measurable.

    Task 0: class = (x[0] > 0). Task 1: class = (x[1] > 0).
    Task 2: class = (x[0] + x[1] > 0). The three boundaries are different
    and in conflict: a model that fits task 0 by looking at x[0] gets
    task 1 wrong (which depends only on x[1]), and so on. This is what
    makes the baseline forget; the 30-degree rotation used before left
    all tasks linearly separable with the same boundary, forgetting was
    zero for everyone and the benchmark proved nothing.
    """
    rng = np.random.RandomState(seed)
    tasks = []
    X = rng.randn(2 * n_per, 2).astype(np.float32) * 1.5
    for k in range(min(n_tasks, 3)):
        if k == 0:
            y = (X[:, 0] > 0).astype(np.int32)
        elif k == 1:
            y = (X[:, 1] > 0).astype(np.int32)
        else:
            y = (X[:, 0] + X[:, 1] > 0).astype(np.int32)
        tasks.append((jnp.asarray(X), jnp.asarray(y)))
    return tasks


def make_model(key):
    return Sequential([Dense(2, 8, key), Relu(), Dense(8, 2, key)])


def accuracy(model, X, y):
    pred = model.forward(X, training=False)
    return float((np.argmax(np.asarray(pred), axis=-1) == np.asarray(y)).mean())


def flat_grad(model):
    return jnp.concatenate([p.grad.reshape(-1) for p in model.params()])


def set_flat_grad(model, g):
    offset = 0
    for p in model.params():
        n = p.value.size
        p.grad = g[offset:offset + n].reshape(p.value.shape)
        offset += n






def a12(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    gt = np.sum(a[:, None] > b[None, :])
    eq = np.sum(a[:, None] == b[None, :])
    return float((gt + 0.5 * eq) / (a.size * b.size))


def train_one(name, seed):
    """Train ``name`` on the 3 tasks, return (model, acc_before, acc_after).

    ``acc_before[k]`` is the accuracy on task k measured right after that
    task is learned; ``acc_after[k]`` is the accuracy on task k at the end
    of the sequence. ``forgetting_k = acc_before[k] - acc_after[k]``.
    """
    tasks = make_tasks(3, 100, seed)
    key = jax.random.PRNGKey(seed)
    model = make_model(key)
    if name == "dgr":
        dgr = DeepGenerativeReplay(
            n_x=2, n_y=2, dim_z=4, hidden=16, n_fake=32, seed=seed, lr=0.05
        )
        acc_before = []
        for k, (X, y) in enumerate(tasks):
            dgr.train_task(X, y, n_epochs=10, batch_size=32)
            acc_before.append(accuracy(dgr.solver, X, y))
        model = dgr.solver
    else:
        loss = SoftmaxCrossEntropy()
        opt_per = [Adam(lr=0.01) for _ in model.params()]
        ewc = EWC(model, loss, lambda_=0.4) if name == "ewc" else None
        mem = (
            GEM(memory_per_task=32, seed=seed) if name == "gem"
            else AGEM(memory_per_task=32, seed=seed) if name == "agem"
            else None
        )
        past_grads: list = []
        acc_before = []
        X_prev = y_prev = None
        for k, (X, y) in enumerate(tasks):
            if name == "ewc" and k > 0 and X_prev is not None:
                ewc.consolidate(X_prev, y_prev)
            if mem is not None:
                for i in range(len(X)):
                    mem.observe(X[i], y[i], task_id=k)
            n = len(X)
            for _epoch in range(20):
                key, sub = jax.random.split(key)
                idx = np.asarray(jax.random.permutation(sub, n))
                for b in range(0, n, 32):
                    batch = idx[b:b + 32]
                    for pp in model.params():
                        pp.zero_grad()
                    pred = model.forward(X[batch], training=True)
                    loss.forward(pred, y[batch])
                    model.backward(loss.backward())
                    if name == "ewc" and k > 0:
                        theta = ewc.flat_params()
                        diff = theta - ewc.theta_star_
                        pen = ewc.lambda_ * ewc.F_ * diff
                        cur = flat_grad(model)
                        set_flat_grad(model, cur + pen)
                    if mem is not None and past_grads:
                        g = flat_grad(model)
                        if name == "gem":
                            g = mem.project(g, past_grads)
                        else:
                            g_ref = jnp.mean(jnp.stack(past_grads, axis=0), axis=0)
                            g = mem.project(g, g_ref)
                        set_flat_grad(model, g)
                    for pp, o in zip(model.params(), opt_per):
                        pp.update(o)
            acc_before.append(accuracy(model, X, y))
            if mem is not None:
                mems = mem.memory()
                grads = []
                for Xm, ym in mems:
                    for pp in model.params():
                        pp.zero_grad()
                    pred = model.forward(Xm, training=True)
                    loss.forward(pred, jnp.asarray(ym, dtype=jnp.int32))
                    model.backward(loss.backward())
                    grads.append(flat_grad(model))
                past_grads = grads
            X_prev, y_prev = X, y
    acc_after = [accuracy(model, Xk, yk) for Xk, yk in tasks]
    return model, acc_before, acc_after


def run(seeds):
    learners = ["baseline", "ewc", "gem", "agem", "dgr"]
    acc_avg_per_seed = {k: [] for k in learners}
    forgetting_per_seed = {k: [] for k in learners}
    rows = []
    for seed in seeds:
        for name in learners:
            _, acc_before, acc_after = train_one(name, seed)
            acc_avg = float(np.mean(acc_after))
            forgetting = float(np.mean(
                [max(0.0, acc_before[k] - acc_after[k]) for k in range(len(acc_after))]
            ))
            acc_avg_per_seed[name].append(acc_avg)
            forgetting_per_seed[name].append(forgetting)
            rows.append({
                "seed": int(seed), "learner": name,
                "acc_avg": acc_avg, "forgetting": forgetting,
                "acc_before": acc_before, "acc_after": acc_after,
            })
    return acc_avg_per_seed, forgetting_per_seed, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    seeds = list(range(20)) if args.full else [0, 1]
    acc, forg, rows = run(seeds)
    print(f"seeds: {len(seeds)}")
    print(f"{'learner':10s}  {'acc_avg (median [IQR])':>30s}  {'forgetting (median)':>20s}")
    for name in acc:
        med = float(np.median(acc[name]))
        q1, q3 = np.percentile(acc[name], [25, 75])
        fmed = float(np.median(forg[name]))
        print(f"{name:10s}  {med:.3f} [{q1:.3f}, {q3:.3f}]  {fmed:.3f}")
    if len(seeds) >= 4:
        for name in acc:
            if name == "baseline":
                continue
            try:
                stat, p = wilcoxon(acc[name], acc["baseline"])
                print(f"  wilcoxon {name} vs baseline: p={p:.4f}  A12={a12(acc[name], acc['baseline']):.3f}")
            except Exception as e:
                print(f"  wilcoxon {name}: {e}")
    out = Path("benchmarks/neural/continual_learning.json")
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
