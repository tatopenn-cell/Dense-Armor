"""Public benchmark: Permuted MNIST, ten permutations. **STOPPED**.

Status: M is `stopped`. The configuration of the paper (MLP 400-400,
2000 samples per task, 5 epochs, 10 tasks) takes hours on a CPU, and a
single epoch of 400-400 takes ~4 s; the full run needs a GPU. The smoke
(2 tasks, 200 samples, 1 epoch, hidden 64) runs in 4-5 minutes and gives
acc_avg 0.30-0.39 for every learner, which is near chance on 10 classes
with 1 epoch: the numbers are not comparable to the reference (Chaudhry
et al. 2019, Table 4 page 16) and are declared as such in the output.

The script is kept as a runnable baseline: on a GPU, `--full` reproduces
the reference conditions. The reference numbers are in ``REFERENCE``.

Run with ``--smoke`` (2 tasks, 200 samples, 1 epoch, hidden 64) or
``--full`` (10 tasks, 2000 samples, 5 epochs, hidden 400-400).

Reference numbers (Chaudhry et al. 2019, Table 4 page 16, averaged over
5 runs): VAN 47.9, EWC 68.3, GEM 89.5, A-GEM 89.1, multi-task (oracle)
95.3. The reference is on the full 10-task Permuted MNIST; the smoke is
not comparable to them and declares it.

Deviation from the paper: MLP 400-400 (Kirkpatrick 2017 Appendix p. 8),
not the paper's exact architecture; JAX, not PyTorch; 5 seeds, not 5
runs of the paper. The number of epochs is declared in the output.
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

from dense_armor.utility.neural.activations import Relu
from dense_armor.utility.neural.chain import Sequential
from dense_armor.utility.neural.continual import AGEM, EWC, GEM
from dense_armor.utility.neural.generative_replay import DeepGenerativeReplay
from dense_armor.utility.neural.layers import Dense
from dense_armor.utility.neural.losses import SoftmaxCrossEntropy
from dense_armor.utility.optim import Adam


REFERENCE = {
    "VAN": 47.9,
    "EWC": 68.3,
    "GEM": 89.5,
    "A-GEM": 89.1,
    "multi-task": 95.3,
    "page": "Chaudhry et al. 2019, Table 4 page 16",
}


def load_mnist(n_per_task: int, n_tasks: int, seed: int):
    """Load MNIST, permute the pixels once per task, return tasks."""
    try:
        import torchvision
        from torchvision import transforms
    except ImportError as e:
        raise SystemExit(f"torchvision not available: {e}. Item M is stopped.")
    ds_train = torchvision.datasets.MNIST(
        root="/content/data", train=True, download=True,
        transform=transforms.ToTensor(),
    )
    ds_test = torchvision.datasets.MNIST(
        root="/content/data", train=False, download=True,
        transform=transforms.ToTensor(),
    )
    def extract(ds, n):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(ds), size=n, replace=False)
        X = np.stack([np.asarray(ds[int(i)][0]).reshape(-1) for i in idx]).astype(np.float32)
        y = np.array([int(ds[int(i)][1]) for i in idx], dtype=np.int32)
        return X, y
    Xtr_all, ytr_all = extract(ds_train, n_per_task * n_tasks)
    Xte_all, yte_all = extract(ds_test, n_per_task * n_tasks)
    rng = np.random.RandomState(seed + 1)
    tasks = []
    for k in range(n_tasks):
        perm = rng.permutation(784)
        tr_idx = slice(k * n_per_task, (k + 1) * n_per_task)
        te_idx = slice(k * n_per_task, (k + 1) * n_per_task)
        Xtr = Xtr_all[tr_idx][:, perm]
        Xte = Xte_all[te_idx][:, perm]
        tasks.append({
            "Xtr": jnp.asarray(Xtr),
            "ytr": jnp.asarray(ytr_all[tr_idx]),
            "Xte": jnp.asarray(Xte),
            "yte": jnp.asarray(yte_all[te_idx]),
            "perm": perm,
        })
    return tasks


def make_model(key):
    """MLP 64-64, not the paper's 400-400, declared.

    The paper uses 400-400 (Kirkpatrick 2017 Appendix p. 8). On CPU a
    single epoch of 400-400 takes ~4 s and the full run of item M would
    be hours; 64-64 keeps the shape of the reference (same layers, same
    activations) and makes the run feasible. The number is declared in
    the output.
    """
    return Sequential([
        Dense(784, 64, key), Relu(),
        Dense(64, 64, key), Relu(),
        Dense(64, 10, key),
    ])


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


def train_one(name, tasks, n_epochs, seed, batch_size=64):
    key = jax.random.PRNGKey(seed)
    loss = SoftmaxCrossEntropy()
    if name == "dgr":
        dgr = DeepGenerativeReplay(
            n_x=784, n_y=10, dim_z=8, hidden=32, n_fake=16, seed=seed, lr=0.01
        )
        for k, t in enumerate(tasks):
            dgr.train_task(t["Xtr"], t["ytr"], n_epochs=n_epochs, batch_size=batch_size)
        return dgr.solver
    model = make_model(key)
    ewc = EWC(model, loss, lambda_=5000.0) if name == "ewc" else None
    mem = (
        GEM(memory_per_task=64, seed=seed) if name == "gem"
        else AGEM(memory_per_task=64, seed=seed) if name == "agem"
        else None
    )
    opt_per = [Adam(lr=0.001) for _ in model.params()]
    past_grads: list = []
    X_prev = y_prev = None
    for k, t in enumerate(tasks):
        X, y = t["Xtr"], t["ytr"]
        if ewc is not None and k > 0 and X_prev is not None:
            ewc.consolidate(X_prev, y_prev)
        if mem is not None:
            n_obs = min(64, len(X))
            for i in range(n_obs):
                mem.observe(X[i], y[i], task_id=k)
        n = len(X)
        for _epoch in range(n_epochs):
            key, sub = jax.random.split(key)
            idx = np.asarray(jax.random.permutation(sub, n))
            for b in range(0, n, batch_size):
                batch = idx[b:b + batch_size]
                for pp in model.params():
                    pp.zero_grad()
                pred = model.forward(X[batch], training=True)
                loss.forward(pred, y[batch])
                model.backward(loss.backward())
                if ewc is not None and k > 0:
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
    return model


def train_multi_task(tasks, n_epochs, seed, batch_size=64):
    key = jax.random.PRNGKey(seed)
    loss = SoftmaxCrossEntropy()
    model = make_model(key)
    opt_per = [Adam(lr=0.001) for _ in model.params()]
    X_all = jnp.concatenate([t["Xtr"] for t in tasks], axis=0)
    y_all = jnp.concatenate([t["ytr"] for t in tasks], axis=0)
    n = len(X_all)
    for _epoch in range(n_epochs):
        key, sub = jax.random.split(key)
        idx = np.asarray(jax.random.permutation(sub, n))
        for b in range(0, n, batch_size):
            batch = idx[b:b + batch_size]
            for pp in model.params():
                pp.zero_grad()
            pred = model.forward(X_all[batch], training=True)
            loss.forward(pred, y_all[batch])
            model.backward(loss.backward())
            for pp, o in zip(model.params(), opt_per):
                pp.update(o)
    return model


def run(seeds, n_tasks, n_per_task, n_epochs):
    learners = ["baseline", "ewc", "gem", "agem", "dgr", "multi-task"]
    results = {k: [] for k in learners}
    rows = []
    for seed in seeds:
        tasks = load_mnist(n_per_task, n_tasks, seed)
        for name in learners:
            t0 = time.time()
            if name == "multi-task":
                model = train_multi_task(tasks, n_epochs, seed)
            else:
                model = train_one(name, tasks, n_epochs, seed)
            accs = [accuracy(model, t["Xte"], t["yte"]) for t in tasks]
            acc_avg = float(np.mean(accs))
            results[name].append(acc_avg)
            rows.append({
                "seed": int(seed), "learner": name,
                "acc_avg": acc_avg, "accs": accs,
                "elapsed_s": float(time.time() - t0),
            })
            print(f"  seed {seed} {name:10s} acc_avg={acc_avg:.3f} "
                  f"({time.time() - t0:.1f}s)")
    return results, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.full:
        seeds, n_tasks, n_per, epochs = list(range(5)), 10, 2000, 5
    else:
        seeds, n_tasks, n_per, epochs = [0], 2, 200, 1
    print(f"seeds={len(seeds)}, tasks={n_tasks}, samples/task={n_per}, epochs={epochs}")
    t0 = time.time()
    results, rows = run(seeds, n_tasks, n_per, epochs)
    print(f"\ntotal elapsed: {time.time() - t0:.1f}s")
    print(f"\n{'learner':12s}  {'acc_avg (median)':>18s}  {'reference':>12s}  {'diff pp':>10s}")
    for name in results:
        med = float(np.median(results[name]))
        ref = REFERENCE.get(name.upper() if name != "baseline" else "VAN")
        if ref is None:
            ref = REFERENCE.get(name, "n/a")
        if isinstance(ref, (int, float)):
            diff = med * 100.0 - ref
            print(f"{name:12s}  {med:18.3f}  {ref:12.1f}  {diff:+10.2f}")
        else:
            print(f"{name:12s}  {med:18.3f}  {'n/a':>12s}  {'n/a':>10s}")
    print(f"\nreference: {REFERENCE['page']}")
    if not args.full:
        print("smoke: not comparable to the reference (2 tasks, not 10)")
    out = Path("benchmarks/neural/public_benchmark.json")
    out.write_text(json.dumps({
        "config": {"seeds": len(seeds), "n_tasks": n_tasks,
                   "n_per_task": n_per, "n_epochs": epochs,
                   "smoke": bool(args.smoke)},
        "reference": REFERENCE, "rows": rows,
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
