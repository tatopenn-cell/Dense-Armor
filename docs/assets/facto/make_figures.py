"""Generate the six PNGs of the two guide pages (task 16).

Run from the repository root::

    python docs/assets/facto/make_figures.py

Every quantity in every figure is computed by the library primitives
(``dense_armor.utility.facto``, ``reco``, ``misc``); matplotlib only
draws. The measured values are stored in ``figures.json`` by
``benchmarks/facto_quality.py``. Figure standard (task 16):
``figsize=(7.2, 3.6)``, ``dpi=150``, surface ``#fcfcfb``, data
``#2a78d6``, notice ``#eb6834``, true or best dashed ``#0b0b0b``.
"""

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dense_armor.utility.facto import (
    FMRegressor,
)
from dense_armor.utility.misc import SDFT, Skyline
from dense_armor.utility.optim import SGD
from dense_armor.utility.reco import BiasedMF

HERE = Path(__file__).resolve().parent
FACTO = HERE / "facto"
RECO = HERE / "reco"
FACTO.mkdir(exist_ok=True)
RECO.mkdir(exist_ok=True)

SURFACE = "#fcfcfb"
DATA = "#2a78d6"
NOTICE = "#eb6834"
TRUE = "#0b0b0b"


def _style(ax, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  {path.name}: {path.stat().st_size // 1024} KB")


def fig_fm_pairwise():
    """Pairwise term of the FM: <v_i, v_j> x_i x_j for three pairs."""
    fm = FMRegressor(optimizer=SGD(lr=0.05), k=3, seed=0)
    fm._resolve_keys({"a": 1.0, "b": 1.0, "c": 1.0})
    fm._ensure_state(fm._keys)
    pairs = [("a", "b"), ("a", "c"), ("b", "c")]
    _, w, _ = fm._unpack()
    contrib = []
    for u, v in pairs:
        x = {u: 1.0, v: 1.0}
        total = fm._y_hat(x)
        linear = w[fm._key_index[u]] + w[fm._key_index[v]]
        contrib.append(total - linear)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    xpos = np.arange(len(pairs))
    ax.bar(xpos, contrib, color=DATA, width=0.55)
    ax.axhline(0.0, color=TRUE, linestyle="--", linewidth=1)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{u}-{v}" for u, v in pairs])
    _style(ax, "feature pair", "pairwise term (unit)")
    ax.set_title("FM pairwise term <v_i, v_j> x_i x_j (seed 0)", fontsize=11)
    _save(fig, FACTO / "fm_pairwise.png")


def fig_ok_vs_double():
    """Time per sample of the O(kn) form vs the O(kn^2) double sum."""
    rng = np.random.default_rng(0)
    ns = [10, 20, 40, 80, 160]
    fast_ms, slow_ms = [], []
    for n in ns:
        fm = FMRegressor(optimizer=SGD(lr=0.05), k=4, seed=0)
        keys = [f"x{i}" for i in range(n)]
        fm._resolve_keys({k: 0.0 for k in keys})
        fm._ensure_state(fm._keys)
        x = {k: float(rng.normal()) for k in keys}
        t0 = time.perf_counter()
        for _ in range(200):
            _ = fm._y_hat(x)
        fast_ms.append((time.perf_counter() - t0) / 200 * 1e3)
        t0 = time.perf_counter()
        for _ in range(200):
            _ = fm._pairwise_naive(x)
        slow_ms.append((time.perf_counter() - t0) / 200 * 1e3)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(ns, fast_ms, "o-", color=DATA, label="O(kn) form")
    ax.plot(ns, slow_ms, "s-", color=NOTICE, label="O(kn^2) double sum")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style(ax, "n (features)", "time per sample (ms)")
    ax.legend(frameon=False)
    ax.set_title("Lemma 3.1 (Rendle 2010, page 3): the two forms", fontsize=11)
    _save(fig, FACTO / "ok_vs_double.png")


def fig_comparison():
    """Accuracy on DriftStream sudden, 5 seeds, IQR error bars."""
    import json as _json

    figures = _json.loads((HERE / "figures.json").read_text())
    rows = figures["comparison"]["rows"]
    names = [r["model"] for r in rows]
    med = np.array([r["acc_median"] for r in rows])
    iqr = np.array([r["acc_iqr"] for r in rows])

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(names))
    ax.bar(x, med, yerr=iqr, capsize=5, color=DATA, width=0.55)
    ax.axhline(0.5, color=TRUE, linestyle="--", linewidth=1, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    _style(ax, "learner", "accuracy (median, IQR over 5 seeds)")
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False)
    ax.set_title("DriftStream sudden, 2000 samples, 5 seeds", fontsize=11)
    _save(fig, FACTO / "comparison.png")


def fig_recovery():
    """Median error vs number of ratings on the planted matrix."""
    figures = json.loads((HERE / "figures.json").read_text())
    trace = figures["recovery"]["trace"]
    epochs = [t[0] for t in trace]
    errs = [t[1] for t in trace]
    n_items = figures["recovery"]["n_items"]
    n_users = figures["recovery"]["n_users"]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(epochs, errs, "-", color=DATA, linewidth=1.8)
    ax.axhline(
        0.01, color=TRUE, linestyle="--", linewidth=1, label="noise floor (0.01)"
    )
    _style(ax, "epoch", f"median |error| over {n_users * n_items} ratings")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    ax.set_title("BiasedMF on a planted low-rank matrix (k=3, seed 0)", fontsize=11)
    _save(fig, RECO / "recovery.png")


def fig_rank():
    """Precision@k of BiasedMF.rank against the true top-k."""
    rng = np.random.default_rng(0)
    n_users, n_items, k = 20, 15, 3
    P = rng.normal(size=(n_users, k))
    Q = rng.normal(size=(n_items, k))
    mu = 3.5
    model = BiasedMF(k=k, lr=0.02, l2=0.001, seed=0)
    for _ in range(60):
        for u in range(n_users):
            for i in range(n_items):
                r = mu + float(P[u] @ Q[i]) + float(rng.normal(0, 0.01))
                _ = model.learn_one(u, i, r)
    ks = [1, 2, 3, 5, 8, 10, 15]
    precs = []
    for kk in ks:
        correct = 0
        total = 0
        for u in range(n_users):
            true_rank = np.argsort(-(P[u] @ Q.T))
            true_top = set(true_rank[:kk].tolist())
            pred_top = set(model.rank(u, list(range(n_items)), top_k=kk))
            correct += len(true_top & pred_top)
            total += kk
        precs.append(correct / total)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(ks, precs, "o-", color=DATA)
    ax.axhline(
        ks[-1] / n_items,
        color=TRUE,
        linestyle="--",
        linewidth=1,
        label="random (k/n_items)",
    )
    _style(ax, "k", "precision@k")
    ax.legend(frameon=False)
    ax.set_title("BiasedMF.rank against the true top-k, 20 users", fontsize=11)
    _save(fig, RECO / "rank.png")


def fig_sdft_skyline():
    """SDFT vs numpy.fft (left) and skyline at two times (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.6))

    rng = np.random.default_rng(0)
    N = 16
    values = rng.normal(size=64)
    s = SDFT(n=N)
    for v in values:
        _ = s.update(v)
    window = values[-N:]
    expected = np.fft.fft(window)
    ks = np.arange(N)
    ax1.plot(ks, np.abs(s.X), "o", color=DATA, label="SDFT")
    ax1.plot(ks, np.abs(expected), "x", color=NOTICE, label="numpy.fft")
    ax1.set_facecolor(SURFACE)
    ax1.set_xlabel("k")
    ax1.set_ylabel("|X[k]|")
    ax1.grid(True, alpha=0.3)
    for sp in ("top", "right"):
        ax1.spines[sp].set_visible(False)
    ax1.legend(frameon=False)
    ax1.set_title("(a) SDFT vs numpy.fft on one window", fontsize=10)

    rng = np.random.default_rng(1)
    s2 = Skyline(window=20)
    pts = rng.normal(size=(40, 2))
    for p in pts[:20]:
        _ = s2.update({"x": float(p[0]), "y": float(p[1])})
    sky_t1 = [list(map(float, p)) for p in s2.skyline_points()]
    for p in pts[20:]:
        _ = s2.update({"x": float(p[0]), "y": float(p[1])})
    sky_t2 = [list(map(float, p)) for p in s2.skyline_points()]

    ax2.scatter(pts[:, 0], pts[:, 1], s=8, color="#bdbdbd", label="all")
    if sky_t1:
        arr1 = np.array(sky_t1)
        ax2.scatter(
            arr1[:, 0],
            arr1[:, 1],
            s=40,
            color=DATA,
            marker="s",
            label="skyline t1",
            zorder=3,
        )
    if sky_t2:
        arr2 = np.array(sky_t2)
        ax2.scatter(
            arr2[:, 0],
            arr2[:, 1],
            s=40,
            color=NOTICE,
            marker="^",
            label="skyline t2",
            zorder=3,
        )
    ax2.set_facecolor(SURFACE)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.grid(True, alpha=0.3)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    ax2.legend(frameon=False, fontsize=8)
    ax2.set_title("(b) skyline at two times (window 20)", fontsize=10)

    _save(fig, RECO / "sdft_skyline.png")


def main():
    print("Generating docs/assets/facto/{facto,reco}/")
    fig_fm_pairwise()
    fig_ok_vs_double()
    fig_comparison()
    fig_recovery()
    fig_rank()
    fig_sdft_skyline()
    print("Done.")


if __name__ == "__main__":
    main()
