"""Figures for the trees docs page.

Generates three PNGs in the same folder:
- hoeffding_adaptive.png   accuracy over time, plain vs adaptive tree, before and after a change
- mondrian_uncertainty.png Mondrian forest prediction with a +/-1 sigma band, trained on [0, 1]
- sgt_surface.png          stochastic gradient tree vs the true gravity torque of a two-link arm

Run from the repo root:
    python docs/assets/tree/make_figure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.utility.drift.adwin import ADWIN
from dense_armor.utility.tree import (
    HoeffdingAdaptiveTreeClassifier,
    HoeffdingTreeClassifier,
    MondrianForestRegressor,
    SGTRegressor,
)

HERE = Path(__file__).resolve().parent
DPI = 130

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": False,
})


def figure_adaptive():
    """Plain vs adaptive Hoeffding tree under a sudden change."""
    rng = np.random.default_rng(1)
    ht = HoeffdingTreeClassifier(grace_period=100)
    hat = HoeffdingAdaptiveTreeClassifier(
        grace_period=100, drift_detector=ADWIN()
    )

    n, change, window = 6000, 3000, 200
    correct_ht = np.zeros(n, dtype=bool)
    correct_hat = np.zeros(n, dtype=bool)
    acc_ht = np.full(n, np.nan)
    acc_hat = np.full(n, np.nan)
    drift_steps = []

    for i in range(n):
        y = int(rng.random() < 0.5)
        c = 3.0 * y if i < change else 3.0 * (1 - y)
        x = {"tau": rng.normal(2.0 + c, 0.5), "qd": rng.normal(0.0, 1.0)}
        correct_ht[i] = ht.predict_one(x) == y
        correct_hat[i] = hat.predict_one(x) == y
        if i >= window - 1:
            lo = i - window + 1
            acc_ht[i] = correct_ht[lo : i + 1].mean()
            acc_hat[i] = correct_hat[lo : i + 1].mean()
        n_before = hat.n_drifts_
        ht.learn_one(x, y)
        hat.learn_one(x, y)
        if hat.n_drifts_ > n_before:
            drift_steps.append(i)

    post = slice(change, n)
    mean_ht = float(correct_ht[post].mean())
    mean_hat = float(correct_hat[post].mean())

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.axvspan(0, change, color="C2", alpha=0.07)
    ax.axvspan(change, n, color="C3", alpha=0.07)
    ax.plot(np.arange(n), acc_ht, color="C3", lw=1.6,
            label=f"HoeffdingTreeClassifier  (post-change mean {mean_ht:.3f})")
    ax.plot(np.arange(n), acc_hat, color="C0", lw=1.6,
            label=f"HoeffdingAdaptiveTreeClassifier  (post-change mean {mean_hat:.3f})")
    ax.axhline(mean_ht, color="C3", lw=0.8, ls=":", alpha=0.6)
    ax.axhline(mean_hat, color="C0", lw=0.8, ls=":", alpha=0.6)
    for s in drift_steps:
        ax.axvline(s, color="C0", lw=0.7, alpha=0.45)
    ax.axvline(change, color="k", lw=1.0, ls=":")
    ax.text(change + 100, 0.03, "signal flips", fontsize=8)
    ax.text(change / 2, 0.03, "before", fontsize=8, ha="center")
    ax.set_xlabel("sample")
    ax.set_ylabel(f"accuracy (rolling window of {window})")
    ax.set_xlim(0, n)
    ax.set_ylim(-0.02, 1.03)
    ax.legend(loc="center left")
    fig.tight_layout()
    out = HERE / "hoeffding_adaptive.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def figure_mondrian():
    """Mondrian forest prediction with uncertainty, inside and outside the data."""
    rng = np.random.default_rng(4)
    mf = MondrianForestRegressor(n_trees=10, seed=0)
    for _ in range(1000):
        q = rng.uniform(0.0, 1.0)
        mf.learn_one({"q": q}, np.sin(6.0 * q) + rng.normal(0.0, 0.05))

    qs = np.linspace(-0.5, 3.5, 200)
    means = np.zeros_like(qs)
    stds = np.zeros_like(qs)
    for i, q in enumerate(qs):
        e = mf.predict_one({"q": float(q)}, return_estimate=True)
        means[i] = e.mean
        stds[i] = e.std

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.axvspan(0.0, 1.0, color="C2", alpha=0.10, label="training range  [0, 1]")
    ax.fill_between(qs, means - stds, means + stds,
                    color="C0", alpha=0.22, label="prediction +/- 1 sigma")
    ax.plot(qs, means, color="C0", lw=1.8, label="prediction")
    ax.plot(qs, np.sin(6.0 * qs), "k--", lw=1.0, alpha=0.55,
            label="true  sin(6q)")
    for q0 in (0.5, 1.5, 3.0):
        e = mf.predict_one({"q": float(q0)}, return_estimate=True)
        ax.errorbar([q0], [e.mean], yerr=[e.std], fmt="o",
                    color="C3", capsize=3, markersize=4, zorder=5)
    ax.set_xlabel("joint angle  q  (rad)")
    ax.set_ylabel("target")
    ax.set_xlim(-0.5, 3.5)
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    out = HERE / "mondrian_uncertainty.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def figure_sgt():
    """SGT prediction surface vs the true gravity torque of a two-link arm."""
    rng = np.random.default_rng(3)
    sgt = SGTRegressor()
    for _ in range(20000):
        q1, q2 = rng.uniform(-np.pi, np.pi, 2)
        tau = 9.81 * (2.0 * np.cos(q1) + np.cos(q1 + q2))
        sgt.learn_one({"q1": float(q1), "q2": float(q2)}, tau)

    n = 60
    qs = np.linspace(-np.pi, np.pi, n)
    Q1, Q2 = np.meshgrid(qs, qs)
    pred = np.zeros_like(Q1)
    for i in range(n):
        for j in range(n):
            pred[i, j] = sgt.predict_one(
                {"q1": float(Q1[i, j]), "q2": float(Q2[i, j])}
            )
    true = 9.81 * (2.0 * np.cos(Q1) + np.cos(Q1 + Q2))
    vmin, vmax = float(true.min()), float(true.max())
    err = pred - true
    elim = float(np.abs(err).max())

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9))
    ext = (qs[0], qs[-1], qs[0], qs[-1])
    im0 = axes[0].imshow(true, extent=ext, origin="lower", aspect="auto",
                         cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("true  $m g J_z(q)$", fontsize=10)
    im1 = axes[1].imshow(pred, extent=ext, origin="lower", aspect="auto",
                         cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("SGT prediction", fontsize=10)
    im2 = axes[2].imshow(err, extent=ext, origin="lower", aspect="auto",
                         cmap="RdBu_r", vmin=-elim, vmax=elim)
    axes[2].set_title("prediction - truth", fontsize=10)
    for ax in axes:
        ax.set_xlabel("$q_1$ (rad)")
        ax.grid(False)
    axes[0].set_ylabel("$q_2$ (rad)")
    fig.colorbar(im0, ax=axes[:2], fraction=0.045, pad=0.02,
                 label="torque (N m)")
    fig.colorbar(im2, ax=axes[2], fraction=0.06, pad=0.02,
                 label="error (N m)")
    fig.tight_layout()
    out = HERE / "sgt_surface.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    figure_adaptive()
    figure_mondrian()
    figure_sgt()


if __name__ == "__main__":
    main()
