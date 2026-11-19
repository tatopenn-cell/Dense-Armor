"""Dark figure for the page of RecursiveLeastSquares, drawn with the library itself, on a fee formula that
changes: the manual says 2 % of the amount, the real fee has a fixed 0.50 on top, and at payment 300 the
commission becomes 3.5 %.

Generates in the same folder:
- rls_tracking.png   the fee that a learner with and without forgetting predicts for a payment of 100

Run from the repo root:
    python docs/assets/learn/make_dynamics_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.core.online.learn.online_dynamics import RecursiveLeastSquares

HERE = Path(__file__).resolve().parent
BG, PANEL, GRID = "#0b1118", "#111a24", "#26364a"
INK, MUTED = "#e8fbff", "#8fa3b8"
CYAN, ORANGE, PINK = "#3ec6e8", "#ffa21f", "#ff2d87"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "grid.color": GRID,
        "grid.alpha": 0.6,
        "font.size": 10,
    }
)


def main():
    rng = np.random.default_rng(0)
    models = {
        "lam = 1 (never forgets)": RecursiveLeastSquares(lam=1.0),
        "lam = 0.99": RecursiveLeastSquares(lam=0.99),
    }
    pred = {k: [] for k in models}
    for i in range(600):
        amount = rng.uniform(10, 500)
        fee = (0.02 if i < 300 else 0.035) * amount + 0.5 + rng.normal(0, 0.05)
        x = {"amount": amount, "one": 1.0}
        for k, m in models.items():
            m.learn_one(x, fee)
            pred[k].append(m.predict_one({"amount": 100.0, "one": 1.0}))
    t = np.arange(1, 601)
    true = np.where(t <= 300, 2.5, 4.0)
    fig = plt.figure(figsize=(16, 6.0))
    fig.text(
        0.04,
        0.945,
        "THE FORMULA IN THE MANUAL, AND THE REAL FEE",
        fontsize=20,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.04,
        0.885,
        "the fee of a payment of 100: the manual says 2 % (2.00); the real fee has 0.50 on top, and at payment 300 the commission becomes 3.5 %",
        fontsize=11.5,
        color=CYAN,
    )
    a1 = fig.add_axes([0.05, 0.12, 0.45, 0.62])
    a1.plot(t, true, color=INK, lw=1.4, ls="--", label="the real fee")
    a1.axhline(2.0, color=MUTED, lw=1.0, ls=":")
    a1.text(8, 2.08, "the manual: 2.00", color=MUTED, fontsize=10)
    a1.plot(
        t,
        pred["lam = 1 (never forgets)"],
        color=ORANGE,
        lw=2.6,
        label="lam = 1 (never forgets)",
    )
    a1.plot(t, pred["lam = 0.99"], color=CYAN, lw=2.6, label="lam = 0.99")
    a1.set_xlabel("payments")
    a1.set_ylabel("fee predicted for a payment of 100")
    a1.set_ylim(1.5, 4.5)
    a1.legend(loc="lower right", facecolor=PANEL, edgecolor=GRID, labelcolor=INK)
    a1.grid(True)
    a2 = fig.add_axes([0.58, 0.12, 0.4, 0.62])
    for k, col in (("lam = 1 (never forgets)", ORANGE), ("lam = 0.99", CYAN)):
        err = np.abs(np.array(pred[k]) - true)
        a2.plot(t, np.maximum(err, 1e-4), color=col, lw=2.4, label=k)
    a2.set_yscale("log")
    a2.set_xlabel("payments")
    a2.set_ylabel("absolute error of that prediction")
    a2.set_title(
        "after the change, forgetting brings the error back down",
        color=INK,
        loc="left",
        fontsize=12,
    )
    a2.legend(loc="upper right", facecolor=PANEL, edgecolor=GRID, labelcolor=INK)
    a2.grid(True)
    fig.savefig(HERE / "rls_tracking.png", dpi=150, facecolor=BG)
    plt.close(fig)
    after = np.abs(np.array(pred["lam = 0.99"]) - true)[300:]
    print(
        "lam 0.99 back within 0.1 after",
        int(np.argmax(after < 0.1)),
        "payments; final errors",
        round(float(np.abs(np.array(pred["lam = 1 (never forgets)"]) - true)[-1]), 2),
        round(float(after[-1]), 2),
    )


if __name__ == "__main__":
    main()
