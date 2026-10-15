"""Figures for the finance docs pages.

Market pages (surveillance, risk, execution, single ticker, static vs online):
the figures read ``results.json`` in this folder, written by running the
pipelines on the fixtures in ``test/fixtures/finance``. If the file is missing,
write it first with ``measure_market.py`` (several minutes).

Fraud, unsupervised fraud, intrusion and credit scoring: the figures draw the
counts reported on their pages, taken from the run of
``test_slow/finance/test_measurements.py`` on the downloaded datasets.

Run from the repo root:
    python docs/assets/finance/make_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DPI = 130
GREEN, RED, GRAY, DARK = "#2e8b57", "#c0392b", "#b8b8b8", "#333333"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.labelsize": 10,
    }
)


def _load():
    path = HERE / "results.json"
    if not path.exists():
        raise SystemExit(
            "results.json missing: run the pipelines first (see the module docstring)"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _returns_figure(name, dates, ret, flags, title, start=0, hl=None):
    ret = np.asarray(ret) * 100
    x = np.arange(len(ret))
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(x, ret, color=GRAY, lw=0.8)
    idx = [i for i in range(len(flags)) if flags[i] and i >= start]
    ax.scatter(idx, ret[idx], color=RED, s=22, zorder=3)
    if start:
        ax.axvspan(0, start, color="#eeeeee", zorder=0)
        ax.text(
            start / 2,
            ax.get_ylim()[1] * 0.92,
            "warm-up",
            ha="center",
            va="top",
            color="#777777",
            fontsize=9,
        )
    ax.axhline(0, color=DARK, lw=0.5)
    if hl:
        for day in hl:
            i = dates.index(day) - 1
            ax.annotate(
                f"{day}\n{ret[i]:+.2f}%",
                (i, ret[i]),
                xytext=(0, 14 if ret[i] > 0 else -30),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color=DARK,
                arrowprops={"arrowstyle": "-", "color": DARK, "lw": 0.6},
            )
    ticks = list(range(0, len(ret), 250))
    ax.set_xticks(ticks)
    ax.set_xticklabels([dates[i + 1][:7] for i in ticks])
    ax.set_ylabel("daily return (%)")
    ax.set_title(f"{title}: {len(idx)} red days", loc="left")
    fig.tight_layout()
    return fig


def _two_rows(fname, title, rows):
    fig, ax = plt.subplots(figsize=(8.5, 2.6))
    for k, (label, good, bad, good_label, bad_label) in enumerate(rows):
        y = len(rows) - 1 - k
        total = good + bad
        ax.barh(y, good, color=GREEN, height=0.55)
        ax.barh(y, bad, left=good, color=RED, height=0.55)
        ax.text(
            good / 2,
            y,
            f"{good:,} {good_label}",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold",
        )
        if bad / total > 0.15:
            ax.text(
                good + bad / 2,
                y,
                f"{bad:,} {bad_label}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontweight="bold",
            )
        else:
            ax.text(
                total + total * 0.01,
                y,
                f"{bad:,} {bad_label}",
                ha="left",
                va="center",
                color=RED,
                fontsize=9,
                fontweight="bold",
            )
        ax.text(0, y + 0.42, label, ha="left", va="bottom", fontsize=9.5, color=DARK)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_ylim(-0.5, len(rows) - 0.1)
    ax.set_xlim(0, max(g + b for _, g, b, _, _ in rows) * 1.2)
    ax.set_title(title, loc="left")
    fig.tight_layout()
    fig.savefig(HERE / fname, dpi=DPI)
    plt.close(fig)


def market():
    R = _load()

    s = R["surv_GSPC"]
    fig = _returns_figure(
        "surveillance",
        s["dates"],
        s["ret"],
        s["anomalous"],
        "S&P 500, five years, the days the five tools flagged",
        start=s["post_from"],
        hl=["2024-08-05", "2025-04-04"],
    )
    fig.savefig(HERE / "surveillance.png", dpi=DPI)
    plt.close(fig)

    a = R["single_AAPL"]
    fig = _returns_figure(
        "single_ticker",
        a["dates"],
        a["ret"],
        a["anomalous"],
        "Apple, five years, the strange days",
        hl=["2025-04-03", "2025-04-09"],
    )
    fig.savefig(HERE / "single_ticker.png", dpi=DPI)
    plt.close(fig)

    so = R["static_online"]
    ret = np.asarray(so["ret"]) * 100
    warm = 249
    st = sorted(i for i in so["static_idx"] if i >= warm)
    on = sorted(i - 1 for i in so["online_idx"] if i - 1 >= warm)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, idx, title in (
        (axes[0], st, f"Fixed 3% cutoff: {len(st)} red days"),
        (axes[1], on, f"Adaptive detectors: {len(on)} red days"),
    ):
        ax.plot(ret, color=GRAY, lw=0.7)
        ax.scatter(idx, ret[idx], color=RED, s=20, zorder=3)
        ax.axhline(0, color=DARK, lw=0.5)
        ax.axvspan(0, warm, color="#eeeeee", zorder=0)
        ax.set_title(title, loc="left")
        ax.set_xticks([0, 250, 500, 750, 1000, 1250])
        ax.set_xticklabels(
            [so["dates"][i + 1][:7] for i in (0, 250, 500, 750, 1000, 1250)]
        )
    axes[0].axhline(3, color=RED, lw=0.8, ls="--")
    axes[0].axhline(-3, color=RED, lw=0.8, ls="--")
    axes[0].set_ylabel("daily return (%)")
    fig.tight_layout()
    fig.savefig(HERE / "static_vs_online.png", dpi=DPI)
    plt.close(fig)

    rk = R["risk"]
    loss = np.asarray(rk["loss"]) * 100
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.bar(np.arange(len(loss)), loss, color=GRAY, width=1.0)
    ax.axhline(rk["var99"] * 100, color=RED, lw=1.4, ls="--")
    n_above = int((loss > rk["var99"] * 100).sum())
    ax.text(
        len(loss) * 0.01,
        rk["var99"] * 100 + 0.15,
        f"worst 1% of days start here: {rk['var99'] * 100:.2f}% ({n_above} days above)",
        color=RED,
        fontsize=9.5,
        fontweight="bold",
    )
    ticks = list(range(0, len(loss), 250))
    ax.set_xticks(ticks)
    ax.set_xticklabels([rk["dates"][i + 1][:7] for i in ticks])
    ax.set_ylabel("daily loss (%)")
    ax.set_title("S&P 500, five years: the loss of each day", loc="left")
    fig.tight_layout()
    fig.savefig(HERE / "risk.png", dpi=DPI)
    plt.close(fig)

    ex = R["execution"]
    approved = np.asarray(ex["approved"])
    n = len(approved)
    refused = int((~approved).sum())
    reasons = ex["reason"]
    n_orca = sum(1 for r in reasons if r == "passed through Orca")
    direct = sum(1 for r in reasons if r.startswith("passed without Orca"))
    assert direct + n_orca + refused == n
    fig, ax = plt.subplots(figsize=(8.5, 2.2))
    left = 0
    for val, col, lab, dy in (
        (direct, GREEN, f"{direct} sent as they are", 0),
        (n_orca, GRAY, f"{n_orca} checked by Orca", -0.42),
        (refused, RED, f"{refused} refused", 0.42),
    ):
        ax.barh(0, val, left=left, color=col, height=0.5)
        if val > 100:
            ax.text(
                left + val / 2,
                0,
                lab,
                ha="center",
                va="center",
                color="white",
                fontsize=9.5,
                fontweight="bold",
            )
        else:
            ax.annotate(
                lab,
                (left + val / 2, dy * 0.6),
                xytext=(left + val / 2 - 120, dy * 1.3),
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="#777777" if col == GRAY else col,
                arrowprops={"arrowstyle": "-", "color": "#999999", "lw": 0.6},
            )
        left += val
    ax.set_ylim(-0.75, 0.75)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title(f"What happened to {n:,} real BTC/USD orders", loc="left")
    fig.tight_layout()
    fig.savefig(HERE / "execution.png", dpi=DPI)
    plt.close(fig)


def counts():
    _two_rows(
        "fraud.png",
        "Fraud detection on 10,000 test transactions",
        [
            ("Of the 102 real frauds", 69, 33, "caught", "missed"),
            ("Of the 72 alerts raised", 69, 3, "real frauds", "false alarms"),
        ],
    )

    fig, ax = plt.subplots(figsize=(8.5, 2.9))
    names = [
        "Supervised (needs labels)",
        "Ensemble, no labels",
        "LODA",
        "Mahalanobis",
        "Half-Space Trees",
    ]
    vals = [0.969, 0.959, 0.966, 0.957, 0.935]
    cols = [DARK, GREEN, GRAY, GRAY, GRAY]
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=cols, height=0.55)
    ax.axvline(0.5, color=RED, lw=0.8, ls="--")
    for yy, v in zip(y, vals):
        ax.text(v + 0.004, yy, f"{v:.3f}", va="center", fontsize=9.5, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0.45, 1.05)
    ax.set_xticks([0.5, 1.0])
    ax.set_xticklabels(["0.5 = guessing", "1.0 = perfect"])
    ax.set_title(
        "Fraud without labels: quality score on the same 10,000 transactions",
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(HERE / "unsupervised_fraud.png", dpi=DPI)
    plt.close(fig)

    _two_rows(
        "intrusion.png",
        "Intrusion detection on 25,195 test network flows",
        [
            ("Of the 11,704 attacks", 10954, 750, "caught", "missed"),
            (
                "Of the 12,875 alerts raised",
                10954,
                1921,
                "real attacks",
                "false alarms",
            ),
        ],
    )

    _two_rows(
        "credit_scoring.png",
        "Telemarketing on 9,043 test calls, threshold 0.25",
        [
            ("Of the 1,048 customers who would sign", 779, 269, "reached", "missed"),
            ("Of the 1,991 calls made", 779, 1212, "signed", "did not sign"),
        ],
    )


def main():
    counts()
    if (HERE / "results.json").exists():
        market()


if __name__ == "__main__":
    main()
