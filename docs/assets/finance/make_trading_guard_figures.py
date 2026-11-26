"""The figures of the trading_guard page, drawn with the library itself.

trading_guard_3d.png: the number of ticks the guard flags, for 25 guard thresholds and 25 sizes of the three shocks.
trading_guard_finance.png: the price of the stock index over 300 ticks, with the shocks, the ticks the guard flagged and
    the orders that left and the ones the pre-trade checker refused.

Run from the folder of the page:
    python make_trading_guard_figures.py
"""

import collections
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from dense_armor.core.finance import PreTradeChecker
from dense_armor.core.online.linear.glm import LinearRegression
from dense_armor.core.online.optim import SGD
from dense_armor.core.signals.stats.moments import RunningMoments
from dense_armor.roles.safety import HealthMonitor, SafeEstimator

HERE = Path(__file__).resolve().parent
BG, PANEL, GRID = "#0b1118", "#111a24", "#26364a"
INK, MUTED = "#e8fbff", "#8fa3b8"
CYAN, ORANGE, PINK, GREEN = "#3ec6e8", "#ffa21f", "#ff2d87", "#46d37f"
CMAP = LinearSegmentedColormap.from_list("s", [CYAN, "#6a5acd", PINK])


def simulate(guard=0.03, shock=(0.06, -0.05, 0.04), band=0.02, max_orders=40, n=300, seed=42):
    """One run of the loop of the page: returns the series and the counts."""
    shocks = {80: shock[0], 170: shock[1], 240: shock[2]}
    rng = random.Random(seed)
    norm = RunningMoments(feature="ret")
    model = LinearRegression(optimizer=SGD(lr=5e-3), l2=1e-3, intercept=True, feature_keys=["ret_n"])
    safe = SafeEstimator(model=model, guard=lambda x: abs(x["ret"]) > guard, fallback=0.0, monitor=HealthMonitor(window=50, warmup=20))
    checker = PreTradeChecker(max_notional=100_000.0, max_size=0.5, price_band=band, max_orders=max_orders, window_s=60.0, duplicate_window_s=1.0)
    prev, ref, last = 100.0, 100.0, 0.0
    out = {"price": [], "flagged": [], "orders": [], "refused": [], "reasons": collections.Counter()}
    for i in range(n):
        std = norm.std if norm.std > 1e-6 else 1.0
        x = {"ret": last, "ret_n": last / std}
        pred = safe.predict_one(x)
        if safe.is_flagged(x):
            out["flagged"].append(i)
        elif abs(pred) > 0.2:
            side = "buy" if pred > 0 else "sell"
            d = checker.check("acct1", side, min(abs(pred), 0.5), prev, ref, t=float(i))
            out["reasons"][d.reason] += 1
            (out["orders"] if d.approved else out["refused"]).append((i, side))
        ret = 0.3 * last + rng.gauss(0, 0.004) + shocks.get(i, 0.0)
        price = prev * (1 + ret)
        safe.learn_one(x, ret / std, t=float(i))
        norm.update({"ret": ret})
        ref = 0.9 * ref + 0.1 * price
        prev, last = price, ret
        out["price"].append(price)
    out["n_blocked"] = safe.n_blocked_
    out["shocks"] = sorted(shocks)
    return out


def surface():
    guards = np.linspace(0.005, 0.08, 25)
    sizes = np.linspace(0.01, 0.10, 25)
    Z = np.array([[len(simulate(guard=g, shock=(s, -s, s))["flagged"]) for g in guards] for s in sizes])
    X, Y = np.meshgrid(guards * 100, sizes * 100)
    fig = plt.figure(figsize=(12, 8.2))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(top=0.93, bottom=0.0, left=-0.05, right=1.05)
    ax.set_facecolor(BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.07, 0.10, 0.14, 1.0))
        axis._axinfo["grid"]["color"] = GRID
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1.35, 1.2, 0.7), zoom=1.1)
    ax.set_xlabel("guard threshold (% of one tick)", color=CYAN, labelpad=8)
    ax.set_ylabel("size of the three shocks (%)", color=CYAN, labelpad=8)
    ax.set_zlabel("ticks flagged by the guard", color=CYAN, labelpad=6)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.text(0.05, 0.955, "THE GUARD FLAGS THE TICK AFTER A JUMP", fontsize=21, fontweight="bold", color=INK)
    fig.text(0.05, 0.915, "a stock index over 300 ticks with three shocks: ticks flagged by the guard for 25 thresholds and 25 shock sizes", fontsize=11.5, color=CYAN)
    ax.plot_surface(X, Y, Z, facecolors=CMAP(np.clip(Z / max(Z.max(), 1), 0, 1)), rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=0.93)
    fig.savefig(HERE / "trading_guard_3d.png", dpi=150, facecolor=BG)
    plt.close(fig)
    pick = lambda arr, v: int(np.argmin(np.abs(arr - v)))
    for g, s in ((0.03, 0.06), (0.03, 0.02), (0.01, 0.06), (0.08, 0.10), (0.005, 0.01)):
        print("flagged ticks, guard", round(float(guards[pick(guards, g)]) * 100, 2), "% shock", round(float(sizes[pick(sizes, s)]) * 100, 2), "% ->", int(Z[pick(sizes, s), pick(guards, g)]))


def finance():
    r = simulate()
    fig, ax = plt.subplots(figsize=(11, 4.2), facecolor=BG)
    ax.set_facecolor(PANEL)
    t = np.arange(len(r["price"]))
    ax.plot(t, r["price"], color=CYAN, lw=1.6, label="index price")
    for s in r["shocks"]:
        ax.axvline(s, color=MUTED, lw=1.0, ls="--")
    pr = np.array(r["price"])
    if r["orders"]:
        ax.scatter([i for i, _ in r["orders"]], [pr[i] for i, _ in r["orders"]], s=10, color=GREEN, label="order sent", zorder=4)
    if r["refused"]:
        ax.scatter([i for i, _ in r["refused"]], [pr[i] for i, _ in r["refused"]], s=14, color=ORANGE, marker="x", label="refused by the checker", zorder=5)
    ax.scatter(r["flagged"], [pr[i] for i in r["flagged"]], s=90, color=PINK, marker="o", facecolors="none", linewidths=2, label="flagged by the guard", zorder=6)
    ax.set_xlabel("tick", color=CYAN)
    ax.set_ylabel("index price", color=CYAN)
    ax.tick_params(colors=MUTED)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(color=GRID, lw=0.5)
    ax.set_title("A stock index with three jumps: the guard, the orders and the checker", color=INK, loc="left", fontsize=13, fontweight="bold")
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=INK, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(HERE / "trading_guard_finance.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("loop:", "flagged", r["flagged"], "| n_blocked_", r["n_blocked"], "| reasons", dict(r["reasons"]), "| final price", round(r["price"][-1], 2))


if __name__ == "__main__":
    finance()
    surface()
