"""The figures of the pages of the robust filters, drawn with the library itself.

Each page has a 3D surface, the number of days that the function flags on a stream of the cost of executing an order (500 days
around 12, noise 1.5, one day at 25) against two settings of the function, and a finance figure of its own. The numbers printed
here are the ones written in the pages.

Run from the repo root:
    python docs/assets/robust_filters/make_robust_filters_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from dense_armor.core.online.anomaly.robust_filters import chauvenet_criterion, hampel_filter, pressure_valve, sigma_clip, tukey_fences

HERE = Path(__file__).resolve().parent
BG, GRID = "#0b1118", "#26364a"
INK, MUTED = "#e8fbff", "#8fa3b8"
CYAN, ORANGE, PINK, GREY = "#3ec6e8", "#ffa21f", "#ff2d87", "#8fa3b8"
CMAP = LinearSegmentedColormap.from_list("s", [CYAN, "#6a5acd", PINK])


def stream(seed=0, spike=25.0):
    x = 12.0 + 1.5 * np.random.default_rng(seed).standard_normal(500)
    x[250] = spike
    return x


def surface(Z, xs, ys, name, title, subtitle, xl, yl):
    X, Y = np.meshgrid(xs, ys)
    fig = plt.figure(figsize=(12, 8.2))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(top=0.93, bottom=0.0, left=-0.05, right=1.05)
    ax.set_facecolor(BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.07, 0.10, 0.14, 1.0))
        axis._axinfo["grid"]["color"] = GRID
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1.35, 1.2, 0.7), zoom=1.1)
    ax.set_xlabel(xl, color=CYAN, labelpad=8)
    ax.set_ylabel(yl, color=CYAN, labelpad=8)
    ax.set_zlabel("days flagged", color=CYAN, labelpad=6)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.text(0.05, 0.955, title, fontsize=21, fontweight="bold", color=INK)
    fig.text(0.05, 0.915, subtitle, fontsize=11.5, color=CYAN)
    top = float(Z.max()) or 1.0
    ax.plot_surface(X, Y, Z, facecolors=CMAP(np.clip(Z / top, 0, 1)), rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=0.93)
    ax.set_zlim(0, top)
    fig.savefig(HERE / f"{name}_3d.png", dpi=150, facecolor=BG)
    plt.close(fig)


def canvas(title, subtitle, xl, yl):
    fig, ax = plt.subplots(figsize=(12, 6.2))
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.08, right=0.97)
    ax.set_facecolor(BG)
    ax.grid(color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.set_xlabel(xl, color=CYAN)
    ax.set_ylabel(yl, color=CYAN)
    fig.text(0.05, 0.95, title, fontsize=18, fontweight="bold", color=INK)
    fig.text(0.05, 0.915, subtitle, fontsize=11, color=CYAN)
    return fig, ax


def save(fig, name):
    fig.savefig(HERE / f"{name}_finance.png", dpi=150, facecolor=BG)
    plt.close(fig)


def grid(fn, xs, ys, make, seeds=range(4)):
    return np.array([[np.mean([len(fn(make(seed, a, b))[1]) for seed in seeds]) for a in xs] for b in ys])


def main():
    radii = np.array([5, 10, 15, 20, 25, 30, 35, 40])
    cost = stream()

    ns = np.linspace(2.0, 6.0, 8)
    Z = grid(lambda x: hampel_filter(x, radius=15), ns, radii, lambda s, a, b: None) if False else np.array(
        [[np.mean([len(hampel_filter(stream(s), radius=int(r), n_sigmas=float(n))[1]) for s in range(4)]) for n in ns] for r in radii])
    surface(Z, ns, radii, "hampel_filter", "hampel_filter: the threshold", "days flagged out of 500, mean of 4 seeded streams, for 8 values of n_sigmas and 8 window radii", "n_sigmas", "radius of the window")
    clean, idx = hampel_filter(cost, radius=15)
    fig, ax = canvas("hampel_filter: the daily cost, before and after", "days 200 to 300 of the stream; the pink points are the days that the filter replaces", "day", "cost of executing an order")
    ax.plot(range(200, 300), cost[200:300], color=GREY, lw=1.2, label="raw")
    ax.plot(range(200, 300), clean[200:300], color=CYAN, lw=1.6, label="cleaned")
    sel = [i for i in idx if 200 <= i < 300]
    ax.scatter(sel, cost[sel], color=PINK, s=50, zorder=5)
    ax.legend(facecolor=BG, labelcolor=INK)
    save(fig, "hampel_filter")
    print("hampel_filter:", 250 in idx, round(float(clean[250]), 2), len(idx), idx)

    ks = np.linspace(0.5, 3.0, 8)
    Z = np.array([[np.mean([len(tukey_fences(stream(s), radius=int(r), k=float(k))[1]) for s in range(4)]) for k in ks] for r in radii])
    surface(Z, ks, radii, "tukey_fences", "tukey_fences: the fences", "days flagged out of 500, mean of 4 seeded streams, for 8 values of k and 8 window radii", "k, the fence multiplier", "radius of the window")
    clean, idx = tukey_fences(cost, radius=15)
    w = cost[235:266]
    q1, q3 = np.percentile(w, [25, 75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    fig, ax = canvas("tukey_fences: the window of day 250", "the 31 days centred on day 250; the orange band is the inside of the fences, the pink point is day 250", "day", "cost of executing an order")
    ax.axhspan(lo, hi, color=ORANGE, alpha=0.25)
    ax.scatter(range(235, 266), w, color=CYAN, s=35)
    ax.scatter([250], [cost[250]], color=PINK, s=80, zorder=5)
    save(fig, "tukey_fences")
    print("tukey_fences:", 250 in idx, round(float(clean[250]), 2), len(idx), "fences", round(float(lo), 2), round(float(hi), 2))

    spikes = np.linspace(14.0, 40.0, 8)
    Z = np.array([[np.mean([len(chauvenet_criterion(stream(s, spike=float(v)), radius=int(r))[1]) for s in range(4)]) for r in radii] for v in spikes])
    surface(Z, radii, spikes, "chauvenet_criterion", "chauvenet_criterion: the window", "days flagged out of 500, mean of 4 seeded streams, for 8 window radii and 8 sizes of the bad day", "radius of the window", "cost of the bad day")
    clean, idx = chauvenet_criterion(cost, radius=15)
    fig, ax = canvas("chauvenet_criterion: the days that it flags", "the daily cost in grey; every flagged day is a pink line from the usual level to its cost", "day", "cost of executing an order")
    ax.plot(cost, color=GREY, lw=0.8)
    ax.vlines(idx, 12.0, cost[idx], color=PINK, lw=1.6)
    ax.scatter(idx, cost[idx], color=PINK, s=40, zorder=5)
    save(fig, "chauvenet_criterion")
    print("chauvenet_criterion:", 250 in idx, round(float(clean[250]), 2), len(idx), idx)

    its = np.arange(1, 9)
    Z = np.array([[np.mean([len(sigma_clip(stream(s), radius=15, n_sigmas=float(n), max_iters=int(i))[1]) for s in range(4)]) for n in ns] for i in its])
    surface(Z, ns, its, "sigma_clip", "sigma_clip: the passes", "days flagged out of 500, mean of 4 seeded streams, for 8 values of n_sigmas and 1 to 8 passes", "n_sigmas", "max_iters, the number of passes")
    clean, idx = sigma_clip(cost, radius=15)
    fig, ax = canvas("sigma_clip: the costs, before and after", "the 500 daily costs as a histogram; in grey the raw costs, in cyan the cleaned ones", "cost of executing an order", "days")
    bins = np.linspace(cost.min(), cost.max(), 40)
    ax.hist(cost, bins=bins, color=GREY, alpha=0.7)
    ax.hist(clean, bins=bins, color=CYAN, alpha=0.8)
    save(fig, "sigma_clip")
    print("sigma_clip:", 250 in idx, round(float(clean[250]), 2), len(idx), idx)

    sp = np.linspace(4.0, 12.0, 6)
    km = np.linspace(0.0, 6.0, 6)
    Z = np.array([[np.mean([len(pressure_valve(stream(s), radius=15, soglia_pressione=float(a), k_molla=float(b))[1]) for s in range(2)]) for a in sp] for b in km])
    surface(Z, sp, km, "pressure_valve", "pressure_valve: the threshold", "days flagged out of 500, mean of 2 seeded streams, for 6 values of soglia_pressione and 6 values of k_molla", "soglia_pressione", "k_molla")
    clean, idx, p, thr = pressure_valve(cost, radius=15)
    fig, ax = canvas("pressure_valve: the pressure and the threshold", "the pressure of each day in orange, the threshold in grey, the flagged days in pink", "day", "pressure")
    ax.plot(p, color=ORANGE, lw=1.0)
    ax.plot(thr, color=GREY, ls="--", lw=1.2)
    ax.scatter(idx, p[idx], color=PINK, s=50, zorder=5)
    save(fig, "pressure_valve")
    print("pressure_valve:", idx, round(float(p[250]), 1), round(float(np.median(thr)), 2))


if __name__ == "__main__":
    main()
