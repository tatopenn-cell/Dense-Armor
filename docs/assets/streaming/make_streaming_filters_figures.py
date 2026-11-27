"""The figures of the pages of the four classic rules, of the Hampel filter and of the batch function, drawn with the library.

Each page has a 3D surface that shows what is specific to its rule and a finance figure with its own example:
HampelScorer (robust to bad values in the window), TukeyScorer (the fences and the plateau), ChauvenetScorer (the score
depends on the window size), SigmaClipScorer (the passes that clean the window), HampelFilter (the output of the filter) and
class_seg_multi (spike or regime by the length of the run). Every number printed here is the one written in the pages.

Run from the repo root:
    python docs/assets/streaming/make_streaming_filters_figures.py
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from dense_armor.core.online.anomaly.filters import ChauvenetScorer, HampelFilter, HampelScorer, SigmaClipScorer, TukeyScorer
from dense_armor.core.online.anomaly.streaming import class_seg_multi

HERE = Path(__file__).resolve().parent
BG, GRID = "#0b1118", "#26364a"
INK, MUTED = "#e8fbff", "#8fa3b8"
CYAN, ORANGE, PINK, GREY = "#3ec6e8", "#ffa21f", "#ff2d87", "#8fa3b8"
CMAP = LinearSegmentedColormap.from_list("s", [CYAN, "#6a5acd", PINK])
SEEDS = range(10)


def surface(Z, xs, ys, name, title, subtitle, xl, yl, zl, zlim, plane=None):
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
    ax.set_zlabel(zl, color=CYAN, labelpad=6)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.text(0.05, 0.955, title, fontsize=21, fontweight="bold", color=INK)
    fig.text(0.05, 0.915, subtitle, fontsize=11.5, color=CYAN)
    lo, hi = zlim
    ax.plot_surface(X, Y, Z, facecolors=CMAP(np.clip((Z - lo) / (hi - lo), 0, 1)), rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=0.93)
    if plane is not None:
        ax.plot_surface(X, Y, np.full_like(Z, plane), color=ORANGE, alpha=0.25, linewidth=0)
    ax.set_zlim(lo, hi)
    fig.savefig(HERE / f"{name}_3d.png", dpi=150, facecolor=BG)
    plt.close(fig)


def canvas(title, subtitle, xl, yl, figsize=(12, 6.2)):
    fig, ax = plt.subplots(figsize=figsize)
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


def learn(scorer, values):
    for v in values:
        scorer.learn_one({"x": float(v)})
    return scorer


def hampel():
    bad = np.arange(0, 15, 2)
    amps = np.linspace(4.0, 12.0, 8)

    def cell(k, amp, seed):
        w = np.random.default_rng(seed).normal(0.0, 1.0, 30)
        w[:k] = 20.0
        return HampelScorer(radius=15).__class__(radius=15) if False else learn(HampelScorer(radius=15), w).score_one({"x": float(amp)})

    Z = np.array([[np.mean([cell(k, a, s) for s in SEEDS]) for k in bad] for a in amps])
    surface(Z, bad, amps, "hampel_scorer", "HampelScorer: bad values in the window", "mean score of a spike, 10 seeded windows of 30 values: the median and the MAD hold until half the window is bad",
            "bad values in the window of 30", "spike amplitude (standard deviations)", "score_one", (0, float(Z.max())))
    rng = np.random.default_rng(1)
    cost = 12.0 + 1.5 * rng.standard_normal(30)
    cost[[5, 11, 17, 23]] = 40.0
    h = learn(HampelScorer(radius=15), cost)
    med = np.median(cost)
    s = 1.4826 * np.median(np.abs(cost - med))
    mu, sd = cost.mean(), cost.std()
    fig, ax = canvas("HampelScorer: the window of 30 days", "four bad ticks at 40 in the window; the new day costs 25", "day", "cost of executing an order")
    ax.scatter(range(30), cost, color=CYAN, s=30)
    ax.scatter([5, 11, 17, 23], cost[[5, 11, 17, 23]], color=PINK, s=60, zorder=5)
    ax.axhspan(med - 3 * s, med + 3 * s, color=ORANGE, alpha=0.25)
    ax.axhspan(mu - 3 * sd, mu + 3 * sd, color=GREY, alpha=0.18)
    ax.scatter([30], [25.0], color=INK, marker="D", s=70, zorder=6)
    ax.text(0.5, 0.05, "orange: median +/- 3 robust deviations; grey: mean +/- 3 standard deviations", color=MUTED, transform=ax.transAxes, fontsize=10)
    save(fig, "hampel_scorer")
    print("hampel: score", round(h.score_one({"x": 25.0}), 1), "z", round(abs(25.0 - mu) / sd, 1), "bands", round(med - 3 * s, 1), round(med + 3 * s, 1), round(mu - 3 * sd, 1), round(mu + 3 * sd, 1))


def tukey():
    vals = np.linspace(-6.0, 6.0, 9)
    ks = np.linspace(0.5, 3.0, 8)

    def cell(v, k, seed):
        return learn(TukeyScorer(radius=15, k=k), np.random.default_rng(seed).normal(0.0, 1.0, 30)).score_one({"x": float(v)})

    Z = np.array([[np.mean([cell(v, k, s) for s in SEEDS]) for v in vals] for k in ks])
    surface(Z, vals, ks, "tukey_scorer", "TukeyScorer: the fences", "mean score of a value (in standard deviations) against 10 seeded windows of 30 values: zero inside the fences",
            "value (standard deviations)", "k, the fence multiplier", "score_one", (0, float(Z.max())))
    size = np.random.default_rng(2).lognormal(mean=4.0, sigma=0.6, size=60)
    t = learn(TukeyScorer(radius=30, k=1.5), size)
    q1, q3 = np.percentile(size, [25, 75])
    fence = q3 + 1.5 * (q3 - q1)
    top = size.mean() + 3 * size.std()
    fig, ax = canvas("TukeyScorer: order sizes", "60 order sizes, skewed to the right; the new order is 200", "order size (shares)", "orders")
    ax.hist(size, bins=18, color=CYAN, alpha=0.8)
    ax.axvline(fence, color=ORANGE, lw=2)
    ax.axvline(top, color=GREY, lw=2, ls="--")
    ax.axvline(200.0, color=PINK, lw=2)
    ax.text(0.55, 0.9, "orange: upper Tukey fence; grey: mean + 3 standard deviations; pink: the new order", color=MUTED, transform=ax.transAxes, fontsize=10)
    save(fig, "tukey_scorer")
    print("tukey: fence", round(fence, 1), "mean+3sd", round(top, 1), "score(200)", round(t.score_one({"x": 200.0}), 2), t.is_outlier({"x": 200.0}))


def chauvenet():
    zs = np.linspace(0.5, 5.0, 8)
    radii = np.array([5, 10, 20, 40, 60, 90, 120, 150])

    def cell(z, radius, seed):
        c = learn(ChauvenetScorer(radius=int(radius)), np.random.default_rng(seed).normal(0.0, 1.0, 2 * int(radius)))
        return c.score_one({"x": float(z)})

    Z = np.array([[math.log10(max(np.mean([cell(z, r, s) for s in SEEDS]), 1e-4)) for z in zs] for r in radii])
    surface(Z, zs, 2 * radii, "chauvenet_scorer", "ChauvenetScorer: the window size", "log10 of the score N times P, for a value z standard deviations away; the orange plane is the threshold 0.5",
            "z, standard deviations from the mean", "window size N", "log10 of the score", (-4, 3), plane=math.log10(0.5))
    rng = np.random.default_rng(3)
    out = {}
    for radius in (15, 150):
        c = learn(ChauvenetScorer(radius=radius), rng.normal(12.0, 1.5, 2 * radius))
        out[2 * radius] = (round(c.score_one({"x": 16.5}), 2), c.is_outlier({"x": 16.5}))
    fig, ax = canvas("ChauvenetScorer: N times P", "the expected number of values this far from the mean, for two window sizes", "z, standard deviations from the mean", "expected count")
    z = np.linspace(0.5, 5.0, 200)
    for n, col in ((30, ORANGE), (300, CYAN)):
        ax.plot(z, [n * math.erfc(v / math.sqrt(2)) for v in z], color=col, lw=2, label=f"N = {n}")
        ax.scatter([3.0], [out[n][0]], color=PINK, zorder=5)
    ax.axhline(0.5, color=GREY, ls="--")
    ax.set_yscale("log")
    ax.legend(facecolor=BG, labelcolor=INK)
    save(fig, "chauvenet_scorer")
    print("chauvenet:", out)


def sigmaclip():
    bad = np.arange(0, 15, 2)
    iters = np.arange(1, 9)

    def window(k, seed):
        r = np.random.default_rng(seed)
        w = 12.0 + 1.5 * r.standard_normal(40)
        w[:k] = r.uniform(45.0, 55.0, k)
        return w

    def cell(k, it, seed):
        return learn(SigmaClipScorer(radius=20, n_sigmas=3.0, max_iters=int(it)), window(k, seed)).score_one({"x": 18.0})

    Z = np.array([[np.mean([cell(k, it, s) for s in SEEDS]) for k in bad] for it in iters])
    surface(Z, bad, iters, "sigma_clip_scorer", "SigmaClipScorer: the passes", "mean score of a cost of 18, 10 seeded windows of 40 days: more passes clean more of the bad ticks",
            "bad ticks in the window of 40", "max_iters, the number of passes", "score_one", (0, float(Z.max())))
    rng = np.random.default_rng(4)
    cost = 12.0 + 1.5 * rng.standard_normal(40)
    cost[[3, 9, 15, 22, 31]] = [45.0, 50.0, 48.0, 55.0, 47.0]
    keep = np.ones(40, bool)
    pas = np.zeros(40, int)
    stds = []
    for p in range(1, 6):
        mu, sd = cost[keep].mean(), cost[keep].std()
        stds.append(round(float(sd), 2))
        drop = keep & (np.abs(cost - mu) > 3.0 * sd)
        if not drop.any():
            break
        pas[drop] = p
        keep &= ~drop
    fig, ax = canvas("SigmaClipScorer: the passes on 40 days", "cyan days stay; the colours mark the pass that removes a day; the diamond is the new day, 18", "day", "cost of executing an order")
    ax.scatter(np.where(keep)[0], cost[keep], color=CYAN, s=30)
    for p, col in ((1, ORANGE), (2, PINK), (3, INK)):
        idx = np.where(pas == p)[0]
        ax.scatter(idx, cost[idx], color=col, s=70, zorder=5, label=f"removed in pass {p}")
    ax.scatter([40], [18.0], color=INK, marker="D", s=70, zorder=6)
    ax.legend(facecolor=BG, labelcolor=INK)
    save(fig, "sigma_clip_scorer")
    res = {it: (round(learn(SigmaClipScorer(radius=20, n_sigmas=3.0, max_iters=it), cost).score_one({"x": 18.0}), 2)) for it in (1, 5)}
    print("sigma: spread after each pass", stds, "| passes that remove:", sorted(set(pas.tolist()) - {0}), "| score of 18:", res)


def hfilter():
    vals = np.linspace(-10.0, 10.0, 9)
    ns = np.linspace(2.0, 6.0, 8)

    def cell(v, n, seed):
        f = HampelFilter(radius=15, n_sigmas=float(n))
        for x in np.random.default_rng(seed).normal(0.0, 1.0, 30):
            f.learn_one({"x": float(x)})
        return f.transform_one({"x": float(v)})["x"]

    Z = np.array([[np.mean([cell(v, n, s) for s in SEEDS]) for v in vals] for n in ns])
    surface(Z, vals, ns, "hampel_filter", "HampelFilter: what comes out", "mean output for a value (in standard deviations), 10 seeded windows of 30 values: the value itself, or the window median",
            "input value (standard deviations)", "n_sigmas", "output", (-10, 10))
    rng = np.random.default_rng(0)
    cost = 12.0 + 1.5 * rng.standard_normal(500)
    cost[250] = 25.0
    f = HampelFilter(radius=15)
    clean = []
    for v in cost:
        clean.append(f.transform_one({"x": float(v)})["x"])
        f.learn_one({"x": float(v)})
    changed = [i for i, (a, b) in enumerate(zip(clean, cost)) if a != b]
    fig, ax = canvas("HampelFilter: the daily cost, before and after", "days 200 to 300; the filter swaps the days it flags for the window median", "day", "cost of executing an order")
    ax.plot(range(200, 300), cost[200:300], color=GREY, lw=1.2, label="raw")
    ax.plot(range(200, 300), clean[200:300], color=CYAN, lw=1.6, label="cleaned")
    ax.scatter([i for i in changed if 200 <= i < 300], [cost[i] for i in changed if 200 <= i < 300], color=PINK, s=50, zorder=5)
    ax.legend(facecolor=BG, labelcolor=INK)
    save(fig, "hampel_filter")
    print("filter: clean[250]", round(clean[250], 2), "days changed", len(changed), changed[:12])


def segments():
    runs = np.array([2, 6, 12, 18, 24, 36, 48, 60])
    amps = np.array([3.0, 5.0, 8.0, 11.0, 15.0])

    def cell(run, amp, seed):
        x = 12.0 + 1.5 * np.random.default_rng(seed).standard_normal((600, 1))
        x[200:200 + run, 0] += amp
        lab = class_seg_multi(x, radius=10)[0][200:200 + run, 0].tolist()
        return lab.count("regime") / run

    Z = np.array([[np.mean([cell(r, a, s) for s in range(5)]) for r in runs] for a in amps])
    surface(Z, runs, amps, "class_seg_multi", "class_seg_multi: spike or regime", "share of a shifted run labelled regime, 5 seeded channels: a short run is a spike, a long one becomes a regime",
            "length of the shifted run (days)", "size of the shift (standard deviations)", "share labelled regime", (0, 1))
    rng = np.random.default_rng(5)
    X = 12.0 + 1.5 * rng.standard_normal((400, 3))
    X[100, 0] = 40.0
    X[300:, 2] += 9.0
    labels = class_seg_multi(X, radius=10)[0]
    names = ["venue A", "venue B", "venue C"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.4), sharex=True)
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.08, right=0.97, hspace=0.12)
    fig.text(0.05, 0.95, "class_seg_multi: three venues", fontsize=18, fontweight="bold", color=INK)
    fig.text(0.05, 0.915, "the cost of executing an order on three venues; pink days are spikes, orange days are a regime", fontsize=11, color=CYAN)
    for c, ax in enumerate(axes):
        ax.set_facecolor(BG)
        ax.grid(color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.plot(X[:, c], color=CYAN, lw=0.9)
        for lab, col in (("spike", PINK), ("regime", ORANGE)):
            idx = [i for i, v in enumerate(labels[:, c]) if v == lab]
            ax.scatter(idx, X[idx, c], color=col, s=18, zorder=5)
        ax.set_ylabel(names[c], color=CYAN)
    axes[-1].set_xlabel("day", color=CYAN)
    fig.savefig(HERE / "class_seg_multi_finance.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("segments:", [{k: labels[:, c].tolist().count(k) for k in ("spike", "regime")} for c in range(3)])


if __name__ == "__main__":
    hampel()
    tukey()
    chauvenet()
    sigmaclip()
    hfilter()
    segments()
