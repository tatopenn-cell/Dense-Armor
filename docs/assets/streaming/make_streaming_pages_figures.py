"""The 3D figures of the Streaming anomaly detection pages, drawn with the library itself.

Same style as docs/assets/drift/make_detectors_style_figures.py (colours, `surface`, title and subtitle).
Run from the repo root:
    python docs/assets/streaming/make_streaming_pages_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from dense_armor.core.online.anomaly.deviation import streamDS
from dense_armor.core.online.anomaly.filters import ChauvenetScorer, HampelScorer, SigmaClipScorer, TukeyScorer
from dense_armor.core.online.anomaly.streaming import MultistreamDD
from dense_armor.core.online.anomaly.streaming import streamDD

HERE = Path(__file__).resolve().parent
BG, GRID = "#0b1118", "#26364a"
INK, MUTED = "#e8fbff", "#8fa3b8"
CYAN, ORANGE, PINK = "#3ec6e8", "#ffa21f", "#ff2d87"
CMAP = LinearSegmentedColormap.from_list("s", [CYAN, "#6a5acd", PINK])
SEEDS = range(20)
AMPS = np.linspace(2.0, 10.0, 8)
SIGMAS = np.linspace(2.0, 6.0, 8)
LEN, SPIKE = 200, 100
SPIKE_F = 250
RADIUS, REF_MULT = 30, 3


def spike_flagged(amp, n_sigmas, seed):
    r = np.random.default_rng(seed)
    y = r.normal(0.0, 1.0, LEN)
    y[SPIKE] = amp
    det = streamDD(radius=RADIUS, ref_mult=REF_MULT, n_sigmas=n_sigmas)
    return bool([det.update(float(v)) for v in y][SPIKE])


def grid_fraction(amps, sigmas):
    return np.array([[np.mean([spike_flagged(a, s, k) for k in SEEDS]) for a in amps] for s in sigmas])


def surface(Z, xvals, yvals, name, title, subtitle, yl, xl, zl):
    X, Y = np.meshgrid(xvals, yvals)
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
    ax.plot_surface(X, Y, Z, facecolors=CMAP(np.clip(Z, 0, 1)), rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=0.93)
    ax.set_zlim(0, 1)
    fig.savefig(HERE / f"{name}_3d.png", dpi=150, facecolor=BG)
    plt.close(fig)


def finance_figure():
    rng = np.random.default_rng(0)
    cost = 12.0 + 1.5 * rng.standard_normal(500)
    cost[SPIKE_F] = 25.0
    det = streamDD(radius=RADIUS, ref_mult=REF_MULT, n_sigmas=4.0)
    flags = [i for i, v in enumerate(cost) if det.update(float(v))]
    fig, ax = plt.subplots(figsize=(12, 6.2))
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.08, right=0.97)
    ax.set_facecolor(BG)
    ax.grid(color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.plot(cost, color=CYAN, lw=1.0)
    ax.scatter(flags, cost[flags], color=PINK, s=40, zorder=5)
    ax.set_xlabel("day", color=CYAN)
    ax.set_ylabel("daily cost of executing an order", color=CYAN)
    fig.text(0.05, 0.95, "streamDD: daily cost of executing an order", fontsize=18, fontweight="bold", color=INK)
    fig.text(0.05, 0.915, "500 seeded days around 12 with noise 1.5, one spike at day 250 to 25; flagged days in pink", fontsize=11, color=CYAN)
    out = HERE / "streaming_deviation_detector_finance.png"
    fig.savefig(out, dpi=150, facecolor=BG)
    plt.close(fig)
    print("wrote", out, "exists", out.exists())
    print("flagged days", flags)


def scorer_spike_score(amp, radius, seed):
    r = np.random.default_rng(seed)
    y = r.normal(0.0, 1.0, LEN)
    y[SPIKE] = amp
    sc = streamDS(radius=radius, ref_mult=REF_MULT)
    score = 0.0
    for i, v in enumerate(y):
        score = sc.score_one({"x": float(v)})
        sc.learn_one({"x": float(v)})
        if i == SPIKE:
            return score
    return score


def streaming_deviation_scorer_3d():
    radii = np.linspace(3.0, 20.0, 8)
    Z = np.array([[np.mean([scorer_spike_score(a, int(r), k) for k in SEEDS]) for a in AMPS] for r in radii])
    sub = "mean score_one on the injected spike on 20 seeded series; x = spike amplitude (standard deviations), y = radius"
    surface(Z, AMPS, radii, "streaming_deviation_scorer",
            "streamDS: score on the spike against radius", sub,
            "radius", "spike amplitude (standard deviations)", "mean score_one on the spike")
    out = HERE / "streaming_deviation_scorer_3d.png"
    print("wrote", out, "exists", out.exists())
    print("grid min", round(float(Z.min()), 4), "max", round(float(Z.max()), 4))
    print("Z row radius=3:", [round(float(v), 4) for v in Z[0]])
    print("Z row radius=20:", [round(float(v), 4) for v in Z[-1]])


def streaming_deviation_scorer_finance():
    rng = np.random.default_rng(0)
    cost = 12.0 + 1.5 * rng.standard_normal(500)
    cost[SPIKE_F] = 25.0
    sc = streamDS(radius=RADIUS, ref_mult=REF_MULT)
    scores = []
    for v in cost:
        scores.append(sc.score_one({"x": float(v)}))
        sc.learn_one({"x": float(v)})
    n_sigmas = 4.0
    flagged = [i for i, s in enumerate(scores) if s > n_sigmas]
    x = np.arange(len(cost))
    fig, ax = plt.subplots(figsize=(12, 6.2))
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.08, right=0.97)
    ax.set_facecolor(BG)
    ax.grid(color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.plot(cost, color=CYAN, lw=1.0, label="daily cost")
    ax.plot(x, scores, color=ORANGE, lw=1.0, label="score_one")
    ax.axhline(n_sigmas, color=PINK, lw=1.0, ls="--", label="n_sigmas=4.0")
    ax.scatter(flagged, [scores[i] for i in flagged], color=PINK, s=40, zorder=5)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=INK)
    ax.set_xlabel("day", color=CYAN)
    ax.set_ylabel("daily cost / score", color=CYAN)
    fig.text(0.05, 0.95, "streamDS: daily cost of executing an order", fontsize=18, fontweight="bold", color=INK)
    fig.text(0.05, 0.915, "500 seeded days around 12 with noise 1.5, one spike at day 250 to 25; score_one in orange; days above n_sigmas=4.0 in pink", fontsize=11, color=CYAN)
    out = HERE / "streaming_deviation_scorer_finance.png"
    fig.savefig(out, dpi=150, facecolor=BG)
    plt.close(fig)
    print("wrote", out, "exists", out.exists())
    print("score at spike day 250", round(float(scores[SPIKE_F]), 4))
    print("days above n_sigmas=4.0", flagged)


def multi_spike_flagged(amp, n_sigmas, seed, n_channels=3):
    r = np.random.default_rng(seed)
    y = r.normal(0.0, 1.0, (LEN, n_channels))
    y[SPIKE, 0] = amp
    det = MultistreamDD(n_channels=n_channels, radius=RADIUS, ref_mult=REF_MULT, n_sigmas=n_sigmas)
    flags = [det.update(row) for row in y]
    return bool(flags[SPIKE][0])


def multi_channel_3d():
    Z = np.array([[np.mean([multi_spike_flagged(a, s, k) for k in SEEDS]) for a in AMPS] for s in SIGMAS])
    sub = "fraction of injected spikes flagged on the injected channel on 20 seeded 3-channel series; x = spike amplitude (standard deviations), y = n_sigmas"
    surface(Z, AMPS, SIGMAS, "multi_channel_streaming_deviation_detector",
            "MultistreamDD: fraction of spikes flagged", sub,
            "n_sigmas", "spike amplitude (standard deviations)", "fraction of injected spikes flagged")
    out = HERE / "multi_channel_streaming_deviation_detector_3d.png"
    print("wrote", out, "exists", out.exists())
    print("grid min", round(float(Z.min()), 4), "max", round(float(Z.max()), 4))
    print("Z row n_sigmas=2:", [round(float(v), 4) for v in Z[0]])
    print("Z row n_sigmas=6:", [round(float(v), 4) for v in Z[-1]])


def multi_channel_finance():
    rng = np.random.default_rng(0)
    base = 12.0 + 1.5 * rng.standard_normal((500, 3))
    base[SPIKE_F, 1] = 25.0
    det = MultistreamDD(n_channels=3, radius=RADIUS, ref_mult=REF_MULT, n_sigmas=4.0)
    flags = np.array([det.update(row) for row in base])
    x = np.arange(len(base))
    fig, ax = plt.subplots(figsize=(12, 6.2))
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.08, right=0.97)
    ax.set_facecolor(BG)
    ax.grid(color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    names = ["cost A", "cost B", "cost C"]
    colors = [CYAN, ORANGE, "#6a5acd"]
    for j in range(3):
        ax.plot(base[:, j], color=colors[j], lw=0.9, label=names[j])
    hits = np.argwhere(flags[10:])
    ax.scatter(10 + hits[:, 0], base[10 + hits[:, 0], hits[:, 1]], color=PINK, s=40, zorder=5, label="flagged")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=INK)
    ax.set_xlabel("day", color=CYAN)
    ax.set_ylabel("daily cost of executing an order", color=CYAN)
    fig.text(0.05, 0.95, "MultistreamDD: three order-cost streams", fontsize=18, fontweight="bold", color=INK)
    fig.text(0.05, 0.915, "500 seeded days around 12 with noise 1.5 per channel, one spike at day 250 to 25 on channel 1; flagged in pink", fontsize=11, color=CYAN)
    out = HERE / "multi_channel_streaming_deviation_detector_finance.png"
    fig.savefig(out, dpi=150, facecolor=BG)
    plt.close(fig)
    print("wrote", out, "exists", out.exists())
    print("flag at day 250", flags[SPIKE_F].tolist())
    print("flags after day 10", np.argwhere(flags[10:]).tolist())


def scorer_flagged(cls, amp, param, seed, pname):
    r = np.random.default_rng(seed)
    y = r.normal(0.0, 1.0, LEN)
    y[SPIKE] = amp
    s = cls(radius=RADIUS, **{pname: param})
    for i, v in enumerate(y):
        s.learn_one({"x": float(v)})
        if i == SPIKE:
            return bool(s.is_outlier({"x": float(v)}))
    return False


def scorer3d(name, cls, pname, pvals, plabel, sub, ftitle):
    Z = np.array([[np.mean([scorer_flagged(cls, a, p, k, pname) for k in SEEDS]) for a in AMPS] for p in pvals])
    surface(Z, AMPS, pvals, name, ftitle, sub, plabel, "spike amplitude (standard deviations)", "fraction of injected spikes flagged")
    out = HERE / f"{name}_3d.png"
    print("wrote", out, "exists", out.exists())
    print("grid min", round(float(Z.min()), 4), "max", round(float(Z.max()), 4))
    print("Z first row:", [round(float(v), 4) for v in Z[0]])
    print("Z last row:", [round(float(v), 4) for v in Z[-1]])


def scorer_finance(name, cls, pname, pval, thr_label):
    rng = np.random.default_rng(0)
    cost = 12.0 + 1.5 * rng.standard_normal(500)
    cost[SPIKE_F] = 25.0
    s = cls(radius=RADIUS, **{pname: pval})
    flagged = []
    for i, v in enumerate(cost):
        s.learn_one({"x": float(v)})
        if s.is_outlier({"x": float(v)}):
            flagged.append(i)
    fig, ax = plt.subplots(figsize=(12, 6.2))
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.08, right=0.97)
    ax.set_facecolor(BG)
    ax.grid(color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.plot(cost, color=CYAN, lw=1.0)
    ax.scatter(flagged, cost[flagged], color=PINK, s=40, zorder=5)
    ax.set_xlabel("day", color=CYAN)
    ax.set_ylabel("daily cost of executing an order", color=CYAN)
    fig.text(0.05, 0.95, name, fontsize=18, fontweight="bold", color=INK)
    fig.text(0.05, 0.915, "500 seeded days around 12 with noise 1.5, one spike at day 250 to 25; " + thr_label + "; flagged in pink", fontsize=11, color=CYAN)
    out = HERE / f"{name}_finance.png"
    fig.savefig(out, dpi=150, facecolor=BG)
    plt.close(fig)
    print("wrote", out, "exists", out.exists())
    print("flagged days", flagged[:10], "count", len(flagged))


def main():
    sub = "fraction of injected spikes flagged on 20 seeded series; x = spike amplitude (standard deviations), y = n_sigmas"
    Z = grid_fraction(AMPS, SIGMAS)
    surface(Z, AMPS, SIGMAS, "streaming_deviation_detector",
            "streamDD: fraction of spikes flagged", sub,
            "n_sigmas", "spike amplitude (standard deviations)", "fraction of spikes flagged")
    out = HERE / "streaming_deviation_detector_3d.png"
    print("wrote", out, "exists", out.exists())
    print("grid min", round(float(Z.min()), 4), "max", round(float(Z.max()), 4))
    print("Z row n_sigmas=2:", [round(float(v), 4) for v in Z[0]])
    print("Z row n_sigmas=6:", [round(float(v), 4) for v in Z[-1]])
    finance_figure()
    streaming_deviation_scorer_3d()
    streaming_deviation_scorer_finance()
    multi_channel_3d()
    multi_channel_finance()
    scorer3d("hampel_scorer", HampelScorer, "n_sigmas", np.linspace(2.0, 6.0, 8), "n_sigmas", "fraction of injected spikes flagged on 20 seeded series; x = spike amplitude (standard deviations), y = n_sigmas", "HampelScorer: fraction of spikes flagged")
    scorer_finance("hampel_scorer", HampelScorer, "n_sigmas", 3.0, "n_sigmas=3.0")
    scorer3d("tukey_scorer", TukeyScorer, "k", np.linspace(1.0, 3.0, 8), "k (fence multiplier)", "fraction of injected spikes flagged on 20 seeded series; x = spike amplitude (standard deviations), y = k", "TukeyScorer: fraction of spikes flagged")
    scorer_finance("tukey_scorer", TukeyScorer, "k", 1.5, "k=1.5")
    scorer3d("chauvenet_scorer", ChauvenetScorer, "threshold", np.linspace(0.1, 1.0, 8), "threshold", "fraction of injected spikes flagged on 20 seeded series; x = spike amplitude (standard deviations), y = threshold", "ChauvenetScorer: fraction of spikes flagged")
    scorer_finance("chauvenet_scorer", ChauvenetScorer, "threshold", 0.5, "threshold=0.5")
    scorer3d("sigma_clip_scorer", SigmaClipScorer, "n_sigmas", np.linspace(2.0, 6.0, 8), "n_sigmas", "fraction of injected spikes flagged on 20 seeded series; x = spike amplitude (standard deviations), y = n_sigmas", "SigmaClipScorer: fraction of spikes flagged")
    scorer_finance("sigma_clip_scorer", SigmaClipScorer, "n_sigmas", 3.0, "n_sigmas=3.0")


if __name__ == "__main__":
    main()
