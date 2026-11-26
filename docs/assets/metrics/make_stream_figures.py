"""The figures of the "On a stream" pages, drawn with the library itself.

rolling: a forecast of a stock index that is right 80% of the days for 200 days and 50% after that; the surface is the
    running Rolling(Accuracy) for 25 window sizes. The lines are the accuracy of all the days and of windows of 20 and 100.
rolling_auc: a crash score that is higher by delta on the crash days (noise 1, 30% crash days); the surface is the
    running RollingAUC(window=100) for 25 values of delta from 0 to 2.5.
predict_then_learn: a stock direction that repeats the day before with probability 0.9 and the forecast "same as the last
    label I know"; the surface is the running accuracy of progressive_val_score for 25 label delays from 0 to 24 samples.
model_selection: 25 forecasters that say "up" when a noisy signal is above a threshold c, for 25 values of c; the surface is
    the running accuracy of progressive_val_score.

Run from the repo root:
    python docs/assets/metrics/make_stream_figures.py
"""

from math import erf, sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from dense_armor.core.evaluation.evaluate import progressive_val_score
from dense_armor.core.evaluation.metrics import Accuracy, Rolling, RollingAUC

HERE = Path(__file__).resolve().parent
BG, PANEL, GRID = "#0b1118", "#111a24", "#26364a"
INK, MUTED = "#e8fbff", "#8fa3b8"
CYAN, ORANGE, PINK = "#3ec6e8", "#ffa21f", "#ff2d87"
CMAP = LinearSegmentedColormap.from_list("s", [CYAN, "#6a5acd", PINK])


def Phi(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


class LastValue:
    def __init__(self):
        self.last = 0

    def learn_one(self, x, y, t=None):
        self.last = y
        return self

    def predict_one(self, x, t=None):
        return self.last


class Threshold:
    def __init__(self, c):
        self.c = c

    def learn_one(self, x, y, t=None):
        return self

    def predict_one(self, x, t=None):
        return int(x["s"] > self.c)


def surface(Z, x0, ys, name, title, subtitle, yl, zl, zlim):
    T, G = np.meshgrid(np.arange(x0, x0 + Z.shape[1]), ys)
    fig = plt.figure(figsize=(12, 8.2))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(top=0.93, bottom=0.0, left=-0.05, right=1.05)
    ax.set_facecolor(BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.07, 0.10, 0.14, 1.0))
        axis._axinfo["grid"]["color"] = GRID
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1.35, 1.2, 0.7), zoom=1.1)
    ax.set_xlabel("days seen", color=CYAN, labelpad=8)
    ax.set_ylabel(yl, color=CYAN, labelpad=8)
    ax.set_zlabel(zl, color=CYAN, labelpad=6)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.text(0.05, 0.955, title, fontsize=21, fontweight="bold", color=INK)
    fig.text(0.05, 0.915, subtitle, fontsize=11.5, color=CYAN)
    ax.plot_surface(T, G, Z, facecolors=CMAP(np.clip((Z - zlim[0]) / (zlim[1] - zlim[0]), 0, 1)), rstride=1, cstride=4, linewidth=0, antialiased=False, alpha=0.93)
    ax.set_zlim(*zlim)
    fig.savefig(HERE / f"{name}_3d.png", dpi=150, facecolor=BG)
    plt.close(fig)


def lines(curves, name, title, ylabel, ylim, loc="lower right", marks=()):
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.set_xlabel("days seen", color=CYAN)
    ax.set_ylabel(ylabel, color=CYAN)
    ax.tick_params(colors=MUTED)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(color=GRID, lw=0.5)
    ax.set_title(title, color=INK, loc="left", fontsize=13, fontweight="bold")
    for x in marks:
        ax.axvline(x, color=MUTED, lw=1.0, ls="--")
    for (lab, v), col in zip(curves, (CYAN, ORANGE, PINK)):
        ax.plot(np.arange(1, len(v) + 1), v, color=col, lw=2.0, label=lab)
    ax.set_ylim(*ylim)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=INK, fontsize=9, loc=loc)
    fig.tight_layout()
    fig.savefig(HERE / f"{name}_finance.png", dpi=150, facecolor=BG)
    plt.close(fig)


def at(v, days):
    return [round(float(v[d - 1]), 4) for d in days]


def rolling():
    n = 400
    u = np.random.default_rng(51).random(n)
    hit = u < np.where(np.arange(n) < 200, 0.8, 0.5)

    def run(metric):
        out = np.zeros(n)
        for i, h in enumerate(hit):
            metric.update(int(h), 1)
            out[i] = metric.get()
        return out

    ws = np.unique(np.linspace(10, 200, 25).astype(int))
    full = np.array([run(Rolling(Accuracy(), window=int(w))) for w in ws])
    Z = full[:, 19:]
    surface(Z, 20, ws, "rolling", "A WINDOW FOLLOWS A CHANGE, ALL THE DAYS DO NOT",
            "a forecast right 80% of the days for 200 days and 50% after: running Rolling accuracy for 25 window sizes",
            "window (days)", "running accuracy", (0.2, 1.0))
    for w in (10, 100, 200):
        k = int(np.argmin(np.abs(ws - w)))
        print(" window", int(ws[k]), "accuracy at day 200, 250, 400:", at(full[k], [200, 250, 400]))
    allv, r20, r100 = run(Accuracy()), run(Rolling(Accuracy(), window=20)), run(Rolling(Accuracy(), window=100))
    print(" lines at day 150, 250, 400: all", at(allv, [150, 250, 400]), "w20", at(r20, [150, 250, 400]), "w100", at(r100, [150, 250, 400]))
    lines([("all the days", allv), ("window of 100 days", r100), ("window of 20 days", r20)], "rolling",
          "A forecast right 80% of the days, then 50%: accuracy of all the days and of windows", "Accuracy", (0.2, 1.0), marks=(200,))


def rolling_auc():
    n = 400
    rng = np.random.default_rng(52)
    y = (rng.random(n) < 0.3).astype(int)
    e = rng.normal(0, 1, n)

    def run(delta):
        m = RollingAUC(window=100)
        out = np.full(n, np.nan)
        for i in range(n):
            m.update(int(y[i]), float(y[i] * delta + e[i]))
            v = m.get()
            out[i] = np.nan if v is None else v
        return out

    ds = np.linspace(0, 2.5, 25)
    full = np.array([run(d) for d in ds])
    surface(full[:, 99:], 100, ds, "rolling_auc", "THE AUC OF A CRASH SCORE GROWS WITH ITS SEPARATION",
            "a score higher by delta on the crash days (noise 1), window of 100 days: running RollingAUC for 25 values of delta",
            "separation delta", "running AUC", (0.3, 1.0))
    for d in (0.0, 1.0, 2.0):
        k = int(np.argmin(np.abs(ds - d)))
        print(" delta", round(float(ds[k]), 3), "AUC at day 400:", round(float(full[k, -1]), 3), "theory", round(Phi(float(ds[k]) / sqrt(2)), 3))
    cs = [("delta = " + str(d), run(d)) for d in (0.5, 1.0, 2.0)]
    print(" lines at day 150, 250, 400:", {lab: at(v, [150, 250, 400]) for lab, v in cs})
    lines(cs, "rolling_auc", "A crash score on a stock index: running RollingAUC over a window of 100 days", "RollingAUC", (0.3, 1.0))


def stream_dir(n, seed):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[0] = 1
    for i in range(1, n):
        y[i] = y[i - 1] if rng.random() < 0.9 else 1 - y[i - 1]
    return y


def prequential():
    n = 300
    y = stream_dir(n, 53)
    s = [({}, int(v)) for v in y]
    ds = np.arange(0, 25)
    full = np.array([np.array(progressive_val_score(s, LastValue(), Accuracy(), delay=(int(d) or None), every=1)[1]) for d in ds])
    surface(full[:, 29:], 30, ds, "predict_then_learn", "THE LATER THE LABEL, THE WORSE THE FORECAST",
            "a direction that repeats the day before 90% of the time, forecast = last label known: running accuracy for 25 delays",
            "label delay (samples)", "running accuracy", (0.0, 1.0))
    for d in (0, 5, 24):
        print(" delay", d, "accuracy at day 100, 300:", at(full[d], [100, 300]))
    cs = [("delay " + str(d), np.array(progressive_val_score(s[:250], LastValue(), Accuracy(), delay=(d or None), every=1)[1])) for d in (0, 5, 20)]
    print(" lines at day 100, 250:", {lab: at(v, [100, 250]) for lab, v in cs})
    lines(cs, "predict_then_learn", "A direction that repeats: accuracy of the forecast with late labels", "Accuracy", (0.0, 1.0))


def selection():
    n = 300
    rng = np.random.default_rng(54)
    r = rng.normal(0, 1, n)
    s = [({"s": float(r[i] + rng.normal(0, 0.7))}, int(r[i] > 0)) for i in range(n)]
    cs = np.linspace(-1.2, 1.2, 25)
    full = np.array([np.array(progressive_val_score(s, Threshold(float(c)), Accuracy(), every=1)[1]) for c in cs])
    surface(full[:, 19:], 20, cs, "model_selection", "THE BEST THRESHOLD IS NEAR ZERO",
            "25 forecasters that say up when a noisy signal passes a threshold c, same 300 days: running accuracy",
            "threshold c", "running accuracy", (0.3, 1.0))
    best = int(np.argmax(full[:, -1]))
    print(" best c at day 300:", round(float(cs[best]), 2), round(float(full[best, -1]), 3), "| c=-1.2:", round(float(full[0, -1]), 3), "| c=1.2:", round(float(full[-1, -1]), 3))
    cv = [("c = " + str(c), np.array(progressive_val_score(s[:250], Threshold(c), Accuracy(), every=1)[1])) for c in (-1.0, 0.0, 1.0)]
    print(" lines at day 100, 250:", {lab: at(v, [100, 250]) for lab, v in cv})
    lines(cv, "model_selection", "Three forecasters on the same stock index: running accuracy", "Accuracy", (0.3, 1.0))


if __name__ == "__main__":
    rolling()
    rolling_auc()
    prequential()
    selection()
