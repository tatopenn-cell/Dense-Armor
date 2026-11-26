"""The figures of the Events pages, drawn with the library itself.

One stream of 3600 one-second samples of a stock tick: N(0, 1) noise, and four labelled events of 60 seconds (flash
crashes) at 600, 1400, 2200 and 3000 s in which the mean shifts by delta. The same noise for every delta.
The detector of the first three pages says "alarm" on every sample above a threshold c; the fourth page uses the CUSUM
detector of the library with the decision threshold h.
delays_3d: mean detection delay over (c, delta). range_3d: range F1. nab_3d: NAB score. evaluate_events_3d: NAB score of the CUSUM
detector over (h, delta). point_adjusted_3d: point-adjusted F1 and plain F1 of a random flag over (rate, event length).

Run from the repo root:
    python docs/assets/metrics/make_events_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from dense_armor.core.evaluation.evaluate import evaluate_events
from dense_armor.core.evaluation.metrics import EventMetrics, F1, PointAdjustedF1
from dense_armor.core.online.drift.detector import CUSUMDriftDetector

HERE = Path(__file__).resolve().parent
BG, PANEL, GRID = "#0b1118", "#111a24", "#26364a"
INK, MUTED = "#e8fbff", "#8fa3b8"
CYAN, ORANGE, PINK = "#3ec6e8", "#ffa21f", "#ff2d87"
CMAP = LinearSegmentedColormap.from_list("s", [CYAN, "#6a5acd", PINK])
N = 3600
STARTS = (600, 1400, 2200, 3000)
WINDOWS = [(float(s), float(s + 59)) for s in STARTS]
NORMAL_S = N - 4 * 60.0
E = np.random.default_rng(61).normal(0, 1, N)
MASK = np.zeros(N)
for s in STARTS:
    MASK[s : s + 60] = 1.0
CS = np.linspace(1.0, 4.0, 25)
DS = np.linspace(0.5, 4.0, 25)
HS = np.linspace(4.0, 30.0, 25)


def stream(delta):
    return E + delta * MASK


def surface(Z, xs, ys, name, title, subtitle, xl, yl, zl, zlim, second=None):
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
    if second is None:
        ax.plot_surface(X, Y, Z, facecolors=CMAP(np.clip((np.nan_to_num(Z) - zlim[0]) / (zlim[1] - zlim[0]), 0, 1)), rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=0.93)
    else:
        ax.plot_surface(X, Y, Z, color=CYAN, rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=0.8)
        ax.plot_surface(X, Y, second, color=PINK, rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=0.8)
    ax.set_zlim(*zlim)
    fig.savefig(HERE / f"{name}_3d.png", dpi=150, facecolor=BG)
    plt.close(fig)


def axes2d(title, xl, yl, w=9, h=3.6):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.set_xlabel(xl, color=CYAN)
    ax.set_ylabel(yl, color=CYAN)
    ax.tick_params(colors=MUTED)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(color=GRID, lw=0.5)
    ax.set_title(title, color=INK, loc="left", fontsize=13, fontweight="bold")
    return fig, ax


def save2d(fig, ax, name, loc="best"):
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=INK, fontsize=9, loc=loc)
    fig.tight_layout()
    fig.savefig(HERE / f"{name}_finance.png", dpi=150, facecolor=BG)
    plt.close(fig)


def alarms_over(x, c):
    return [float(i) for i in np.flatnonzero(x > c)]


def delays():
    Z = np.full((len(DS), len(CS)), np.nan)
    for i, d in enumerate(DS):
        x = stream(d)
        for j, c in enumerate(CS):
            m = EventMetrics(WINDOWS, dt=1.0, normal_time_s=NORMAL_S)
            for t in alarms_over(x, c):
                m.add_alarm(t)
            got = [v for v in m.detection_delays_s() if v is not None]
            if got:
                Z[i, j] = float(np.mean(got))
    surface(Z, CS, DS, "events_delays", "A HIGHER THRESHOLD CATCHES LATER, OR NOT AT ALL",
            "flash crashes of 60 s on a stock tick, same 3600 s: mean detection delay for 25 thresholds and 25 shifts",
            "alarm threshold c", "shift of the event", "mean delay (s)", (0, 40))
    for d, c in ((3.0, 2.0), (3.0, 3.0), (1.5, 3.0), (4.0, 1.0)):
        i, j = int(np.argmin(np.abs(DS - d))), int(np.argmin(np.abs(CS - c)))
        m = EventMetrics(WINDOWS, dt=1.0, normal_time_s=NORMAL_S)
        for t in alarms_over(stream(DS[i]), CS[j]):
            m.add_alarm(t)
        print(" delays: shift", round(float(DS[i]), 2), "c", round(float(CS[j]), 2), "-> delays", [None if v is None else round(v, 1) for v in m.detection_delays_s()],
              "missed", m.missed_events(), "false", m.false_alarms(), "per hour", round(m.false_alarms_per_hour(), 2))
    x = stream(3.0)
    c = 2.5
    al = alarms_over(x, c)
    lo, hi = 560, 740
    fig, ax = axes2d("A stock tick with a flash crash of 60 s: the alarms of the threshold 2.5", "seconds", "tick")
    ax.plot(np.arange(lo, hi), x[lo:hi], color=CYAN, lw=1.4, label="tick")
    ax.axvspan(600, 659, color=ORANGE, alpha=0.18, label="the labelled event")
    ax.axhline(c, color=MUTED, lw=1.0, ls="--")
    first = True
    for t in al:
        if lo <= t < hi:
            ax.axvline(t, color=PINK, lw=1.2, alpha=0.8, label="alarm" if first else None)
            first = False
    save2d(fig, ax, "events_delays", "upper left")
    mm = EventMetrics(WINDOWS, dt=1.0, normal_time_s=NORMAL_S)
    for t in al:
        mm.add_alarm(t)
    print(" delays figure: shift 3, c 2.5: delays", [None if v is None else round(v, 1) for v in mm.detection_delays_s()], "missed", mm.missed_events(),
          "false", mm.false_alarms(), "per hour", round(mm.false_alarms_per_hour(), 2), "| alarms in the figure:", sum(1 for t in al if lo <= t < hi))


def range_():
    Z = np.zeros((len(DS), len(CS)))
    for i, d in enumerate(DS):
        x = stream(d)
        for j, c in enumerate(CS):
            m = EventMetrics(WINDOWS, dt=1.0, normal_time_s=NORMAL_S, threshold_window=1.0)
            for t in alarms_over(x, c):
                m.add_alarm(t)
            Z[i, j] = m.range_f1()
    surface(Z, CS, DS, "events_range", "THE RANGE F1 NEEDS BOTH: CATCH THE EVENT AND STAY QUIET",
            "same 3600 s: range F1 (front bias) of the alarms of 25 thresholds and 25 shifts",
            "alarm threshold c", "shift of the event", "range F1", (0, 1))
    for d, c in ((3.0, 2.0), (3.0, 3.0), (1.5, 2.0), (4.0, 1.0), (4.0, 4.0)):
        i, j = int(np.argmin(np.abs(DS - d))), int(np.argmin(np.abs(CS - c)))
        print(" range: shift", round(float(DS[i]), 2), "c", round(float(CS[j]), 2), "-> F1", round(float(Z[i, j]), 3))
    fig, ax = axes2d("One event of 60 s, caught from the second d to its end: the recall of the four biases", "second d of the first alarm", "range recall")
    ds = np.arange(0, 60, 2)
    out = {}
    for b, col in (("front", CYAN), ("flat", ORANGE), ("middle", PINK), ("back", MUTED)):
        v = []
        for d in ds:
            m = EventMetrics([(0.0, 59.0)], positional_bias=b, threshold_window=1.0)
            for t in range(int(d), 60):
                m.add_alarm(float(t))
            v.append(m.range_recall())
        out[b] = [round(v[k], 3) for k in (0, 5, 15, 29)]
        ax.plot(ds, v, color=col, lw=2.0, label=b)
    ax.set_ylim(0, 1.05)
    save2d(fig, ax, "events_range", "upper right")
    print(" range figure: recall at d=0, 10, 30, 58 for each bias:", out)


def nab():
    Z = np.zeros((len(DS), len(CS)))
    for i, d in enumerate(DS):
        x = stream(d)
        for j, c in enumerate(CS):
            m = EventMetrics(WINDOWS, dt=1.0, normal_time_s=NORMAL_S)
            for t in alarms_over(x, c):
                m.add_alarm(t)
            Z[i, j] = m.nab_score()
    surface(Z, CS, DS, "events_nab", "THE NAB SCORE REWARDS EARLY AND PUNISHES FALSE ALARMS",
            "same 3600 s: NAB score of the alarms of 25 thresholds and 25 shifts",
            "alarm threshold c", "shift of the event", "NAB score", (-200, 100))
    for d, c in ((3.0, 2.0), (3.0, 3.0), (1.5, 2.0), (4.0, 1.0), (4.0, 4.0)):
        i, j = int(np.argmin(np.abs(DS - d))), int(np.argmin(np.abs(CS - c)))
        print(" nab: shift", round(float(DS[i]), 2), "c", round(float(CS[j]), 2), "-> NAB", round(float(Z[i, j]), 2))
    fig, ax = axes2d("One event of 60 s: the NAB score of a single alarm at the second t", "second t of the alarm (the event is from 0 to 60)", "NAB score")
    ts = np.arange(0, 121, 1.0)
    v = []
    for t in ts:
        m = EventMetrics([(0.0, 60.0)])
        m.add_alarm(float(t))
        v.append(m.nab_score())
    ax.plot(ts, v, color=CYAN, lw=2.0, label="one alarm at t")
    ax.axvspan(0, 60, color=ORANGE, alpha=0.18, label="the event")
    save2d(fig, ax, "events_nab", "upper right")
    print(" nab figure: score at t=0, 15, 30, 45, 59, 70, 120:", [round(v[int(k)], 2) for k in (0, 15, 30, 45, 59, 70, 120)])


def evaluate():
    names = ("nab_score", "mean_delay_s", "missed", "false_alarms")
    Z = np.zeros((len(DS), len(HS)))
    D = np.full((len(DS), len(HS)), np.nan)
    for i, d in enumerate(DS):
        s = [(float(t), float(v)) for t, v in enumerate(stream(d))]
        for j, h in enumerate(HS):
            det = CUSUMDriftDetector(reference="fixed", radius=10, ref_mult=3, k=0.5, h=float(h), reset_on_alarm=True)
            r = evaluate_events(s, det, WINDOWS, dt=1.0, normal_time_s=NORMAL_S)
            Z[i, j] = r["nab_score"]
            D[i, j] = np.nan if r.get("mean_delay_s") is None else r["mean_delay_s"]
    surface(Z, HS, DS, "events_evaluate", "THE CUSUM THRESHOLD h AGAINST THE SHIFT",
            "same 3600 s through evaluate_events: NAB score of the CUSUM detector for 25 thresholds h and 25 shifts",
            "CUSUM threshold h", "shift of the event", "NAB score", (-200, 100))
    for d, h in ((3.0, 10.0), (3.0, 20.0), (1.5, 10.0), (4.0, 5.0), (4.0, 30.0)):
        i, j = int(np.argmin(np.abs(DS - d))), int(np.argmin(np.abs(HS - h)))
        print(" evaluate: shift", round(float(DS[i]), 2), "h", round(float(HS[j]), 2), "-> NAB", round(float(Z[i, j]), 2), "mean delay", None if np.isnan(D[i, j]) else round(float(D[i, j]), 2))
    fig, ax = axes2d("CUSUM on a stock tick: the mean delay of the flash crashes against their shift", "shift of the event", "mean delay (s)")
    for h, col in ((8.0, CYAN), (15.0, ORANGE), (25.0, PINK)):
        v = []
        for d in np.linspace(1.0, 4.0, 13):
            s = [(float(t), float(x)) for t, x in enumerate(stream(d))]
            det = CUSUMDriftDetector(reference="fixed", radius=10, ref_mult=3, k=0.5, h=h, reset_on_alarm=True)
            r = evaluate_events(s, det, WINDOWS, dt=1.0, normal_time_s=NORMAL_S)
            v.append(np.nan if r.get("mean_delay_s") is None else r["mean_delay_s"])
        ax.plot(np.linspace(1.0, 4.0, 13), v, color=col, lw=2.0, marker="o", ms=3, label="h = " + str(int(h)))
        print(" evaluate figure h", h, "mean delay at shift 1, 2, 3, 4:", [None if np.isnan(v[k]) else round(float(v[k]), 2) for k in (0, 4, 8, 12)])
    save2d(fig, ax, "events_evaluate", "upper right")
    s = [(float(t), float(x)) for t, x in enumerate(stream(3.0))]
    det = CUSUMDriftDetector(reference="fixed", radius=10, ref_mult=3, k=0.5, h=15.0, reset_on_alarm=True)
    r = evaluate_events(s, det, WINDOWS, dt=1.0, normal_time_s=NORMAL_S)
    print(" report keys:", sorted(r.keys()))


def point_adjusted():
    ps = np.linspace(0.0, 0.2, 25)
    Ls = np.linspace(10, 200, 25).astype(int)
    n = 2000
    u = np.random.default_rng(62).random(n)

    def run(p, L):
        y = np.zeros(n, dtype=int)
        for s in (200, 700, 1200, 1700):
            y[s : s + L] = 1
        flag = (u < p).astype(int)
        pa, f1 = PointAdjustedF1(positive=1, warn=False), F1(positive=1)
        for a, b in zip(y.tolist(), flag.tolist()):
            pa.update(a, b)
            f1.update(a, b)
        return pa.get(), f1.get()

    PA = np.zeros((len(Ls), len(ps)))
    PL = np.zeros((len(Ls), len(ps)))
    for i, L in enumerate(Ls):
        for j, p in enumerate(ps):
            PA[i, j], PL[i, j] = run(p, int(L))
    surface(PA, ps, Ls, "events_point_adjusted", "A RANDOM FLAG LOOKS GOOD AFTER POINT ADJUSTMENT",
            "random flags of 25 rates, events of 25 lengths: point-adjusted F1 (cyan) and plain F1 (pink)",
            "share of samples flagged at random", "event length (samples)", "F1", (0, 1), second=PL)
    for p, L in ((0.05, 100), (0.05, 10), (0.1, 100), (0.2, 200)):
        i, j = int(np.argmin(np.abs(Ls - L))), int(np.argmin(np.abs(ps - p)))
        print(" point-adjusted: rate", round(float(ps[j]), 3), "length", int(Ls[i]), "-> PA", round(float(PA[i, j]), 3), "F1", round(float(PL[i, j]), 3))
    fig, ax = axes2d("A random flag: point-adjusted F1 and plain F1 against the share of samples flagged", "share of samples flagged at random", "F1")
    pp = np.linspace(0.0, 0.2, 21)
    out = {}
    for L, col, ls in ((100, CYAN, "-"), (20, ORANGE, "-")):
        a = [run(p, L)[0] for p in pp]
        ax.plot(pp, a, color=col, lw=2.0, label="point-adjusted F1, events of " + str(L))
        out[L] = [round(a[k], 3) for k in (5, 10, 20)]
    f = [run(p, 100)[1] for p in pp]
    ax.plot(pp, f, color=PINK, lw=2.0, label="plain F1, events of 100")
    out["f1"] = [round(f[k], 3) for k in (5, 10, 20)]
    ax.set_ylim(0, 1.05)
    save2d(fig, ax, "events_point_adjusted", "center right")
    print(" point-adjusted figure at rate 0.05, 0.1, 0.2:", out)


if __name__ == "__main__":
    delays()
    range_()
    nab()
    evaluate()
    point_adjusted()
