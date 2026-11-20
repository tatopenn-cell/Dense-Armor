"""The figure that explains stable_var, in four steps, with numbers computed by the library and read from the GPU test.

1. the problem: quotes around 4500 in float32 and the variance of the usual formula;
2. the shift: the error of the squares of the quotes is bigger than the variance, after the shift it is zero;
3. the two-sum: a sum that keeps the error it would lose;
4. the depth: the gradient of a stack of layers, from benchmarks/stable_variance_depth_grid_gpu.json.

Run from the repo root:
    python docs/assets/stable_variance/make_explained.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource, LinearSegmentedColormap
from mpl_toolkits.mplot3d import proj3d
from scipy.interpolate import RectBivariateSpline

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from dense_armor.core.signals.stats.stable import (  # noqa: E402
    _compensated_tree,
    stable_var,
)

BG, GRID, PANEL = "#0b1118", "#26364a", "#0f1822"
INK, MUTED = "#e8fbff", "#8fa3b8"
CYAN, PINK, GOLD = "#3ec6e8", "#ff2d87", "#ffb703"
NL = chr(10)


def pane(ax, title, sub):
    ax.set_facecolor(PANEL)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.06, 0.09, 0.13, 1.0))
        axis._axinfo["grid"]["color"] = GRID
        axis.line.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.figure.text(0.04, 0.93, title, color=INK, fontsize=26, fontweight="bold")
    ax.figure.text(0.04, 0.885, sub, color=CYAN, fontsize=15)


def label(ax, x, y, z, text, color, dx=0, dy=40):
    x2, y2, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
    ax.annotate(
        text,
        (x2, y2),
        xycoords="data",
        xytext=(dx, dy),
        textcoords="offset points",
        color=color,
        fontsize=13,
        fontweight="bold",
        ha="center",
        bbox={"boxstyle": "round,pad=0.35", "fc": BG, "ec": color, "lw": 1.2},
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.5},
    )


def note(ax, text):
    ax.text2D(
        0.0,
        -0.02,
        text,
        transform=ax.transAxes,
        color=INK,
        fontsize=12,
        family="monospace",
        linespacing=1.5,
        va="top",
    )


def quotes():
    rng = np.random.default_rng(0)
    return (np.round(rng.normal(4500.0, 0.5, 3000) * 4.0) / 4.0).astype(np.float32)


def two_sum(a, b):
    s = np.float32(a + b)
    bb = np.float32(s - a)
    return s, np.float32((a - np.float32(s - bb)) + (b - bb))


def panel_problem(ax, x):
    ref = float(np.var(x.astype(np.float64)))
    xj = jnp.asarray(x)
    textbook = float(jnp.mean(xj * xj) - jnp.mean(xj) ** 2)
    stable = float(stable_var(xj))
    n = 26
    idx = np.arange(n)
    dev = x[:n].astype(np.float64) - 4500.0
    cols = [CYAN if v >= 0 else PINK for v in dev]
    ax.bar3d(
        idx, np.zeros(n), np.zeros(n), 0.7, 0.7, dev, color=cols, shade=True, alpha=0.95
    )
    ax.set_zlim(-1.6, 1.6)
    ax.view_init(elev=24, azim=-62)
    ax.set_xlabel("quote number", color=MUTED, fontsize=8, labelpad=-2)
    ax.set_zlabel("price minus 4500", color=MUTED, fontsize=8, labelpad=-2)
    ax.set_yticks([])
    note(
        ax,
        f"3000 quotes of an index near 4500, in steps of 0.25{NL}"
        f"true variance            {ref:.4f}{NL}"
        f"usual formula, float32   {textbook:.4f}   (the market looks still){NL}"
        f"stable_var, float32      {stable:.4f}",
    )
    pane(ax, "The problem", "the usual formula in float32 loses the small wiggle")


def panel_shift(ax, x):
    big = float(np.float32(4500.0) ** 2)
    gap_big = float(np.spacing(np.float32(big)))
    gap_small = float(np.spacing(np.float32(0.25)))
    unit = 0.25
    far = gap_big / unit
    for k in range(4):
        ax.bar3d(k * far - 0.12, 0.0, 0.0, 0.24, 0.5, 3.0, color=PINK, alpha=0.95)
    ax.bar3d(2.5, 0.1, 0.0, 1.0, 0.3, 1.4, color=CYAN, alpha=1.0)
    for k in range(0, int(3 * far / 0.25)):
        ax.bar3d(k * 0.25 - 0.04, 6.0, 0.0, 0.08, 0.5, 3.0, color=GOLD, alpha=0.95)
    ax.bar3d(2.5, 6.1, 0.0, 1.0, 0.3, 1.4, color=CYAN, alpha=1.0)
    ax.set_xlim(-0.5, 3 * far + 0.5)
    ax.set_ylim(-0.5, 7.0)
    ax.set_zlim(0, 4.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=38, azim=-90)
    ax.set_box_aspect((2.6, 1.4, 0.6), zoom=1.0)
    label(
        ax,
        0.0,
        0.2,
        3.0,
        "BEFORE: numbers near 20 million, 2 apart",
        PINK,
        dx=60,
        dy=-70,
    )
    label(
        ax,
        0.0,
        6.2,
        3.0,
        "AFTER the shift: numbers near 0, much closer",
        GOLD,
        dx=60,
        dy=55,
    )
    label(
        ax,
        3.0,
        0.25,
        0.0,
        "the wiggle (0.25) falls between two numbers: LOST",
        CYAN,
        dx=40,
        dy=-75,
    )
    label(ax, 3.0, 6.25, 0.0, "the wiggle sits on a number: KEPT", CYAN, dx=330, dy=-95)
    note(
        ax,
        f"every number float32 can hold is a post; near 20 million the posts are {gap_big:.0f} apart{NL}"
        f"after the shift (x - x[0], exact) the posts are {gap_small:.0e} apart; drawn far less dense than real{NL}"
        f"the wiggle is 0.25: it falls between far posts and is lost, it sits on close ones and is kept",
    )
    pane(ax, "1  Shift", "take away the big part first, and the small part fits")


def panel_twosum(ax):
    a, b = np.float32(16777216.0), np.float32(1.0)
    plain = np.float32(a + b)
    s, e = two_sum(a, b)
    ax.bar3d(0.0, 0.0, 0.0, 0.9, 0.9, 10.0, color=CYAN, alpha=0.95)
    ax.bar3d(0.25, 0.25, 10.0, 0.4, 0.4, 0.9, color=GOLD, alpha=1.0)
    ax.bar3d(2.4, 0.0, 0.0, 0.9, 0.9, 10.0, color=CYAN, alpha=0.95)
    ax.bar3d(4.0, 0.2, 0.0, 0.4, 0.4, 0.9, color=PINK, alpha=0.45)
    ax.bar3d(4.8, 0.0, 0.0, 0.9, 0.9, 10.0, color=CYAN, alpha=0.95)
    ax.bar3d(6.4, 0.2, 0.0, 0.4, 0.4, 0.9, color=GOLD, alpha=1.0)
    ax.set_xlim(-0.2, 7.2)
    ax.set_ylim(-0.2, 1.4)
    ax.set_zlim(0, 12.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=14, azim=-72)
    ax.set_box_aspect((2.6, 0.6, 1.0), zoom=1.0)
    label(ax, 0.45, 0.45, 10.9, "EXACT: a + b", INK, dy=50)
    label(ax, 2.85, 0.45, 10.0, "PLAIN SUM: a only", INK, dy=60)
    label(ax, 4.2, 0.4, 0.9, "the 1 fell off: LOST", PINK, dx=0, dy=-60)
    label(ax, 5.25, 0.45, 10.0, "TWO-SUM: s", INK, dy=60)
    label(ax, 6.6, 0.4, 0.9, "the 1 is caught: e", GOLD, dx=0, dy=-60)
    note(
        ax,
        f"a = {int(a)} and b = {int(b)}, in float32{NL}"
        f"plain sum: {int(plain)}   the {int(b)} is gone{NL}"
        f"TwoSum: s = {int(s)} and e = {int(e)}   a + b = s + e exactly{NL}"
        f"x = fl(a + b),  z = fl(x - a),  y = fl((a - (x - z)) + (b - z))   (Knuth; Ogita, Rump, Oishi, Algorithm 3.1)",
    )
    pane(ax, "3  TwoSum", "a sum and its exact error: nothing is lost")


def panel_centre(ax, x):
    n = 20
    idx = np.arange(n)
    y = x[:n] - x[0]
    m = float(np.mean(y))
    d = y - np.float32(m)
    ax.bar3d(
        idx, np.zeros(n), np.zeros(n), 0.7, 0.5, y, color=CYAN, alpha=0.95, shade=True
    )
    cols = [CYAN if v >= 0 else PINK for v in d]
    ax.bar3d(
        idx,
        np.full(n, 4.0),
        np.zeros(n),
        0.7,
        0.5,
        d,
        color=cols,
        alpha=0.95,
        shade=True,
    )
    gx, gy = np.meshgrid([-0.5, n], [-0.3, 0.9])
    ax.plot_surface(
        gx, gy, np.full_like(gx, m, dtype=float), color=GOLD, alpha=0.4, linewidth=0
    )
    ax.set_xlim(-0.5, n)
    ax.set_ylim(-0.3, 5.0)
    ax.set_zlim(float(d.min()) * 1.3, float(y.max()) * 1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.view_init(elev=36, azim=-90)
    ax.set_box_aspect((2.6, 1.4, 0.8), zoom=1.0)
    label(
        ax,
        2.0,
        0.2,
        float(y[2]),
        f"y = x - x[0]: small numbers, but their mean m = {m:.4f} is not 0",
        CYAN,
        dx=80,
        dy=-70,
    )
    label(ax, 2.0, 4.2, float(d[2]), "d = y - m: centred on 0", GOLD, dx=80, dy=-70)
    note(
        ax,
        f"the first 20 of the 3000 quotes, after the shift y = x - x[0]{NL}"
        f"the mean m of the shifted values is at the level of the wiggle, and so is its rounding error,{NL}"
        f"not at the level of the offset; d = y - m is formed before the squares{NL}"
        f"the two-term formula  s2/n - (s1/n)**2  cancels the linear part of the error of m",
    )
    pane(
        ax,
        "2  Centre",
        "subtract the mean of the small part, so the squares stay small",
    )


def panel_tree(ax):
    v = jnp.asarray([1.0] + [1e-8] * 7, dtype=jnp.float32)
    plain = float(jnp.sum(v))
    tree = float(_compensated_tree(v))
    exact = float(np.sum(np.asarray(v, dtype=np.float64)))
    xs = np.arange(8, dtype=float)
    for lvl, count in enumerate((8, 4, 2, 1)):
        step = 8 / count
        for k in range(count):
            cx = k * step + (step - 1) / 2
            z = lvl * 3.0
            ax.bar3d(cx, 0.0, z, 0.6, 0.6, 1.2, color=CYAN, alpha=0.95)
            ax.bar3d(cx + 0.7, 0.1, z, 0.3, 0.3, 0.45, color=GOLD, alpha=1.0)
            if lvl:
                for ch in (2 * k, 2 * k + 1):
                    px = ch * (step / 2) + (step / 2 - 1) / 2 + 0.3
                    ax.plot(
                        [px, cx + 0.3],
                        [0.3, 0.3],
                        [z - 3.0 + 1.2, z],
                        color=MUTED,
                        lw=1.2,
                    )
    ax.set_xlim(-0.5, 9)
    ax.set_ylim(-0.5, 1.5)
    ax.set_zlim(0, 11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=16, azim=-78)
    ax.set_box_aspect((2.4, 0.5, 1.2), zoom=1.0)
    label(ax, 3.5, 0.3, 0.0, "8 numbers: 1.0 and seven of 1e-8", INK, dy=-70)
    label(
        ax,
        3.5,
        0.3,
        3.0,
        "pairs added: TwoSum on the sums, on the carries, on their sum",
        INK,
        dx=-250,
        dy=70,
    )
    label(ax, 3.5, 0.3, 9.0, "S + C, the sum and the carry", INK, dx=170, dy=10)
    note(
        ax,
        f"blue: the partial sums S    gold: the carries C, errors carried up (drawn bigger than real){NL}"
        f"each merge: (s1, e1) = TwoSum of the sums, (s2, e2) = TwoSum of the carries, (s3, e3) = TwoSum(s1, s2){NL}"
        f"new sum S = s3, new carry C = e1 + e2 + e3; after log2(n) levels the result is S + C{NL}"
        f"plain float32 sum {plain:.8f}   compensated tree {tree:.8f}   exact {exact:.8f}",
    )
    pane(
        ax,
        "4  Compensated tree",
        "the author's sum: pairs in a tree, the errors carried up",
    )


def panel_depth(ax):
    d = json.loads(
        (REPO / "benchmarks" / "stable_variance_depth_grid_gpu.json").read_text(
            encoding="utf-8"
        )
    )
    rows = d["rows"]
    depths = sorted({r["depth"] for r in rows})
    spreads = sorted({r["spread"] for r in rows})
    Zu = np.zeros((len(spreads), len(depths)))
    Zs = np.zeros_like(Zu)
    for r in rows:
        i, j = spreads.index(r["spread"]), depths.index(r["depth"])
        Zu[i, j] = 100.0 * r["usual"]["grad_rel_err"]
        Zs[i, j] = 100.0 * r["stable"]["grad_rel_err"]
    xs = np.log2(depths)
    ys = np.log10([d["offset"] / s for s in spreads])
    order = np.argsort(ys)
    ys, Zu, Zs = ys[order], Zu[order], Zs[order]
    fx, fy = np.linspace(xs.min(), xs.max(), 80), np.linspace(ys.min(), ys.max(), 50)
    X, Y = np.meshgrid(fx, fy)
    SU = RectBivariateSpline(ys, xs, Zu, kx=3, ky=3)(fy, fx)
    SS = RectBivariateSpline(ys, xs, Zs, kx=3, ky=3)(fy, fx)
    cn = LinearSegmentedColormap.from_list("n", ["#7b2ff7", "#f72585", "#ffb703"])
    cs = LinearSegmentedColormap.from_list("s", ["#35d0ff", "#35d0ff"])
    ls = LightSource(azdeg=300, altdeg=40)
    for S, cm in ((SS, cs), (SU, cn)):
        shade = ls.shade(
            S, cmap=cm, vert_exag=0.4, blend_mode="soft", vmin=S.min(), vmax=S.max()
        )
        if S is SU:
            shade[..., 3] = np.clip(S / (0.35 * S.max()), 0.0, 1.0) ** 0.7
        ax.plot_surface(
            X,
            Y,
            S,
            facecolors=shade,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=True,
            shade=False,
        )
    worst = max(rows, key=lambda r: r["usual"]["grad_rel_err"])
    top = min(spreads)
    over = min(
        (r for r in rows if r["spread"] == top and r["usual"]["grad_rel_err"] > 0.01),
        key=lambda r: r["depth"],
    )
    bx, by = np.log2(over["depth"]), np.log10(d["offset"] / top)
    z1 = 100 * over["usual"]["grad_rel_err"]
    ax.plot([bx, bx], [by, by], [0, z1], color="white", lw=2.2)
    ax.scatter([bx], [by], [z1], color="white", s=30)
    ax.set_zlim(0, float(Zu.max()) * 1.05)
    ax.view_init(elev=24, azim=-128)
    ax.set_xlabel("layers, log2", color=MUTED, fontsize=8, labelpad=-2)
    ax.set_ylabel("offset over spread, log10", color=MUTED, fontsize=8, labelpad=-2)
    ax.set_zlabel("gradient wrong by, %", color=MUTED, fontsize=8, labelpad=-2)
    note(
        ax,
        f"GPU, float32, {len(rows)} stacks of normalised layers{NL}"
        f"pink: usual layer, wrong by up to {100 * worst['usual']['grad_rel_err']:.0f} % at {worst['depth']} layers{NL}"
        f"blue: stable_var, below {100 * max(r['stable']['grad_rel_err'] for r in rows):.3f} % in every stack{NL}"
        f"white line: the usual layer passes 1 % at {over['depth']} layers (offset/spread {d['offset'] / top:.0f})",
    )
    pane(
        ax,
        "The depth",
        "the usual layer breaks as layers pile up; stable_var does not",
    )


def main():
    x = quotes()
    jobs = [
        ("stable_var_1_shift.png", lambda ax: panel_shift(ax, x)),
        ("stable_var_2_centre.png", lambda ax: panel_centre(ax, x)),
        ("stable_var_3_twosum.png", panel_twosum),
        ("stable_var_4_tree.png", panel_tree),
        ("stable_var_1_problem.png", lambda ax: panel_problem(ax, x)),
        ("stable_var_4_depth.png", panel_depth),
    ]
    for name, draw in jobs:
        fig = plt.figure(figsize=(13, 12))
        fig.patch.set_facecolor(BG)
        ax = fig.add_axes((0.0, 0.2, 1.0, 0.66), projection="3d")
        draw(ax)
        fig.savefig(HERE / name, dpi=130, facecolor=BG)
        plt.close(fig)
        print("written", name)


if __name__ == "__main__":
    main()
