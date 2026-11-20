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
from scipy.interpolate import RectBivariateSpline

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from dense_armor.core.signals.stats.stable import stable_var  # noqa: E402

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
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.set_title(title, color=INK, fontsize=15, fontweight="bold", loc="left", pad=2)
    ax.text2D(0.0, 0.93, sub, transform=ax.transAxes, color=CYAN, fontsize=10.5)


def note(ax, text):
    ax.text2D(
        0.0,
        -0.02,
        text,
        transform=ax.transAxes,
        color=INK,
        fontsize=9.5,
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
    pane(ax, "1  The problem", "the usual formula in float32 loses the small wiggle")


def panel_shift(ax, x):
    n = 24
    idx = np.arange(n)
    v = x[:n].astype(np.float64)
    err_before = np.abs((x[:n] * x[:n]).astype(np.float64) - v * v)
    d = x[:n] - x[0]
    err_after = np.abs((d * d).astype(np.float64) - (v - v[0]) ** 2)
    var = float(np.var(x.astype(np.float64)))
    ax.bar3d(
        idx,
        np.zeros(n),
        np.zeros(n),
        0.7,
        0.5,
        err_before,
        color=PINK,
        alpha=0.95,
        shade=True,
    )
    ax.bar3d(
        idx,
        np.full(n, 1.2),
        np.zeros(n),
        0.7,
        0.5,
        err_after + 0.004,
        color=GOLD,
        alpha=0.95,
        shade=True,
    )
    gx, gy = np.meshgrid([0, n], [-0.3, 1.9])
    ax.plot_surface(
        gx, gy, np.full_like(gx, var, dtype=float), color=CYAN, alpha=0.35, linewidth=0
    )
    ax.set_zlim(0, max(float(err_before.max()), var) * 1.1)
    ax.view_init(elev=22, azim=-62)
    ax.set_yticks([])
    ax.set_xlabel("quote number", color=MUTED, fontsize=8, labelpad=-2)
    ax.set_zlabel("error of the square", color=MUTED, fontsize=8, labelpad=-2)
    note(
        ax,
        f"blue plane: the variance to measure, {var:.2f}{NL}"
        f"pink: error of the squares of the prices in float32, up to {err_before.max():.1f}{NL}"
        f"gold: the same after the shift x - x[0], error {err_after.max():.1f} (exact){NL}"
        f"the error was bigger than the wiggle; the shift removes it",
    )
    pane(ax, "2  The shift", "take away the big part first: the subtraction is exact")


def panel_twosum(ax):
    a, b = np.float32(16777216.0), np.float32(1.0)
    plain = np.float32(a + b)
    s, e = two_sum(a, b)
    for x0, top, col in ((0.0, 1.0, GOLD), (1.4, 0.02, PINK), (2.8, 1.0, GOLD)):
        ax.bar3d([x0], [0], [0], 0.6, 0.6, [10.0], color=CYAN, alpha=0.95)
        ax.bar3d([x0], [0], [10.0], 0.6, 0.6, [top], color=col, alpha=0.95)
    ax.set_zlim(0, 13)
    ax.set_xticks([0.3, 1.7, 3.1])
    ax.set_xticklabels(["exact", "plain sum", "two-sum"], color=MUTED, fontsize=8)
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=22, azim=-60)
    note(
        ax,
        f"a = {int(a)},  b = {int(b)}  (blue: a, gold: b or its error){NL}"
        f"plain float32 sum  {int(plain)}   the {int(b)} is lost (pink, empty){NL}"
        f"two-sum  s = {int(s)}  and  e = {int(e)}   nothing is lost{NL}"
        f"stable_var adds the pieces e back",
    )
    pane(ax, "3  The two-sum", "a sum that also keeps the error it would lose")


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
    best = max(
        rows, key=lambda r: r["usual"]["grad_rel_err"] / r["stable"]["grad_rel_err"]
    )
    k = best["usual"]["grad_rel_err"] / best["stable"]["grad_rel_err"]
    bx, by = np.log2(best["depth"]), np.log10(d["offset"] / best["spread"])
    z0, z1 = 100 * best["stable"]["grad_rel_err"], 100 * best["usual"]["grad_rel_err"]
    ax.plot([bx, bx], [by, by], [z0, z1], color="white", lw=2.2)
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
        f"white line: {best['depth']} layers, {k:,.0f} times closer to the truth",
    )
    pane(
        ax,
        "4  The depth",
        "the usual layer breaks as layers pile up; stable_var does not",
    )


def main():
    x = quotes()
    fig = plt.figure(figsize=(17, 12.5))
    fig.patch.set_facecolor(BG)
    fig.text(
        0.03,
        0.965,
        "HOW stable_var WORKS, AND WHERE IT MATTERS",
        fontsize=24,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.03,
        0.935,
        "a variance, and the gradient of a deep network, when the numbers are big and the wiggle is small, in float32",
        fontsize=12.5,
        color=CYAN,
    )
    boxes = [
        (0.02, 0.57, 0.46, 0.3),
        (0.52, 0.57, 0.46, 0.3),
        (0.02, 0.1, 0.46, 0.27),
        (0.52, 0.1, 0.46, 0.27),
    ]
    axes = [fig.add_axes(b, projection="3d") for b in boxes]
    panel_problem(axes[0], x)
    panel_shift(axes[1], x)
    panel_twosum(axes[2])
    panel_depth(axes[3])
    fig.savefig(HERE / "stable_var_explained.png", dpi=130, facecolor=BG)
    print("written stable_var_explained.png")


if __name__ == "__main__":
    main()
