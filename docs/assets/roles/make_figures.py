import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from dense_armor.core.online.anomaly.hst import HalfSpaceTrees
from dense_armor.core.online.drift.page_hinkley import PageHinkley
from dense_armor.core.online.learn.online_classifiers import OnlineGaussianNB
from dense_armor.core.online.tree.hoeffding_regressor import HoeffdingTreeRegressor
from dense_armor.core.signals.preprocessing.scale import StandardScaler
from dense_armor.roles import Classifier

BG = "#0d131a"
CMAP = LinearSegmentedColormap.from_list(
    "roles", ["#00b4d8", "#4cc9f0", "#7209b7", "#f72585", "#ff9e00"]
)
OUT = Path(__file__).resolve().parent


def stream(n, seed):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        k = int(rng.integers(2))
        c = 1.4 if k else -1.4
        a, b = rng.normal(c, 0.7, 2)
        yield {"a": float(a), "b": float(b)}, k


def train():
    nb = OnlineGaussianNB()
    rt = HoeffdingTreeRegressor(grace_period=40)
    hst = HalfSpaceTrees(seed=0)
    for x, k in stream(9000, 0):
        nb.learn_one(x, k)
        rt.learn_one(x, float(np.sin(1.3 * x["a"])) + 0.3 * x["b"])
        hst.learn_one(x)
    return nb, rt, hst


def surface(fn, g):
    z = np.zeros((len(g), len(g)))
    for i, b in enumerate(g):
        for j, a in enumerate(g):
            z[i, j] = fn({"a": float(a), "b": float(b)})
    return z


def panel(fig, rect, g, z, zlabel, pts, zpts, xl="a", yl="b"):
    ax = fig.add_axes(rect, projection="3d")
    ax.set_facecolor(BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#1e293b")
        axis._axinfo["grid"]["color"] = (0.20, 0.30, 0.42, 0.45)
        axis._axinfo["grid"]["linewidth"] = 0.7
    a, b = np.meshgrid(g, g)
    ax.plot_surface(
        a,
        b,
        z,
        cmap=CMAP,
        alpha=0.78,
        edgecolor="#48cae4",
        linewidth=0.25,
        rstride=1,
        cstride=1,
    )
    if pts is not None:
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            zpts,
            color="#ffffff",
            s=9,
            alpha=0.9,
            edgecolors="#90e0ef",
            linewidth=0.3,
        )
    ax.set_xlabel(xl, color="#90e0ef", fontsize=8, fontweight="bold")
    ax.set_ylabel(yl, color="#90e0ef", fontsize=8, fontweight="bold")
    ax.set_zlabel(zlabel, color="#90e0ef", fontsize=8, fontweight="bold")
    ax.tick_params(colors="#64748b", labelsize=7)
    ax.view_init(elev=24, azim=-58)
    return ax


def overview():
    nb, rt, hst = train()
    g = np.linspace(-3.5, 3.5, 36)
    zc = surface(lambda x: nb.predict_proba_one(x).get(1, 0.0), g)
    zr = surface(lambda x: rt.predict_one(x), g)
    za = surface(lambda x: hst.score_one(x), g)
    pts = np.array([[x["a"], x["b"]] for x, _ in stream(120, 7)])
    fig = plt.figure(figsize=(13.5, 6.0), facecolor=BG)
    fig.text(
        0.5,
        0.94,
        "ONE LOOP, THREE ROLES",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color="#e0fbfc",
    )
    fig.text(
        0.5,
        0.88,
        "The same dict goes in; each role answers with a different number",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#4cc9f0",
    )
    cols = [
        (
            "01",
            "CLASSIFIER",
            "predict_proba_one(x)[1]",
            0.02,
            zc,
            "P(class 1)",
            lambda p: nb.predict_proba_one({"a": float(p[0]), "b": float(p[1])}).get(
                1, 0.0
            ),
        ),
        (
            "02",
            "REGRESSOR",
            "predict_one(x)",
            0.35,
            zr,
            "prediction",
            lambda p: rt.predict_one({"a": float(p[0]), "b": float(p[1])}),
        ),
        (
            "03",
            "ANOMALY DETECTOR",
            "score_one(x)",
            0.68,
            za,
            "score",
            lambda p: hst.score_one({"a": float(p[0]), "b": float(p[1])}),
        ),
    ]
    for num, title, call, x0, z, zl, fz in cols:
        fig.text(x0 + 0.04, 0.78, num, fontsize=13, fontweight="bold", color="#f72585")
        fig.text(
            x0 + 0.075, 0.78, title, fontsize=11, fontweight="bold", color="#ffffff"
        )
        fig.text(
            x0 + 0.04, 0.735, call, fontsize=9.5, color="#cbd5e1", family="monospace"
        )
        zp = np.array([fz(p) for p in pts])
        panel(fig, [x0, 0.0, 0.31, 0.74], g, z, zl, pts, zp)
    fig.savefig(OUT / "overview.png", dpi=200, facecolor=BG)


def title(fig, big, small):
    fig.text(
        0.5,
        0.94,
        big,
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color="#e0fbfc",
    )
    fig.text(0.5, 0.88, small, ha="center", va="center", fontsize=10.5, color="#4cc9f0")


def heading(fig, x0, num, name, code):
    fig.text(x0 + 0.04, 0.78, num, fontsize=13, fontweight="bold", color="#f72585")
    fig.text(x0 + 0.075, 0.78, name, fontsize=11, fontweight="bold", color="#ffffff")
    fig.text(x0 + 0.04, 0.735, code, fontsize=9.5, color="#cbd5e1", family="monospace")


def supervised():
    g = np.linspace(-3.5, 3.5, 36)
    nb = OnlineGaussianNB()
    seen = 0
    fig = plt.figure(figsize=(13.5, 6.0), facecolor=BG)
    title(
        fig,
        "A CLASSIFIER LEARNS ONE SAMPLE AT A TIME",
        "The same model, asked the same question after more and more learn_one calls",
    )
    src = stream(3000, 0)
    for i, (n, x0) in enumerate(((12, 0.02), (60, 0.35), (3000, 0.68))):
        while seen < n:
            x, k = next(src)
            nb.learn_one(x, k)
            seen += 1
        z = surface(lambda x: nb.predict_proba_one(x).get(1, 0.0), g)
        heading(fig, x0, f"0{i + 1}", f"AFTER {n} SAMPLES", "predict_proba_one(x)[1]")
        ax = panel(fig, [x0, 0.0, 0.31, 0.74], g, z, "P(class 1)", None, None)
        ax.set_zlim(0.0, 1.0)
    fig.savefig(OUT / "supervised.png", dpi=200, facecolor=BG)


def transformers():
    rng = np.random.default_rng(3)
    sa, sb = StandardScaler(features=["a"]), StandardScaler(features=["b"])
    for _ in range(2000):
        x = {"a": float(rng.normal(5.0, 2.0)), "b": float(rng.normal(-3.0, 0.5))}
        sa.learn_one(x)
        sb.learn_one(x)
    prod = sa * sb
    ga = np.linspace(1.0, 9.0, 36)
    gb = np.linspace(-4.5, -1.5, 36)
    a, b = np.meshgrid(ga, gb)

    def grid(fn):
        return np.vectorize(lambda u, v: fn({"a": float(u), "b": float(v)}))(a, b)

    za = grid(lambda x: sa.transform_one(x)["a"])
    zb = grid(lambda x: sb.transform_one(x)["b"])
    zp = grid(lambda x: prod.transform_one(x)["a*b"])
    fig = plt.figure(figsize=(13.5, 6.0), facecolor=BG)
    title(
        fig,
        "TWO TRANSFORMERS, ONE PRODUCT",
        "a * b makes one feature for each pair of features; here the two scaled values multiplied",
    )
    specs = (
        (0.02, "01", "TRANSFORMER a", "StandardScaler(features=['a'])", za, "a scaled"),
        (0.35, "02", "TRANSFORMER b", "StandardScaler(features=['b'])", zb, "b scaled"),
        (0.68, "03", "PRODUCT a * b", "(a * b).transform_one(x)['a*b']", zp, "a*b"),
    )
    for x0, num, name, code, z, zl in specs:
        heading(fig, x0, num, name, code)
        ax = panel(fig, [x0, 0.0, 0.31, 0.74], ga, np.zeros((36, 36)), zl, None, None)
        ax.collections[0].remove()
        ax.plot_surface(
            a,
            b,
            z,
            cmap=CMAP,
            alpha=0.78,
            edgecolor="#48cae4",
            linewidth=0.25,
            rstride=1,
            cstride=1,
        )
        ax.set_zlim(float(np.min(z)), float(np.max(z)))
        ax.set_xlabel("a", color="#90e0ef", fontsize=8, fontweight="bold")
        ax.set_ylabel("b", color="#90e0ef", fontsize=8, fontweight="bold")
        ax.set_xlim(1.0, 9.0)
        ax.set_ylim(-4.5, -1.5)
    fig.savefig(OUT / "transformers.png", dpi=200, facecolor=BG)


def detectors():
    rng = np.random.default_rng(1)
    hst = HalfSpaceTrees(seed=0)
    for _ in range(600):
        hst.learn_one({"a": float(rng.normal()), "b": float(rng.normal())})
    g = np.linspace(-3.5, 3.5, 36)
    z = surface(lambda x: hst.score_one(x), g)
    ok = rng.normal(0.0, 1.0, (60, 2))
    bad = np.array([[3.2, 3.0], [-3.0, 2.8], [2.9, -3.1], [-3.2, -3.0]])
    zo = np.array([hst.score_one({"a": float(p[0]), "b": float(p[1])}) for p in ok])
    zb = np.array([hst.score_one({"a": float(p[0]), "b": float(p[1])}) for p in bad])
    fig = plt.figure(figsize=(13.5, 6.0), facecolor=BG)
    title(
        fig,
        "A SCORE FOR EVERY SAMPLE, A FLAG FOR EVERY CHANGE",
        "An anomaly detector scores each point; a drift detector watches one number over time",
    )
    heading(fig, 0.02, "01", "ANOMALY DETECTOR", "score_one(x)  (high = anomalous)")
    ax = panel(fig, [0.02, 0.0, 0.46, 0.74], g, z, "score", ok, zo)
    ax.scatter(
        bad[:, 0],
        bad[:, 1],
        zb + 0.04,
        color="#f72585",
        s=140,
        marker="^",
        edgecolors="#ffffff",
        linewidth=1.0,
        depthshade=False,
    )
    heading(fig, 0.53, "02", "DRIFT DETECTOR", "update(v), then drift_detected")
    ph = PageHinkley()
    vals = rng.normal(0.0, 1.0, 400)
    vals[200:] += 3.0
    flag = None
    for i, v in enumerate(vals):
        ph.update(float(v))
        if ph.drift_detected and flag is None:
            flag = i
    ax2 = fig.add_axes([0.56, 0.12, 0.40, 0.56], facecolor=BG)
    ax2.plot(np.arange(400), vals, color="#4cc9f0", linewidth=0.9)
    ax2.axvline(200, color="#ff9e00", linewidth=1.2, linestyle="--")
    ax2.text(
        196,
        vals.max() - 0.3,
        "mean changes: step 200",
        color="#ff9e00",
        fontsize=9,
        ha="right",
    )
    if flag is not None:
        ax2.axvline(flag, color="#f72585", linewidth=1.8)
        ax2.text(
            flag + 4,
            vals.min() + 0.3,
            f"flag: step {flag}",
            color="#f72585",
            fontsize=9,
            fontweight="bold",
        )
    for sp in ax2.spines.values():
        sp.set_color("#1e293b")
    ax2.tick_params(colors="#64748b", labelsize=8)
    ax2.set_xlabel("step", color="#90e0ef", fontsize=9, fontweight="bold")
    ax2.set_ylabel("value", color="#90e0ef", fontsize=9, fontweight="bold")
    ax2.grid(color="#1e293b", linewidth=0.6)
    fig.savefig(OUT / "detectors.png", dpi=200, facecolor=BG)


class Majority(Classifier):
    """Predict the label seen most often so far."""

    def __init__(self, prior=1.0):
        self.prior = prior
        self.counts_ = {}

    def learn_one(self, x, y, t=None):
        self.counts_[y] = self.counts_.get(y, 0) + 1
        return self

    def predict_proba_one(self, x, t=None):
        n = sum(self.counts_.values())
        return {c: v / n for c, v in self.counts_.items()} if n else {}


def pipelines():
    rng = np.random.default_rng(5)
    pipe = StandardScaler() | OnlineGaussianNB()
    pts, ks = [], []
    for _ in range(1500):
        k = int(rng.integers(2))
        c = 1.0 if k else -1.0
        x = {
            "a": float(rng.normal(25.0 + 3.0 * c, 1.5)),
            "b": float(rng.normal(0.4 * c, 0.35)),
        }
        pipe.learn_one(x, k)
        pts.append(x)
        ks.append(k)
    sc = pipe.steps[0][1]
    raw = np.array([[p["a"], p["b"]] for p in pts[:300]])
    cls = np.array(ks[:300])
    scaled = np.array([[v for v in sc.transform_one(p).values()] for p in pts[:300]])
    fig = plt.figure(figsize=(13.5, 6.0), facecolor=BG)
    title(
        fig,
        "A SAMPLE TRAVELS THROUGH THE CHAIN",
        "StandardScaler | OnlineGaussianNB: each step hands its output to the next one",
    )
    colors = np.where(cls == 1, "#ff9e00", "#4cc9f0")
    ex = pts[ks.index(1)]
    heads = (
        (
            0.02,
            "01",
            "THE RAW SAMPLE",
            f"x = {{'a': {ex['a']:.1f}, 'b': {ex['b']:.2f}}}",
        ),
        (0.35, "02", "AFTER THE SCALER", "StandardScaler.transform_one(x)"),
        (0.68, "03", "THE CLASSIFIER ANSWERS", "pipe.predict_proba_one(x)[1]"),
    )
    for x0, num, name, code in heads:
        heading(fig, x0, num, name, code)
    for x0, data, xl, yl in (
        (0.045, raw, "a", "b"),
        (0.375, scaled, "a (scaled)", "b (scaled)"),
    ):
        ax = fig.add_axes([x0 + 0.01, 0.16, 0.26, 0.52], facecolor=BG)
        ax.scatter(
            data[:, 0],
            data[:, 1],
            c=colors,
            s=16,
            alpha=0.85,
            edgecolors="#e0fbfc",
            linewidth=0.3,
        )
        for sp in ax.spines.values():
            sp.set_color("#1e293b")
        ax.tick_params(colors="#64748b", labelsize=8)
        ax.set_xlabel(xl, color="#90e0ef", fontsize=9, fontweight="bold")
        ax.set_ylabel(yl, color="#90e0ef", fontsize=9, fontweight="bold")
        ax.grid(color="#1e293b", linewidth=0.6)
    for x in (0.325, 0.655):
        fig.text(
            x, 0.40, "\u279c", fontsize=26, color="#f72585", ha="center", va="center"
        )
    ga = np.linspace(20.0, 30.0, 36)
    gb = np.linspace(-1.2, 1.2, 36)
    a, b = np.meshgrid(ga, gb)
    z = np.vectorize(
        lambda u, v: pipe.predict_proba_one({"a": float(u), "b": float(v)}).get(1, 0.0)
    )(a, b)
    ax3 = panel(
        fig, [0.68, 0.0, 0.31, 0.74], ga, np.zeros((36, 36)), "P(class 1)", None, None
    )
    ax3.collections[0].remove()
    ax3.plot_surface(
        a,
        b,
        z,
        cmap=CMAP,
        alpha=0.78,
        edgecolor="#48cae4",
        linewidth=0.25,
        rstride=1,
        cstride=1,
    )
    ax3.set_zlim(0.0, 1.0)
    ax3.set_xlim(20.0, 30.0)
    ax3.set_ylim(-1.2, 1.2)
    fig.text(
        0.5,
        0.012,
        "orange: class 1   cyan: class 0   (the same 300 samples in the first two panels)",
        ha="center",
        fontsize=9,
        color="#64748b",
    )
    fig.savefig(OUT / "pipelines.png", dpi=200, facecolor=BG)


def checks():
    from dense_armor.checks import check_estimator

    res = check_estimator(Majority())
    groups = (
        (
            "PARAMETERS AND CLONE",
            [
                "check_repr",
                "check_repr_clone_equal",
                "check_get_params_signature",
                "check_default_params_non_mutable",
                "check_clone",
                "check_clone_new_params",
                "check_clone_independent",
                "check_clone_drops_learned_state",
                "check_mutate_idempotent",
            ],
        ),
        (
            "STATE AND DESCRIPTION",
            [
                "check_pickle_roundtrip",
                "check_state_dict_roundtrip",
                "check_checkpoint_roundtrip",
                "check_describe_serializable",
                "check_schema_json",
                "check_docstring",
            ],
        ),
        (
            "THE ROLE CONTRACT",
            [
                "check_learn_one_does_not_modify_x",
                "check_predict_before_learning",
                "check_classifier_proba_sum",
                "check_classifier_multiclass_bool",
                "check_roc_auc",
                "check_dt_accepted",
                "check_non_increasing_t_raises",
            ],
        ),
        (
            "TIME AND MEMORY",
            [
                "check_p99_within_budget",
                "check_memory_growth_bounded",
                "check_step_is_pure",
                "check_step_is_jittable",
                "check_estimate_variance_non_negative",
                "check_health_is_reachable",
            ],
        ),
    )
    named = [n for _, names in groups for n in names]
    assert sorted(named) == sorted(res["run"]), (
        "the groups must hold exactly the checks that ran"
    )
    fig = plt.figure(figsize=(13.5, 6.4), facecolor=BG)
    title(
        fig,
        f"{len(res['run'])} CHECKS, ONE VERDICT",
        "check_estimator(Majority()): every tile is a check that ran and passed",
    )
    ax = fig.add_axes([0.03, 0.04, 0.94, 0.78], facecolor=BG)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 7.4)
    ax.axis("off")
    for gi, (name, names) in enumerate(groups):
        x0 = gi + 0.04
        ax.text(
            x0 + 0.02,
            7.15,
            f"0{gi + 1}",
            color="#f72585",
            fontsize=13,
            fontweight="bold",
            va="center",
        )
        ax.text(
            x0 + 0.17,
            7.15,
            name,
            color="#ffffff",
            fontsize=9.5,
            fontweight="bold",
            va="center",
        )
        for i, n in enumerate(names):
            y = 6.45 - i * 0.74
            ax.add_patch(
                plt.Rectangle(
                    (x0, y - 0.3),
                    0.92,
                    0.6,
                    facecolor="#0e3a47",
                    edgecolor="#4cc9f0",
                    linewidth=1.0,
                )
            )
            ax.text(
                x0 + 0.06,
                y,
                n.replace("check_", "").replace("_", " "),
                color="#e0fbfc",
                fontsize=8.2,
                va="center",
            )
    fig.text(
        0.5,
        0.025,
        f"{len(res['run'])} ran, {len(res['not_applicable'])} not applicable, {len(res['skipped'])} skipped",
        ha="center",
        fontsize=9.5,
        color="#4cc9f0",
    )
    fig.savefig(OUT / "checks.png", dpi=200, facecolor=BG)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "overview"):
        overview()
    if which in ("all", "supervised"):
        supervised()
    if which in ("all", "transformers"):
        transformers()
    if which in ("all", "detectors"):
        detectors()
    if which in ("all", "pipelines"):
        pipelines()
    if which in ("all", "checks"):
        checks()
