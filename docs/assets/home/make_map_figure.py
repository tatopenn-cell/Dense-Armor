from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from dense_armor.usable.catalog import get_catalog

BG = "#0d131a"
BAND = "#0f1822"
CARD = "#101a24"
CYAN = "#4cc9f0"
ORANGE = "#ff9e00"
PINK = "#f72585"
TEXT = "#e0fbfc"
MUTED = "#94a3b8"
LINE = "#1e293b"
OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]


def counts():
    items = [i for i in get_catalog(REPO) if i.get("kind", "class") == "class"]
    c = Counter()
    roles = Counter()
    for i in items:
        p = i["module"].split(".")
        key = (
            ".".join(p[1:4])
            if len(p) > 3 and p[1] == "core"
            else ".".join(p[1:3])
            if len(p) > 2
            else p[1]
        )
        c[key] += 1
        roles[i["role"]] += 1

    def total(prefix):
        return sum(v for k, v in c.items() if k.startswith(prefix))

    return {
        "online": total("core.online"),
        "ml": total("ml"),
        "deep": total("core.deep"),
        "vision": total("core.signals.vision"),
        "audio": total("core.audio"),
        "finance": total("core.finance"),
        "graph": total("core.graph"),
        "stats": total("core.signals.stats"),
        "prep": total("core.signals.preprocessing"),
        "eval": total("core.evaluation"),
        "datasets": total("core.signals.datasets"),
        "online_parts": {
            k.split(".")[-1]: v for k, v in c.items() if k.startswith("core.online")
        },
        "roles": roles,
        "all": len(items),
    }


def band(ax, x, y, w, h, num, title, note=None):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.2,rounding_size=1.4",
            fc=BAND,
            ec=LINE,
            lw=1.2,
        )
    )
    ax.text(
        x + 1.4,
        y + h - 2.0,
        num,
        color=PINK,
        fontsize=11,
        fontweight="bold",
        va="center",
    )
    ax.text(
        x + 4.8,
        y + h - 2.0,
        title,
        color="#ffffff",
        fontsize=10.5,
        fontweight="bold",
        va="center",
    )
    if note:
        ax.text(
            x + w - 1.4,
            y + h - 2.0,
            note,
            color=MUTED,
            fontsize=8,
            ha="right",
            va="center",
        )


def card(ax, x, y, w, h, name, line1, color=CYAN, line2=None, ls="-"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.2,rounding_size=1.0",
            fc=CARD,
            ec=color,
            lw=1.5,
            ls=ls,
        )
    )
    ax.text(
        x + 1.0,
        y + h - 2.1,
        name,
        color=TEXT,
        fontsize=10,
        fontweight="bold",
        va="center",
    )
    ax.text(
        x + 1.0,
        y + h - 4.6,
        line1,
        color=color,
        fontsize=8.6,
        fontweight="bold",
        va="center",
    )
    if line2:
        ax.text(x + 1.0, y + h - 6.9, line2, color=MUTED, fontsize=7.6, va="center")


def down(ax, x, y0, y1):
    ax.add_patch(
        FancyArrowPatch(
            (x, y0),
            (x, y1),
            arrowstyle="-|>",
            mutation_scale=13,
            color=MUTED,
            lw=1.4,
            shrinkA=0,
            shrinkB=0,
        )
    )


def draw():
    n = counts()
    fig = plt.figure(figsize=(13.5, 9.4), facecolor=BG)
    fig.text(
        0.5,
        0.965,
        "THE MAP OF DENSE-ARMOR",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(
        0.5,
        0.932,
        f"Five layers, each one uses the ones below it; {n['all']} classes in the catalog",
        ha="center",
        va="center",
        fontsize=10.5,
        color=CYAN,
    )
    ax = fig.add_axes([0.0, 0.035, 1.0, 0.88], facecolor=BG)
    ax.set_xlim(0, 135)
    ax.set_ylim(0, 96)
    ax.axis("off")

    band(ax, 1, 79, 133, 16, "01", "APPLICATIONS", "what the library was built for")
    apps = (
        ("Robot control", "core.robot", "rate limiter, CBF filter"),
        ("Drones", "pipelines/drones", "faults, battery, landing"),
        ("Finance", f"{n['finance']} classes", "decision log, pre-trade"),
        ("Vision", f"{n['vision']} classes", "frames, motion, landmarks"),
        ("Audio", f"{n['audio']} classes", "filters, STFT, features"),
        ("IoT and biomedical", "pipelines", "telemetry, ECG"),
    )
    for i, (a, b, c) in enumerate(apps):
        card(ax, 2.5 + i * 21.8, 80.2, 19.6, 9.2, a, b, ORANGE, c)

    down(ax, 22, 78.4, 75.2)
    down(ax, 67, 78.4, 75.2)
    band(ax, 1, 61, 88, 14, "02", "PROTECTION", "optional, around any online model")
    prot = (
        ("Armor", "one series", None),
        ("Orca", "a whole model", None),
        ("StreamingArbiter", "segments, streaming", None),
        ("StreamingHealing", "healing filter", None),
    )
    for i, (a, b, _) in enumerate(prot):
        card(ax, 2.5 + i * 21.4, 62.2, 20.0, 7.4, a, b, PINK)
    band(ax, 91, 61, 43, 14, "+", "BRIDGES", "optional extras")
    ax.text(
        93, 68.4, "river  scikit-learn  PyTorch", color=TEXT, fontsize=8.8, va="center"
    )
    ax.text(93, 65.6, "TensorFlow  LLM agents", color=TEXT, fontsize=8.8, va="center")
    ax.text(
        93,
        63.0,
        "so other libraries can use the tools",
        color=MUTED,
        fontsize=7.6,
        va="center",
    )

    down(ax, 22, 60.4, 57.2)
    down(ax, 67, 60.4, 57.2)
    band(ax, 1, 36, 133, 21, "03", "THREE KINDS OF LEARNING", "the core")
    parts = sorted(n["online_parts"].items(), key=lambda kv: -kv[1])[:4]
    chips = "  ".join(f"{k} {v}" for k, v in parts)
    learn = (
        ("Online learning", f"{n['online']} classes", "one sample at a time", chips),
        (
            "Machine learning",
            f"{n['ml']} classes",
            "fit and predict on a whole table",
            "statistics, linear, trees, ensembles",
        ),
        (
            "Deep learning",
            f"{n['deep']} classes",
            "layers, attention, training",
            "language models, continual learning",
        ),
    )
    for i, (a, b, c, d) in enumerate(learn):
        x = 2.5 + i * 44.2
        card(ax, x, 37.4, 41.2, 14.2, a, b, CYAN, c)
        ax.text(x + 1.0, 37.4 + 3.2, d, color=MUTED, fontsize=7.4, va="center")

    down(ax, 22, 35.4, 32.2)
    down(ax, 67, 35.4, 32.2)
    down(ax, 112, 35.4, 32.2)
    band(ax, 1, 17, 133, 15, "04", "SIGNALS, STATISTICS AND GRAPHS", "the layer below")
    sig = (
        ("Statistics", f"{n['stats']} classes", "moments, covariance"),
        ("Preprocessing", f"{n['prep']} classes", "scalers, selectors"),
        ("Metrics", f"{n['eval']} classes", "accuracy, F1, model selection"),
        ("Graph and ontology", f"{n['graph']} classes", "typed graphs, rules"),
        ("Datasets", f"{n['datasets']} classes", "Casper, DriftStream"),
    )
    for i, (a, b, c) in enumerate(sig):
        card(ax, 2.5 + i * 26.2, 18.2, 24.0, 9.2, a, b, CYAN, c)

    down(ax, 67, 16.4, 13.2)
    band(ax, 1, 1, 133, 12, "05", "ROLES", "Root > Estimator > one role")
    order = (
        "Classifier",
        "Regressor",
        "Transformer",
        "AnomalyDetector",
        "DriftDetector",
        "ModelWrapper",
    )
    for i, r in enumerate(order):
        k = n["roles"].get(r, 0)
        card(
            ax,
            2.5 + i * 21.8,
            1.6,
            19.6,
            7.2,
            r,
            f"{k} classes" if k else "no class yet",
            ORANGE if k else MUTED,
            None,
            "-" if k else "--",
        )
    fig.text(
        0.5,
        0.012,
        "counts: classes in the catalog of the library (dense_armor.usable.catalog), drawn by docs/assets/home/make_map_figure.py",
        ha="center",
        fontsize=8.6,
        color=MUTED,
    )
    fig.savefig(OUT / "map.png", dpi=200, facecolor=BG)
    print(
        {k: v for k, v in n.items() if k not in ("roles", "online_parts")},
        dict(n["roles"]),
    )


if __name__ == "__main__":
    draw()
