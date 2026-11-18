from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from dense_armor.core.online.anomaly.hst import HalfSpaceTrees
from dense_armor.core.online.drift.page_hinkley import PageHinkley
from dense_armor.core.online.learn.online_classifiers import OnlineGaussianNB
from dense_armor.core.online.tree.hoeffding_regressor import HoeffdingTreeRegressor
from dense_armor.core.signals.preprocessing.scale import StandardScaler
from dense_armor.roles import (
    AnomalyDetector,
    AnomalyGate,
    Classifier,
    DriftDetector,
    Estimator,
    ModelWrapper,
    Protected,
    Regressor,
    Root,
    Transformer,
)

BG = "#0d131a"
CARD = "#101a24"
CYAN = "#4cc9f0"
ORANGE = "#ff9e00"
PINK = "#f72585"
TEXT = "#e0fbfc"
MUTED = "#94a3b8"
OUT = Path(__file__).resolve().parent

ROLES = (
    (Classifier, ("learn_one(x, y)", "predict_proba_one(x)"), OnlineGaussianNB),
    (Regressor, ("learn_one(x, y)", "predict_one(x)"), HoeffdingTreeRegressor),
    (Transformer, ("learn_one(x)", "transform_one(x)"), StandardScaler),
    (AnomalyDetector, ("learn_one(x)", "score_one(x)"), HalfSpaceTrees),
    (DriftDetector, ("update(v)", "drift_detected"), PageHinkley),
    (ModelWrapper, ("_wrapped_model", "delegates to it"), None),
)


def box(ax, x, y, w, h, color, ls="-", lw=1.7):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.2,rounding_size=1.1",
            fc=CARD,
            ec=color,
            lw=lw,
            ls=ls,
        )
    )


def link(ax, p, q, color=MUTED):
    ax.add_patch(
        FancyArrowPatch(
            p,
            q,
            arrowstyle="-|>",
            mutation_scale=13,
            color=color,
            lw=1.4,
            shrinkA=0,
            shrinkB=1,
        )
    )


def draw():
    for role, _, example in ROLES:
        assert role.__mro__[1] is Estimator, role
        assert example is None or issubclass(example, role), example
    assert Estimator.__mro__[1] is Root
    assert issubclass(Protected, Estimator) and issubclass(AnomalyGate, Estimator)
    assert not ModelWrapper.__subclasses__()

    fig = plt.figure(figsize=(13.5, 6.8), facecolor=BG)
    fig.text(
        0.5,
        0.95,
        "THE FAMILY OF ROLES",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(
        0.5,
        0.905,
        "Every tool is a Root and an Estimator; most also play one role, and the role fixes the questions you can ask",
        ha="center",
        va="center",
        fontsize=10.5,
        color=CYAN,
    )
    ax = fig.add_axes([0.0, 0.06, 1.0, 0.83], facecolor=BG)
    ax.set_xlim(0, 135)
    ax.set_ylim(0, 66)
    ax.axis("off")

    box(ax, 50, 53, 35, 11, CYAN)
    ax.text(
        51.2, 61.5, "ROOT", color=CYAN, fontsize=7.4, fontweight="bold", va="center"
    )
    ax.text(
        51.2,
        58.4,
        "get_params  clone  mutate",
        color=TEXT,
        fontsize=8.6,
        family="monospace",
        va="center",
    )
    ax.text(
        51.2,
        55.6,
        "state_dict  describe  fit",
        color=TEXT,
        fontsize=8.6,
        family="monospace",
        va="center",
    )
    link(ax, (67.5, 52.6), (67.5, 47.4))

    box(ax, 50, 36, 35, 11, CYAN)
    ax.text(
        51.2,
        44.5,
        "ESTIMATOR",
        color=CYAN,
        fontsize=7.4,
        fontweight="bold",
        va="center",
    )
    ax.text(
        51.2,
        41.4,
        "a | b   builds a pipeline",
        color=TEXT,
        fontsize=8.6,
        family="monospace",
        va="center",
    )
    ax.text(
        51.2,
        38.6,
        "tags, budget_s, memory_class",
        color=MUTED,
        fontsize=8.2,
        family="monospace",
        va="center",
    )

    box(ax, 3, 36, 36, 11, ORANGE, ls="--")
    ax.text(
        4.2,
        44.5,
        "ALSO ESTIMATORS",
        color=ORANGE,
        fontsize=7.4,
        fontweight="bold",
        va="center",
    )
    ax.text(
        4.2,
        41.4,
        "Protected   AnomalyGate",
        color=TEXT,
        fontsize=8.6,
        family="monospace",
        va="center",
    )
    ax.text(
        4.2,
        38.6,
        "wrap a model, no role of their own",
        color=MUTED,
        fontsize=8,
        va="center",
    )
    link(ax, (49.6, 41.5), (39.6, 41.5), ORANGE)

    w, gap = 19.0, 3.0
    for i, (role, methods, example) in enumerate(ROLES):
        x = 3 + i * (w + gap)
        link(ax, (67.5, 35.6), (x + w / 2, 28.6), MUTED)
        box(ax, x, 17, w, 11, CYAN)
        ax.text(
            x + 1,
            25.5,
            "ROLE",
            color=CYAN,
            fontsize=6.8,
            fontweight="bold",
            va="center",
        )
        ax.text(
            x + 1,
            22.5,
            role.__name__,
            color=TEXT,
            fontsize=9.6,
            fontweight="bold",
            va="center",
        )
        ax.text(
            x + 1,
            19.9,
            methods[0],
            color=MUTED,
            fontsize=7.2,
            family="monospace",
            va="center",
        )
        ax.text(
            x + 1,
            18.1,
            methods[1],
            color=MUTED,
            fontsize=7.2,
            family="monospace",
            va="center",
        )
        link(ax, (x + w / 2, 16.6), (x + w / 2, 11.6), ORANGE if example else MUTED)
        box(ax, x, 3, w, 8, ORANGE if example else MUTED, ls="--", lw=1.3)
        ax.text(
            x + 1,
            9.0,
            "A REAL CLASS" if example else "IN THE LIBRARY",
            color=ORANGE if example else MUTED,
            fontsize=6.4,
            fontweight="bold",
            va="center",
        )
        ax.text(
            x + 1,
            6.0,
            example.__name__ if example else "no subclass yet",
            color=TEXT if example else MUTED,
            fontsize=8.6 if example else 8,
            fontweight="bold" if example else "normal",
            va="center",
        )
    fig.text(
        0.5,
        0.02,
        "solid arrow: inherits from   dashed box: a class of the library that plays the role",
        ha="center",
        fontsize=9.5,
        color=MUTED,
    )
    fig.savefig(OUT / "family.png", dpi=200, facecolor=BG)
    print("ok")


if __name__ == "__main__":
    draw()
