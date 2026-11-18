from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from dense_armor.core.graph import Action, ActionRejected, TypedGraph, apply_action

BG = "#0d131a"
CARD = "#101a24"
CYAN = "#4cc9f0"
ORANGE = "#ff9e00"
PINK = "#f72585"
TEXT = "#e0fbfc"
MUTED = "#94a3b8"
OUT = Path(__file__).resolve().parent


def rule(svc, db):
    return not (svc["tier"] == "production" and db["env"] == "test")


def run():
    g = TypedGraph({"service": {"tier": str}, "database": {"env": str}})
    for name, env in (("search", "production"), ("reports", "test")):
        g.add_node(name, "service", tier="staging")
        g.add_node(f"{name}-db", "database", env=env)
        g.add_edge(name, f"{name}-db", "depends_on", rule)
    promote = Action(
        "promote",
        [("s", "tier", "==", "staging")],
        [("s", "tier", "set", "production")],
    )
    result = {}
    for s in ("search", "reports"):
        try:
            h = apply_action(g, promote, {"s": s})
            result[s] = ("accepted", h.graph.nodes[s]["state"]["tier"], None)
        except ActionRejected as e:
            result[s] = ("rejected", g.graph.nodes[s]["state"]["tier"], e.failed)
    return g, promote, result


def card(ax, x, y, kind, name, fields, color, w=17.0, h=11.0, dashed=False, glow=False):
    if glow:
        ax.add_patch(
            FancyBboxPatch(
                (x - 0.5, y - 0.5),
                w + 1.0,
                h + 1.0,
                boxstyle="round,pad=0.2,rounding_size=1.4",
                fc="none",
                ec=color,
                lw=5,
                alpha=0.25,
            )
        )
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.2,rounding_size=1.2",
            fc=CARD,
            ec=color,
            lw=1.8,
            ls="--" if dashed else "-",
        )
    )
    ax.text(
        x + 1.2,
        y + h - 1.8,
        kind,
        color=color,
        fontsize=7.5,
        fontweight="bold",
        va="center",
    )
    ax.text(
        x + 1.2,
        y + h - 4.4,
        name,
        color=TEXT,
        fontsize=11,
        fontweight="bold",
        va="center",
    )
    for i, (k, v) in enumerate(fields):
        ax.text(
            x + 1.2,
            y + 2.6 - i * 2.4,
            f"{k}: {v}",
            color=MUTED,
            fontsize=8.6,
            family="monospace",
            va="center",
        )


def arrow(ax, p, q, color, label=None, lw=1.8, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            p,
            q,
            arrowstyle="-|>",
            mutation_scale=14,
            color=color,
            lw=lw,
            ls=ls,
            shrinkA=0,
            shrinkB=0,
        )
    )
    if label:
        ax.text(
            (p[0] + q[0]) / 2,
            p[1] + 1.4,
            label,
            color=color,
            fontsize=7.8,
            ha="center",
            family="monospace",
        )


def draw():
    g, promote, res = run()
    fig = plt.figure(figsize=(13.5, 6.4), facecolor=BG)
    fig.text(
        0.5,
        0.945,
        "A CHANGE IS CHECKED BEFORE IT HAPPENS",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(
        0.5,
        0.895,
        "apply_action works on a copy and refuses the change if it would break a rule that held",
        ha="center",
        va="center",
        fontsize=10.5,
        color=CYAN,
    )
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.86], facecolor=BG)
    ax.set_xlim(0, 135)
    ax.set_ylim(0, 60)
    ax.axis("off")
    heads = ((2, "01", "BEFORE"), (51, "02", "THE ACTION"), (90, "03", "AFTER"))
    for x, n, t in heads:
        ax.text(x, 56, n, color=PINK, fontsize=13, fontweight="bold", va="center")
        ax.text(
            x + 4.2, 56, t, color="#ffffff", fontsize=12, fontweight="bold", va="center"
        )
    rows = (("search", 38.0), ("reports", 15.0))
    for name, y in rows:
        envs = g.graph.nodes[f"{name}-db"]["state"]["env"]
        card(ax, 2, y, "SERVICE", name, [("tier", "staging")], CYAN)
        card(ax, 27, y, "DATABASE", f"{name}-db", [("env", envs)], ORANGE)
        arrow(ax, (19.4, y + 5.5), (26.6, y + 5.5), CYAN, "depends_on")
        arrow(ax, (44.6, y + 5.5), (51.2, y + 5.5), MUTED)
        arrow(ax, (83.6, y + 5.5), (89.6, y + 5.5), MUTED)
    ax.add_patch(
        FancyBboxPatch(
            (51.5, 17.5),
            32,
            31,
            boxstyle="round,pad=0.2,rounding_size=1.4",
            fc=CARD,
            ec=PINK,
            lw=1.8,
        )
    )
    ax.text(
        53.5, 45.0, "ACTION", color=PINK, fontsize=8, fontweight="bold", va="center"
    )
    ax.text(
        53.5,
        41.8,
        "promote(s)",
        color=TEXT,
        fontsize=13,
        fontweight="bold",
        va="center",
    )
    ax.text(53.5, 37.0, "precondition", color=MUTED, fontsize=8, va="center")
    ax.text(
        53.5,
        34.4,
        "s.tier == 'staging'",
        color=TEXT,
        fontsize=9.4,
        family="monospace",
        va="center",
    )
    ax.text(53.5, 30.0, "effect", color=MUTED, fontsize=8, va="center")
    ax.text(
        53.5,
        27.4,
        "s.tier = 'production'",
        color=TEXT,
        fontsize=9.4,
        family="monospace",
        va="center",
    )
    ax.text(53.5, 22.4, "then: does a rule break", color=MUTED, fontsize=8, va="center")
    ax.text(53.5, 20.0, "that held before?", color=MUTED, fontsize=8, va="center")
    # outcomes
    state, tier, _ = res["search"]
    card(ax, 90, 38, "SERVICE", "search", [("tier", tier)], CYAN, glow=True)
    card(ax, 115, 38, "DATABASE", "search-db", [("env", "production")], ORANGE)
    arrow(ax, (107.4, 43.5), (114.6, 43.5), CYAN, "depends_on")
    ax.text(
        90,
        34.2,
        "✓ " + state.upper(),
        color=CYAN,
        fontsize=11,
        fontweight="bold",
        va="center",
    )
    ax.text(90, 31.6, f"search is now '{tier}'", color=MUTED, fontsize=8.8, va="center")
    state, tier, failed = res["reports"]
    card(ax, 90, 15, "SERVICE", "reports", [("tier", tier)], PINK, dashed=True)
    card(ax, 115, 15, "DATABASE", "reports-db", [("env", "test")], PINK, dashed=True)
    arrow(ax, (107.4, 20.5), (114.6, 20.5), PINK, "depends_on", ls="--")
    ax.text(
        90,
        11.2,
        "✗ " + state.upper(),
        color=PINK,
        fontsize=11,
        fontweight="bold",
        va="center",
    )
    link = failed[0]
    ax.text(
        90,
        8.4,
        f"it would break: {link[0]} -> {link[1]}",
        color=MUTED,
        fontsize=8.8,
        va="center",
    )
    ax.text(
        90, 6.0, "the graph is left unchanged", color=MUTED, fontsize=8.8, va="center"
    )
    fig.text(
        0.5,
        0.025,
        "the rule: a production service must not depend on a test database",
        ha="center",
        fontsize=9.5,
        color=CYAN,
    )
    fig.savefig(OUT / "typed.png", dpi=200, facecolor=BG)
    print(res)


if __name__ == "__main__":
    draw()
