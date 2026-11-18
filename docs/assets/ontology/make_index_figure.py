from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from dense_armor.core.graph import TypedGraph

BG = "#0d131a"
CARD = "#101a24"
CYAN = "#4cc9f0"
ORANGE = "#ff9e00"
PINK = "#f72585"
TEXT = "#e0fbfc"
MUTED = "#94a3b8"
OUT = Path(__file__).resolve().parent

KINDS = {"service": {"tier": str}, "database": {"env": str}}
SERVICES = (
    ("billing", "production"),
    ("search", "production"),
    ("reports", "staging"),
    ("auth", "production"),
    ("jobs", "production"),
)
DATABASES = (
    ("billing-db", "test"),
    ("search-db", "production"),
    ("reports-db", "test"),
    ("auth-db", "production"),
    ("jobs-db", "test"),
)
LINKS = (
    ("billing", "billing-db"),
    ("billing", "auth-db"),
    ("search", "search-db"),
    ("reports", "reports-db"),
    ("auth", "auth-db"),
    ("jobs", "jobs-db"),
    ("jobs", "search-db"),
)


def rule(svc, db):
    return not (svc["tier"] == "production" and db["env"] == "test")


def build():
    g = TypedGraph(KINDS)
    for n, tier in SERVICES:
        g.add_node(n, "service", tier=tier)
    for n, env in DATABASES:
        g.add_node(n, "database", env=env)
    for a, b in LINKS:
        g.add_edge(a, b, "depends_on", rule)
    return g


def card(ax, x, y, w, h, kind, name, color, field=None, size=9.5, lw=1.6):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.2,rounding_size=1.0",
            fc=CARD,
            ec=color,
            lw=lw,
        )
    )
    ax.text(
        x + 0.9,
        y + h - 1.5,
        kind,
        color=color,
        fontsize=6.6,
        fontweight="bold",
        va="center",
    )
    ax.text(
        x + 0.9,
        y + h - 3.7,
        name,
        color=TEXT,
        fontsize=size,
        fontweight="bold",
        va="center",
    )
    if field:
        ax.text(
            x + 0.9,
            y + 1.3,
            field,
            color=MUTED,
            fontsize=7.2,
            family="monospace",
            va="center",
        )


def draw():
    g = build()
    bad = {(a, b) for a, b, _ in g.validate()}
    fig = plt.figure(figsize=(13.5, 6.4), facecolor=BG)
    fig.text(
        0.5,
        0.945,
        "PARTS, RELATIONS, RULES",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(
        0.5,
        0.895,
        "A system as a typed graph: the library finds every relation that breaks a rule",
        ha="center",
        va="center",
        fontsize=10.5,
        color=CYAN,
    )
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.86], facecolor=BG)
    ax.set_xlim(0, 135)
    ax.set_ylim(0, 60)
    ax.axis("off")
    for x, n, t in (
        (2, "01", "THE PARTS"),
        (38, "02", "THE RELATIONS"),
        (108, "03", "THE RULE"),
    ):
        ax.text(x, 56, n, color=PINK, fontsize=13, fontweight="bold", va="center")
        ax.text(
            x + 4.2, 56, t, color="#ffffff", fontsize=12, fontweight="bold", va="center"
        )
    ax.text(
        2,
        52.2,
        "TypedGraph({'service': ..., 'database': ...})",
        color=MUTED,
        fontsize=7.4,
        family="monospace",
        va="center",
    )
    ax.text(
        38,
        52.2,
        "add_edge(a, b, 'depends_on', rule)",
        color=MUTED,
        fontsize=7.4,
        family="monospace",
        va="center",
    )
    ax.text(
        108,
        52.2,
        "g.validate()",
        color=MUTED,
        fontsize=7.4,
        family="monospace",
        va="center",
    )
    # 01: the two kinds
    card(ax, 2, 33, 27, 12, "KIND", "service", CYAN, "tier: str", size=11)
    card(ax, 2, 15, 27, 12, "KIND", "database", ORANGE, "env: str", size=11)
    ax.text(
        2,
        9.5,
        "every part has a kind and typed fields:",
        color=MUTED,
        fontsize=8,
        va="center",
    )
    ax.text(
        2,
        7.2,
        "a wrong value is refused when added",
        color=MUTED,
        fontsize=8,
        va="center",
    )
    # 02: services on top, databases below, links in between
    w, h, step = 11.4, 8.4, 13.9
    sx = {n: 36 + i * step for i, (n, _) in enumerate(SERVICES)}
    dx = {n: 36 + i * step for i, (n, _) in enumerate(DATABASES)}
    ytop, ybot = 36.0, 12.0
    for a, b in LINKS:
        p = (sx[a] + w / 2, ytop)
        q = (dx[b] + w / 2, ybot + h)
        broken = (a, b) in bad
        ax.add_patch(
            FancyArrowPatch(
                p,
                q,
                arrowstyle="-|>",
                mutation_scale=13,
                color=PINK if broken else CYAN,
                lw=2.6 if broken else 1.3,
                alpha=1.0 if broken else 0.7,
                shrinkA=0,
                shrinkB=1,
                zorder=3 if broken else 2,
            )
        )
    for n, tier in SERVICES:
        card(ax, sx[n], ytop, w, h, "SERVICE", n, CYAN, f"tier: {tier}", size=9)
    for n, env in DATABASES:
        card(ax, dx[n], ybot, w, h, "DATABASE", n, ORANGE, f"env: {env}", size=9)
    ax.text(
        36, 49.0, "services", color=CYAN, fontsize=8.5, fontweight="bold", va="center"
    )
    ax.text(
        36, 8.6, "databases", color=ORANGE, fontsize=8.5, fontweight="bold", va="center"
    )
    # 03: the rule and its verdict
    ax.add_patch(
        FancyBboxPatch(
            (108, 33),
            25,
            14,
            boxstyle="round,pad=0.2,rounding_size=1.2",
            fc=CARD,
            ec=PINK,
            lw=1.8,
        )
    )
    ax.text(
        109.2, 44.3, "RULE", color=PINK, fontsize=7.4, fontweight="bold", va="center"
    )
    ax.text(109.2, 41.0, "a production service", color=TEXT, fontsize=9.2, va="center")
    ax.text(109.2, 38.4, "must not depend on", color=TEXT, fontsize=9.2, va="center")
    ax.text(109.2, 35.8, "a test database", color=TEXT, fontsize=9.2, va="center")
    ax.add_patch(
        FancyBboxPatch(
            (108, 9),
            25,
            19,
            boxstyle="round,pad=0.2,rounding_size=1.2",
            fc=CARD,
            ec=PINK,
            lw=1.8,
            ls="--",
        )
    )
    ax.text(
        109.2,
        25.2,
        "validate()",
        color=PINK,
        fontsize=9.4,
        fontweight="bold",
        family="monospace",
        va="center",
    )
    ax.text(
        109.2, 22.0, f"{len(bad)} wrong relations:", color=TEXT, fontsize=9, va="center"
    )
    for i, (a, b) in enumerate(sorted(bad)):
        ax.text(
            109.2,
            18.6 - i * 3.2,
            f"{a} -> {b}",
            color=PINK,
            fontsize=9,
            family="monospace",
            va="center",
        )
    ax.text(
        109.2,
        11.6,
        f"{len(LINKS) - len(bad)} of {len(LINKS)} are fine",
        color=MUTED,
        fontsize=8.2,
        va="center",
    )
    fig.text(
        0.5,
        0.025,
        "pink: a production service that depends on a test database",
        ha="center",
        fontsize=9.5,
        color=PINK,
    )
    fig.savefig(OUT / "index.png", dpi=200, facecolor=BG)
    print("wrong relations:", sorted(bad))


if __name__ == "__main__":
    draw()
