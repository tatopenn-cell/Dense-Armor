from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.core.graph import TypedGraph

BG = "#0d131a"
OUT = Path(__file__).resolve().parent


def rule(svc, db):
    return not (svc["tier"] == "production" and db["env"] == "test")


def build():
    g = TypedGraph({"service": {"tier": str}, "database": {"env": str}})
    services = [
        ("billing", "production"),
        ("search", "production"),
        ("reports", "staging"),
        ("auth", "production"),
        ("mailer", "staging"),
        ("gateway", "production"),
        ("cache", "staging"),
        ("jobs", "production"),
    ]
    databases = [
        ("billing-db", "test"),
        ("search-db", "production"),
        ("reports-db", "test"),
        ("auth-db", "production"),
        ("events-db", "production"),
        ("jobs-db", "test"),
    ]
    for n, tier in services:
        g.add_node(n, "service", tier=tier)
    for n, env in databases:
        g.add_node(n, "database", env=env)
    links = [
        ("billing", "billing-db"),
        ("billing", "auth-db"),
        ("search", "search-db"),
        ("reports", "reports-db"),
        ("auth", "auth-db"),
        ("mailer", "events-db"),
        ("gateway", "auth-db"),
        ("gateway", "search-db"),
        ("cache", "events-db"),
        ("jobs", "jobs-db"),
        ("jobs", "events-db"),
    ]
    for a, b in links:
        g.add_edge(a, b, "depends_on", rule)
    return g, [n for n, _ in services], [n for n, _ in databases], links


def draw():
    g, svc, dbs, links = build()
    bad = {(a, b) for a, b, _ in g.validate()}
    pos = {}
    for i, n in enumerate(svc):
        t = 2 * np.pi * i / len(svc)
        pos[n] = (3.6 * np.cos(t), 3.6 * np.sin(t), 1.0)
    for i, n in enumerate(dbs):
        t = 2 * np.pi * (i + 0.5) / len(dbs)
        pos[n] = (2.5 * np.cos(t), 2.5 * np.sin(t), -1.0)
    fig = plt.figure(figsize=(13.5, 6.0), facecolor=BG)
    fig.text(
        0.5,
        0.93,
        "PARTS, RELATIONS, RULES",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color="#e0fbfc",
    )
    fig.text(
        0.5,
        0.865,
        "A system as a typed graph: the library finds every relation that breaks a rule",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#4cc9f0",
    )
    notes = (
        (
            0.06,
            "01",
            "WHAT THE PARTS ARE",
            "TypedGraph({'service': ..., 'database': ...})",
        ),
        (
            0.38,
            "02",
            "HOW THEY ARE RELATED",
            "add_edge(a, b, 'depends_on', rule)",
        ),
        (0.70, "03", "WHAT MUST BE TRUE", f"validate() -> {len(bad)} wrong relations"),
    )
    for x0, num, name, code in notes:
        fig.text(x0, 0.77, num, fontsize=14, fontweight="bold", color="#f72585")
        fig.text(x0 + 0.04, 0.77, name, fontsize=12, fontweight="bold", color="#ffffff")
        fig.text(x0, 0.715, code, fontsize=9.5, color="#cbd5e1", family="monospace")
    ax = fig.add_axes([0.0, -0.05, 1.0, 0.78], projection="3d")
    ax.set_facecolor(BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#1e293b")
        axis._axinfo["grid"]["color"] = (0.20, 0.30, 0.42, 0.25)
        axis._axinfo["grid"]["linewidth"] = 0.5
    th = np.linspace(0, 2 * np.pi, 120)
    ax.plot(
        4.2 * np.cos(th),
        4.2 * np.sin(th),
        np.full_like(th, 1.0),
        color="#4cc9f0",
        linewidth=0.8,
        alpha=0.35,
    )
    ax.plot(
        3.1 * np.cos(th),
        3.1 * np.sin(th),
        np.full_like(th, -1.0),
        color="#ff9e00",
        linewidth=0.8,
        alpha=0.35,
    )
    for a, b in links:
        x = [pos[a][0], pos[b][0]]
        y = [pos[a][1], pos[b][1]]
        z = [pos[a][2], pos[b][2]]
        if (a, b) in bad:
            ax.plot(x, y, z, color="#f72585", linewidth=3.2, alpha=0.95)
        else:
            ax.plot(x, y, z, color="#4cc9f0", linewidth=1.1, alpha=0.55)
    sx = np.array([pos[n] for n in svc])
    dx = np.array([pos[n] for n in dbs])
    ax.scatter(
        sx[:, 0],
        sx[:, 1],
        sx[:, 2],
        s=260,
        color="#4cc9f0",
        edgecolors="#e0fbfc",
        linewidth=0.8,
        depthshade=False,
    )
    ax.scatter(
        dx[:, 0],
        dx[:, 1],
        dx[:, 2],
        s=260,
        color="#ff9e00",
        edgecolors="#ffffff",
        linewidth=0.8,
        marker="s",
        depthshade=False,
    )
    for n in svc + dbs:
        x, y, z = pos[n]
        ax.text(
            x * 1.12,
            y * 1.12,
            z + (0.3 if z > 0 else -0.35),
            n,
            color="#e0fbfc",
            fontsize=11,
            ha="center",
        )
    ax.text(
        -4.9,
        0.0,
        1.0,
        "SERVICES",
        color="#4cc9f0",
        fontsize=10,
        fontweight="bold",
        ha="center",
    )
    ax.text(
        -4.9,
        0.0,
        -1.0,
        "DATABASES",
        color="#ff9e00",
        fontsize=10,
        fontweight="bold",
        ha="center",
    )
    ax.set_xlim(-5.4, 5.4)
    ax.set_ylim(-5.4, 5.4)
    ax.set_zlim(-1.6, 1.6)
    ax.set_box_aspect((1, 1, 0.5), zoom=1.7)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=-62)
    fig.text(
        0.5,
        0.025,
        "pink: a production service that depends on a test database",
        ha="center",
        fontsize=9.5,
        color="#f72585",
    )
    fig.savefig(OUT / "index.png", dpi=200, facecolor=BG)
    print("wrong relations:", sorted(bad))


if __name__ == "__main__":
    draw()
