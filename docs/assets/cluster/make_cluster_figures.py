"""Dark figures for the clustering pages, drawn with the library itself on the examples of the pages.

Generates in the same folder:
- cluster_customers.png    OnlineKMeans on the kinds of customers (overview page)
- kmeans_follow.png        the centres of OnlineKMeans on groups that move, with and without a half-life
- denstream_clusters.png   DenStream on three groups with noise
- vocabulary_hist.png      the visual vocabulary on two textures

Run from the repo root:
    python docs/assets/cluster/make_cluster_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dense_armor.core.online.cluster.denstream import DenStream
from dense_armor.core.online.cluster.kmeans import OnlineKMeans
from dense_armor.core.signals.vision.vocabulary import BagOfVisualWords

HERE = Path(__file__).resolve().parent
BG, PANEL, GRID = "#0b1118", "#111a24", "#26364a"
INK, MUTED = "#e8fbff", "#8fa3b8"
CYAN, ORANGE, PINK = "#3ec6e8", "#ffa21f", "#ff2d87"
COLS = [CYAN, ORANGE, PINK]

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "grid.color": GRID,
        "grid.alpha": 0.6,
        "font.size": 10,
    }
)


def heading(fig, title, sub):
    fig.text(0.04, 0.945, title, fontsize=20, fontweight="bold", color=INK)
    fig.text(0.04, 0.885, sub, fontsize=11.5, color=CYAN)


def customers():
    rng = np.random.default_rng(0)
    kinds = np.array([[2.0, 30.0], [12.0, 60.0], [30.0, 400.0]])
    km = OnlineKMeans(k=3)
    pts = []
    for _ in range(900):
        pts.append(kinds[rng.integers(0, 3)] + rng.normal(0, [1.0, 10.0]))
        km.learn_one({"x": pts[-1].tolist()})
    X = np.array(pts)
    lab = np.array([km.predict_one({"x": p.tolist()}) for p in X])
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6))
    fig.subplots_adjust(top=0.76, bottom=0.13, left=0.06, right=0.985, wspace=0.18)
    heading(
        fig,
        "THE KINDS OF CUSTOMERS THAT NOBODY NAMED",
        "900 customers seen one at a time: payments per month and average amount; the three centres found by OnlineKMeans",
    )
    for ax, logy in zip(axes, (False, True)):
        for g in range(3):
            ax.scatter(X[lab == g, 0], X[lab == g, 1], s=9, color=COLS[g], alpha=0.55)
        c = np.array(km.centers_)
        ax.scatter(
            c[:, 0],
            c[:, 1],
            s=150,
            marker="X",
            color=INK,
            edgecolor=BG,
            zorder=5,
            label="centres",
        )
        ax.set_xlabel("payments per month")
        ax.grid(True)
        if logy:
            ax.set_yscale("log")
            ax.set_title(
                "same points, amount on a log scale",
                color=MUTED,
                loc="left",
                fontsize=11,
            )
        else:
            ax.set_ylabel("average amount of a payment")
    axes[0].legend(loc="upper left", facecolor=PANEL, edgecolor=GRID, labelcolor=INK)
    fig.savefig(HERE / "cluster_customers.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("customers centres:", sorted([round(float(v), 1) for v in c_] for c_ in c))


def follow():
    out = {}
    for name, hl in (("never forgets (1/n step)", None), ("halflife = 50", 50)):
        rng = np.random.default_rng(1)
        km = OnlineKMeans(k=2, halflife=hl)
        track = []
        for i in range(3000):
            shift = i / 3000 * 5.0
            c = [shift, 0.0] if i % 2 == 0 else [shift, 10.0]
            km.learn_one({"x": (np.array(c) + rng.normal(0, 0.3, 2)).tolist()})
            if i >= 40:
                cs = sorted(np.array(km.centers_).tolist(), key=lambda p: p[1])
                track.append((i, cs[0][0], cs[1][0]))
        out[name] = np.array(track)
    fig, ax = plt.subplots(figsize=(16, 5.2))
    fig.subplots_adjust(top=0.76, bottom=0.14, left=0.06, right=0.985)
    heading(
        fig,
        "FOLLOWING GROUPS THAT MOVE",
        "two groups slide from x = 0 to x = 5 over 3000 points; the x of the centre of the lower group, with and without a half-life",
    )
    t = np.arange(40, 3000)
    ax.plot(t, t / 3000 * 5.0, color=INK, lw=1.4, ls="--", label="where the group is")
    for (name, tr), col in zip(out.items(), (ORANGE, CYAN)):
        ax.plot(tr[:, 0], tr[:, 1], color=col, lw=2.4, label=name)
    ax.set_xlabel("points seen")
    ax.set_ylabel("x of the centre")
    ax.legend(loc="upper left", facecolor=PANEL, edgecolor=GRID, labelcolor=INK)
    ax.grid(True)
    fig.savefig(HERE / "kmeans_follow.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print(
        "follow, final x of the lower group:",
        {k: round(float(v[-1, 1]), 2) for k, v in out.items()},
    )


def denstream():
    rng = np.random.default_rng(0)
    ds = DenStream(eps=1.0, beta=0.4, mu=3.0, decay=0.01)
    centres = np.array([[0.0, 0.0], [8.0, 0.0], [0.0, 8.0]])
    pts, noise = [], []
    t = 0.0
    for i in range(600):
        if i % 20:
            x = centres[i % 3] + rng.normal(0, 0.3, 2)
            pts.append(x)
        else:
            x = rng.uniform(-4, 12, 2)
            noise.append(x)
        ds.learn_one({"x": x.tolist()}, t=t)
        t += 1.0
    clusters, _ = ds.macro_clusters()
    fig, ax = plt.subplots(figsize=(16, 6.2))
    fig.subplots_adjust(top=0.78, bottom=0.12, left=0.06, right=0.985)
    heading(
        fig,
        "THREE HIDEOUTS AND THE PRANKSTERS' PINS",
        "600 points, one in twenty is noise; DenStream keeps micro-clusters and macro_clusters() merges the real ones",
    )
    P, N = np.array(pts), np.array(noise)
    ax.scatter(
        P[:, 0],
        P[:, 1],
        s=10,
        color=CYAN,
        alpha=0.5,
        label="points of the three groups",
    )
    ax.scatter(
        N[:, 0],
        N[:, 1],
        s=26,
        color=PINK,
        marker="x",
        alpha=0.9,
        label="noise, one in twenty",
    )
    for c in clusters:
        ax.scatter(
            [c["center"][0]],
            [c["center"][1]],
            s=180,
            marker="o",
            facecolor="none",
            edgecolor=ORANGE,
            lw=2.6,
        )
    ax.scatter(
        [],
        [],
        s=180,
        marker="o",
        facecolor="none",
        edgecolor=ORANGE,
        lw=2.6,
        label="centres of the clusters found",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", facecolor=PANEL, edgecolor=GRID, labelcolor=INK)
    ax.grid(True)
    fig.savefig(HERE / "denstream_clusters.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print(
        "denstream:",
        len(clusters),
        [[round(v, 1) for v in c["center"]] for c in clusters],
    )


def vocabulary():
    vert = np.zeros((32, 32), dtype=np.float32)
    vert[:, ::4] = 1.0
    horz = vert.T.copy()
    bow = BagOfVisualWords(n_words=4, patch_size=8, n_bins=4)
    for _ in range(10):
        bow.learn_one({"frame": vert})
        bow.learn_one({"frame": horz})
    hv = bow.transform_one({"frame": vert})["hist"]
    hh = bow.transform_one({"frame": horz})["hist"]
    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1, 1, 1.6]}
    )
    fig.subplots_adjust(top=0.76, bottom=0.14, left=0.04, right=0.985, wspace=0.18)
    heading(
        fig,
        "TWO TEXTURES, TWO SIGNATURES",
        "each frame is cut into 8 x 8 patches, the patches are grouped into 4 words, a frame becomes a histogram of words",
    )
    for ax, img, name in zip(
        axes[:2], (vert, horz), ("vertical stripes", "horizontal stripes")
    ):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(name, color=INK, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    w = np.arange(4)
    axes[2].bar(w - 0.2, hv, 0.4, color=CYAN, label="vertical stripes")
    axes[2].bar(w + 0.2, hh, 0.4, color=ORANGE, label="horizontal stripes")
    axes[2].set_xticks(w)
    axes[2].set_xlabel("visual word")
    axes[2].set_ylabel("share of the patches of the frame")
    axes[2].legend(facecolor=PANEL, edgecolor=GRID, labelcolor=INK)
    axes[2].grid(True, axis="y")
    fig.savefig(HERE / "vocabulary_hist.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print("vocabulary:", hv, hh)


if __name__ == "__main__":
    customers()
    follow()
    denstream()
    vocabulary()
