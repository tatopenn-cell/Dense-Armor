"""Maps of the main sections of the docs, drawn as trees.

The names of the areas and of the classes are those of the catalog of the library
(``dense_armor.usable.catalog``); the names of the pages are those of the nav of
``mkdocs.yml``. Only the four or five groups of each map are chosen by hand.

    python docs/assets/maps/make_area_maps.py            # all the maps
    python docs/assets/maps/make_area_maps.py ml audio   # some of them
"""

import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MPath

from dense_armor.usable.catalog import get_catalog

BG = "#0d131a"
CARD = "#101a24"
CYAN = "#4cc9f0"
ORANGE = "#ff9e00"
PINK = "#f72585"
TEXT = "#e0fbfc"
MUTED = "#94a3b8"
HERE = Path(__file__).resolve().parent
DOCS = HERE.parent.parent
REPO = DOCS.parent
FOOTER = "drawn by docs/assets/maps/make_area_maps.py"


def plural(n, word="class"):
    return (
        f"{n} {word}"
        if n == 1
        else f"{n} {word}es"
        if word.endswith("s")
        else f"{n} {word}s"
    )


def edge(ax, p, q, color, lw=1.7, alpha=0.65):
    dx = (q[0] - p[0]) * 0.5
    path = MPath(
        [p, (p[0] + dx, p[1]), (q[0] - dx, q[1]), q],
        [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4],
    )
    ax.add_patch(PathPatch(path, fc="none", ec=color, lw=lw, alpha=alpha, zorder=1))
    for pt in (p, q):
        ax.add_patch(Circle(pt, 0.55, fc=color, ec=BG, lw=1.0, zorder=3))


def tree(path, title, subtitle, root, groups, footer):
    """root: (name, line1, line2). groups: (number, title, note, color, leaves).
    leaves: (name, desc, value, right_text); value is a number for a bar or None."""
    lh = [[5.6 if desc else 4.4 for _, desc, _, _ in leaves] for *_, leaves in groups]
    gap = 3.2
    H = sum(sum(r) for r in lh) + gap * (len(groups) - 1) + 2.0
    top_in, bot_in = 1.35, 0.55
    fig_h = top_in + bot_in + H * 0.1
    fig = plt.figure(figsize=(10.5, fig_h), facecolor=BG)
    fig.text(
        0.5,
        1 - 0.5 / fig_h,
        title,
        ha="center",
        va="center",
        fontsize=19,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(
        0.5,
        1 - 1.0 / fig_h,
        subtitle,
        ha="center",
        va="center",
        fontsize=9.6,
        color=CYAN,
    )
    ax = fig.add_axes([0, bot_in / fig_h, 1, H * 0.1 / fig_h], facecolor=BG)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, H)
    ax.axis("off")
    vmax = max(
        (v for *_, leaves in groups for _, _, v, _ in leaves if v is not None),
        default=1,
    )
    y = H - 1.0
    gcenters = []
    for (num, gname, note, color, leaves), rows in zip(groups, lh, strict=True):
        centers = []
        for (name, desc, value, right), h in zip(leaves, rows, strict=True):
            cy = y - h / 2
            ax.add_patch(
                FancyBboxPatch(
                    (59, cy - (h - 0.8) / 2),
                    45,
                    h - 0.8,
                    boxstyle="round,pad=0.15,rounding_size=0.9",
                    fc=CARD,
                    ec=color,
                    lw=1.3,
                    zorder=2,
                )
            )
            ax.text(
                60.2,
                cy + (1.05 if desc else 0),
                name,
                color=TEXT,
                fontsize=9,
                fontweight="bold",
                va="center",
                zorder=4,
            )
            if desc:
                ax.text(
                    60.2,
                    cy - 1.3,
                    desc,
                    color=MUTED,
                    fontsize=6.6,
                    va="center",
                    zorder=4,
                )
            if value is not None:
                ax.add_patch(
                    Rectangle(
                        (82, cy + (1.05 if desc else 0) - 0.5),
                        10.0 * value / vmax,
                        1.0,
                        fc=color,
                        ec="none",
                        alpha=0.75,
                        zorder=4,
                    )
                )
                ax.text(
                    103.4,
                    cy + (1.05 if desc else 0),
                    right,
                    color=color,
                    fontsize=8.8,
                    fontweight="bold",
                    ha="right",
                    va="center",
                    zorder=4,
                )
            elif right:
                ax.text(
                    103.4,
                    cy + (1.05 if desc else 0),
                    right,
                    color=color,
                    fontsize=7.6,
                    family="monospace",
                    ha="right",
                    va="center",
                    zorder=4,
                )
            centers.append(cy)
            y -= h
        gy = sum(centers) / len(centers)
        gcenters.append(gy)
        ax.add_patch(
            FancyBboxPatch(
                (28, gy - 4.0),
                24,
                8.0,
                boxstyle="round,pad=0.15,rounding_size=1.0",
                fc=CARD,
                ec=color,
                lw=1.7,
                zorder=2,
            )
        )
        ax.text(
            29.2,
            gy + 2.2,
            num,
            color=PINK,
            fontsize=9.5,
            fontweight="bold",
            va="center",
            zorder=4,
        )
        ax.text(
            32.6,
            gy + 2.2,
            gname,
            color="#ffffff",
            fontsize=9.2,
            fontweight="bold",
            va="center",
            zorder=4,
        )
        ax.text(
            29.2,
            gy - 1.2,
            textwrap.fill(note, 34),
            color=MUTED,
            fontsize=7.0,
            va="center",
            linespacing=1.15,
            zorder=4,
        )
        for cy in centers:
            edge(ax, (52.15, gy), (58.85, cy), color)
        y -= gap
    ry = sum(gcenters) / len(gcenters)
    ax.add_patch(
        FancyBboxPatch(
            (1, ry - 6.0),
            21,
            12.0,
            boxstyle="round,pad=0.15,rounding_size=1.2",
            fc=CARD,
            ec=CYAN,
            lw=2.0,
            zorder=2,
        )
    )
    ax.text(
        2.4,
        ry + 3.2,
        root[0],
        color=TEXT,
        fontsize=11.5,
        fontweight="bold",
        va="center",
        zorder=4,
    )
    ax.text(
        2.4,
        ry - 0.2,
        root[1],
        color=CYAN,
        fontsize=8.4,
        fontweight="bold",
        va="center",
        zorder=4,
    )
    ax.text(2.4, ry - 3.0, root[2], color=MUTED, fontsize=7.4, va="center", zorder=4)
    for gy in gcenters:
        edge(ax, (22.15, ry), (27.85, gy), MUTED, lw=2.2, alpha=0.75)
    fig.text(
        0.5,
        0.22 / fig_h,
        footer + "; " + FOOTER,
        ha="center",
        fontsize=7.6,
        color=MUTED,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=BG)
    plt.close(fig)
    print("wrote", path.relative_to(REPO))


def classes_of(prefix, depth):
    """The area -> exact class names of the catalog, for the modules under a prefix."""
    out = defaultdict(list)
    for i in get_catalog(REPO):
        if i["module"].startswith(prefix):
            out[i["module"].split(".")[depth]].append(i["class"])
    return out


def sample(names, limit=56):
    """The first exact class names that fit a line, and how many are left."""
    shown, used = [], 0
    for n in names:
        if used + len(n) + 2 > limit and shown:
            break
        shown.append(n)
        used += len(n) + 2
    rest = len(names) - len(shown)
    return ", ".join(shown) + (f"  +{rest}" if rest else "")


def _nav_cfg():
    class L(yaml.SafeLoader):
        pass

    L.add_multi_constructor(
        "tag:yaml.org,2002:python/", lambda loader, suffix, node: None
    )
    L.add_multi_constructor("!", lambda loader, suffix, node: None)
    return yaml.load((REPO / "mkdocs.yml").read_text(encoding="utf-8"), Loader=L)["nav"]


def _tab_items(tab):
    for e in _nav_cfg():
        if isinstance(e, dict) and tab in e:
            for it in e[tab]:
                if isinstance(it, dict):
                    yield from ((k, v) for k, v in it.items() if k != "Overview")


def nav_pages(tab):
    def count(x):
        if isinstance(x, str):
            return 1
        if isinstance(x, list):
            return sum(count(i) for i in x)
        if isinstance(x, dict):
            return sum(count(v) for v in x.values())
        return 0

    return {k: count(v) for k, v in _tab_items(tab)}


def nav_files(tab):
    def first(x):
        if isinstance(x, str):
            return x
        if isinstance(x, list):
            return first(x[0])
        if isinstance(x, dict):
            return first(next(iter(x.values())))
        return ""

    return {k: first(v) for k, v in _tab_items(tab)}


def leaves(c, areas):
    return [(a, sample(c[a]), len(c[a]), plural(len(c[a]))) for a in areas]


def online_map():
    c = classes_of("dense_armor.core.online.", 3)
    spec = (
        (
            "01",
            "PREDICT",
            "a label or a number from the features of a sample",
            CYAN,
            (
                "learn",
                "tree",
                "forest",
                "ensemble",
                "linear",
                "bayes",
                "neighbors",
                "multiclass",
                "multioutput",
            ),
        ),
        (
            "02",
            "DETECT",
            "what is strange, what has changed, what to keep out of the model",
            PINK,
            ("anomaly", "drift", "protect"),
        ),
        (
            "03",
            "DECIDE AND SUGGEST",
            "choose an action, ask for a label, recommend an item",
            ORANGE,
            ("bandit", "active", "reco", "facto"),
        ),
        (
            "04",
            "MODEL THE STREAM",
            "time, distributions, the optimizer, the groups and the rules",
            CYAN,
            ("timeseries", "proba", "optim", "cluster", "rules"),
        ),
    )
    placed = sorted(a for *_, areas in spec for a in areas)
    assert placed == sorted(c), (
        sorted(set(c) - set(placed)),
        sorted(set(placed) - set(c)),
    )
    groups = [(n, t, note, col, leaves(c, areas)) for n, t, note, col, areas in spec]
    total = sum(len(v) for v in c.values())
    tree(
        DOCS / "assets" / "online" / "overview.png",
        "THE MAP OF ONLINE LEARNING",
        f"{len(c)} areas in four groups; the names are those of the catalog, the bar is the number of classes",
        ("online", f"{len(c)} areas", f"{total} classes in the catalog"),
        groups,
        "areas and classes: dense_armor.usable.catalog",
    )


def ml_map():
    c = classes_of("dense_armor.ml.", 2)
    spec = (
        (
            "01",
            "PREPARE",
            "summarise and rescale a table",
            CYAN,
            ("stats", "preprocessing"),
        ),
        (
            "02",
            "PREDICT",
            "a label or a number from the rows of a table",
            CYAN,
            (
                "linear",
                "bayes",
                "neighbors",
                "multiclass",
                "multioutput",
                "tree",
                "forest",
                "ensemble",
            ),
        ),
        (
            "03",
            "DESCRIBE AND DETECT",
            "what a sample looks like, and what is wrong with it",
            PINK,
            ("proba", "anomaly", "drift", "vision"),
        ),
    )
    placed = {a for *_, areas in spec for a in areas}
    assert placed | {"base"} == set(c), (
        sorted(set(c) - placed - {"base"}),
        sorted(placed - set(c)),
    )
    groups = [(n, t, note, col, leaves(c, areas)) for n, t, note, col, areas in spec]
    tree(
        DOCS / "assets" / "ml" / "map.png",
        "THE MAP OF MACHINE LEARNING",
        "fourteen families that read a whole table; the names are those of the catalog",
        (
            "ml",
            f"{len(placed)} families",
            f"{sum(len(v) for v in c.values())} classes in the catalog",
        ),
        groups,
        "families and classes: dense_armor.usable.catalog",
    )


def neural_map():
    c = classes_of("dense_armor.core.deep.neural.", 4)
    pages = nav_pages("Deep learning")
    files = nav_files("Deep learning")
    spec = (
        (
            "01",
            "BUILD THE NETWORK",
            "bricks, layers, chains and attention",
            CYAN,
            (
                "Foundations",
                "Layers",
                "Chains",
                "Convolution and pooling",
                "Recurrent layers and attention",
                "Attention",
                "Composable attention",
                "Modern blocks",
                "Deep",
            ),
        ),
        (
            "02",
            "TRAIN IT",
            "losses, optimisers, speed and memory",
            ORANGE,
            (
                "Training",
                "Compiled training",
                "Mixed precision",
                "Data parallelism",
                "Memory and data",
                "Continuous training",
                "Continual learning",
                "Stable variance in float32",
            ),
        ),
        (
            "03",
            "LANGUAGE MODELS",
            "encoders, decoders and generation",
            PINK,
            (
                "Encoders and decoders",
                "Decoder language models",
                "Language-model training",
                "Rotary decoder and generation cache",
                "Mixture of experts and test-time compute",
            ),
        ),
    )
    named = [p for *_, ps in spec for p in ps]
    assert sorted(named) == sorted(pages), (
        sorted(set(pages) - set(named)),
        sorted(set(named) - set(pages)),
    )
    groups = [
        (
            n,
            t,
            note,
            col,
            [
                (p, "", None, files[p].removesuffix(".md").removeprefix("neural/"))
                for p in ps
            ],
        )
        for n, t, note, col, ps in spec
    ]
    code = "  ".join(
        f"{k} {len(v)}" for k, v in sorted(c.items(), key=lambda kv: -len(kv[1]))
    )
    tree(
        DOCS / "assets" / "neural" / "map.png",
        "THE MAP OF DEEP LEARNING",
        f"{len(pages)} pages in three groups; the page names are those of the menu",
        (
            "deep learning",
            f"{len(pages)} pages",
            f"{sum(len(v) for v in c.values())} classes in 8 sections",
        ),
        groups,
        f"classes per section of the code: {code}",
    )


def audio_map():
    n_classes = sum(len(v) for v in classes_of("dense_armor.core.audio.", 3).values())
    spec = (
        (
            "01",
            "HOLD AND LOOK",
            "a sound, as numbers and as a listener hears it",
            CYAN,
            (
                (
                    "Sound, files and frequency",
                    "sound",
                    "hold a sound, read and write WAV, look at it by frequency",
                ),
                (
                    "Perceptual analysis",
                    "perceptual",
                    "bands, cepstrum, brightness, onsets, envelope",
                ),
                ("Pitch", "pitch", "find the pitch of a block, and name a frequency"),
            ),
        ),
        (
            "02",
            "SHAPE",
            "filters and effects, fixed or trained",
            ORANGE,
            (
                ("Filters", "filters", "filter a sound by biquads, taps or poles"),
                ("Effects", "effects", "how loud, how clean, how roomy or how moving"),
                (
                    "Filters you can train",
                    "differentiable",
                    "fit the numbers of a filter by gradient descent",
                ),
            ),
        ),
        (
            "03",
            "MAKE AND CHANGE",
            "make a sound, or change its length and pitch",
            PINK,
            (
                (
                    "Making sound",
                    "synthesis",
                    "make a sound from nothing, without aliasing",
                ),
                (
                    "Resampling",
                    "resampling",
                    "change the sample rate, or glide the pitch",
                ),
                ("Time and pitch", "timescale", "longer or shorter, higher or lower"),
            ),
        ),
        (
            "04",
            "FIND AND SPLIT",
            "events, live input and sources",
            CYAN,
            (
                ("Separating sources", "analysis", "split a sound into parts"),
                (
                    "Events and streams",
                    "events",
                    "read a file or a live input, find the moment something happens",
                ),
                (
                    "Sources and checks",
                    "sources",
                    "which paper each primitive comes from, what was checked",
                ),
            ),
        ),
    )
    text = (DOCS / "audio" / "index.md").read_text(encoding="utf-8")
    for *_, cards in spec:
        for name, page, desc in cards:
            assert f"[{name}]({page}.md)" in text and desc.split(",")[0] in text, name
    groups = [
        (n, t, note, col, [(name, desc, None, page) for name, page, desc in cards])
        for n, t, note, col, cards in spec
    ]
    tree(
        DOCS / "assets" / "audio" / "map.png",
        "THE MAP OF AUDIO",
        f"twelve pages in four groups; {n_classes} classes in the catalog",
        ("audio", "12 pages", f"{n_classes} classes"),
        groups,
        "page names and descriptions are those of the table of docs/audio/index.md",
    )


def applications_map():
    pages = nav_pages("Applications")
    assert set(pages) == {"Finance", "Drones", "Robot", "IoT", "Biomedical", "ARC"}, (
        pages
    )
    scripts = {
        d.name: sum(1 for f in d.glob("*.py") if f.name != "__init__.py")
        for d in (REPO / "pipelines").iterdir()
        if d.is_dir() and not d.name.startswith("_")
    }
    robot_children = ""
    for entry in _nav_cfg():
        if isinstance(entry, dict) and "Applications" in entry:
            for it in entry["Applications"]:
                if isinstance(it, dict) and "Robot" in it:
                    robot_children = ", ".join(
                        k for child in it["Robot"] for k in child
                    )
    fin = classes_of("dense_armor.core.finance", 2)
    fin_names = sample([n for v in fin.values() for n in v])
    groups = [
        (
            "01",
            "PIPELINES ON A STREAM",
            "each page runs a pipeline of the library and prints what it finds",
            ORANGE,
            [
                (
                    "Finance",
                    f"{fin_names}; {scripts['finance']} scripts",
                    pages["Finance"],
                    plural(pages["Finance"], "page"),
                ),
                (
                    "Drones",
                    f"faults, battery, safety, landing; {scripts['drones']} scripts",
                    pages["Drones"],
                    plural(pages["Drones"], "page"),
                ),
                (
                    "Robot",
                    robot_children,
                    pages["Robot"],
                    plural(pages["Robot"], "page"),
                ),
                (
                    "IoT",
                    "industrial telemetry guard",
                    pages["IoT"],
                    plural(pages["IoT"], "page"),
                ),
                (
                    "Biomedical",
                    "ECG lead guard, a demonstration, not a device",
                    pages["Biomedical"],
                    plural(pages["Biomedical"], "page"),
                ),
            ],
        ),
        (
            "02",
            "A BENCHMARK",
            "tasks that are easy for a person and hard for a program",
            CYAN,
            [
                (
                    "ARC",
                    "grids, objects and programs",
                    pages["ARC"],
                    plural(pages["ARC"], "page"),
                )
            ],
        ),
    ]
    tree(
        DOCS / "assets" / "applications" / "map.png",
        "THE MAP OF THE APPLICATIONS",
        f"{sum(pages.values())} pages in six applications; the names are those of the menu, the bar is the number of pages",
        ("applications", "6 applications", f"{sum(pages.values())} pages"),
        groups,
        "page counts from the nav of mkdocs.yml, script counts from the folder pipelines/",
    )


MAPS = {
    "online": online_map,
    "ml": ml_map,
    "neural": neural_map,
    "audio": audio_map,
    "applications": applications_map,
}

if __name__ == "__main__":
    for name in sys.argv[1:] or MAPS:
        MAPS[name]()
