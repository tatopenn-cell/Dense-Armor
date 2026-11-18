"""Maps of the main sections of the docs, drawn from the catalog, the nav and the page tables.

python docs/assets/maps/make_area_maps.py            # all the maps
python docs/assets/maps/make_area_maps.py ml audio   # some of them
"""

import sys
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from matplotlib.patches import FancyBboxPatch

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
HERE = Path(__file__).resolve().parent
DOCS = HERE.parent.parent
REPO = DOCS.parent


def plural(n, word="class"):
    return (
        f"{n} {word}"
        if n == 1
        else f"{n} {word}es"
        if word.endswith("s")
        else f"{n} {word}s"
    )


def render(path, title, subtitle, bands, footer):
    """bands: (number, title, note, color, [(name, line1, line2), ...]) with at most nine cards per band."""
    nb = len(bands)
    fig = plt.figure(figsize=(13.5, 1.75 + 2.15 * nb), facecolor=BG)
    fig.text(
        0.5,
        1 - 0.55 / fig.get_figheight(),
        title,
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(
        0.5,
        1 - 1.05 / fig.get_figheight(),
        subtitle,
        ha="center",
        va="center",
        fontsize=10.5,
        color=CYAN,
    )
    ax = fig.add_axes(
        [
            0.0,
            0.4 / fig.get_figheight(),
            1.0,
            (fig.get_figheight() - 1.45 - 0.4) / fig.get_figheight(),
        ],
        facecolor=BG,
    )
    H = 22.0 * nb
    ax.set_xlim(0, 135)
    ax.set_ylim(0, H + 1.5)
    ax.axis("off")
    for k, (num, name, note, color, cards) in enumerate(bands):
        y = H - 22.0 * (k + 1) + 2.0
        ax.add_patch(
            FancyBboxPatch(
                (1, y),
                133,
                19,
                boxstyle="round,pad=0.2,rounding_size=1.4",
                fc=BAND,
                ec=LINE,
                lw=1.2,
            )
        )
        ax.text(
            2.4, y + 16.8, num, color=PINK, fontsize=11, fontweight="bold", va="center"
        )
        ax.text(
            6.0,
            y + 16.8,
            name,
            color="#ffffff",
            fontsize=10.5,
            fontweight="bold",
            va="center",
        )
        ax.text(
            132.6, y + 16.8, note, color=MUTED, fontsize=8.2, ha="right", va="center"
        )
        n = len(cards)
        gap = 1.2
        w = min(31.0, (130.2 - (n - 1) * gap) / n)
        chars = max(10, int(w / 0.8))
        for i, (cname, line1, line2) in enumerate(cards):
            x = 2.4 + i * (w + gap)
            ax.add_patch(
                FancyBboxPatch(
                    (x, y + 1.4),
                    w,
                    13.2,
                    boxstyle="round,pad=0.2,rounding_size=1.0",
                    fc=CARD,
                    ec=color,
                    lw=1.5,
                )
            )
            lab = textwrap.fill(cname, chars)
            two = "\n" in lab
            ax.text(
                x + 0.9,
                y + (11.6 if two else 12.2),
                lab,
                color=TEXT,
                fontsize=8.8,
                fontweight="bold",
                va="center",
                linespacing=1.15,
            )
            ax.text(
                x + 0.9,
                y + 8.0,
                line1,
                color=color,
                fontsize=8.6 if len(line1) <= 15 else 7.2,
                fontweight="bold",
                va="center",
            )
            if line2:
                ax.text(
                    x + 0.9,
                    y + 4.6,
                    textwrap.fill(line2, int(w / 0.56)),
                    color=MUTED,
                    fontsize=7.2,
                    va="center",
                    linespacing=1.2,
                )
    fig.text(
        0.5, 0.18 / fig.get_figheight(), footer, ha="center", fontsize=8.6, color=MUTED
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=BG)
    plt.close(fig)
    print("wrote", path.relative_to(REPO))


def catalog():
    return get_catalog(REPO)


def by_module(items, prefix, depth):
    c = Counter()
    for i in items:
        if i["module"].startswith(prefix):
            c[i["module"].split(".")[depth]] += 1
    return c


def nav_pages(tab):
    class L(yaml.SafeLoader):
        pass

    L.add_multi_constructor(
        "tag:yaml.org,2002:python/", lambda loader, suffix, node: None
    )
    L.add_multi_constructor("!", lambda loader, suffix, node: None)
    cfg = yaml.load((REPO / "mkdocs.yml").read_text(encoding="utf-8"), Loader=L)

    def count(x):
        if isinstance(x, str):
            return 1
        if isinstance(x, list):
            return sum(count(i) for i in x)
        if isinstance(x, dict):
            return sum(count(v) for v in x.values())
        return 0

    out = {}
    for entry in cfg["nav"]:
        if isinstance(entry, dict) and tab in entry:
            for item in entry[tab]:
                if isinstance(item, dict):
                    for k, v in item.items():
                        out[k] = count(v)
    return out


def nav_files(tab):
    """The first file of every item of a tab, by item name."""

    class L(yaml.SafeLoader):
        pass

    L.add_multi_constructor(
        "tag:yaml.org,2002:python/", lambda loader, suffix, node: None
    )
    L.add_multi_constructor("!", lambda loader, suffix, node: None)
    cfg = yaml.load((REPO / "mkdocs.yml").read_text(encoding="utf-8"), Loader=L)

    def first(x):
        if isinstance(x, str):
            return x
        if isinstance(x, list):
            return first(x[0])
        if isinstance(x, dict):
            return first(next(iter(x.values())))
        return ""

    out = {}
    for entry in cfg["nav"]:
        if isinstance(entry, dict) and tab in entry:
            for item in entry[tab]:
                if isinstance(item, dict):
                    for k, v in item.items():
                        out[k] = first(v)
    return out


def ml_map():
    c = by_module(catalog(), "dense_armor.ml.", 2)
    groups = (
        (
            "01",
            "PREPARE",
            "summarise and rescale a table",
            CYAN,
            (
                (
                    "stats",
                    "Statistics and sketches",
                    "a mean, a variance or a correlation on a whole table",
                ),
                (
                    "preprocessing",
                    "Preprocessing",
                    "rescale, drop columns, turn text into numbers",
                ),
            ),
        ),
        (
            "02",
            "PREDICT",
            "a label or a number from the rows of a table",
            CYAN,
            (
                ("linear", "Linear models", "a weighted sum of the features"),
                ("bayes", "Naive Bayes", "counts turned into probabilities"),
                ("neighbors", "Nearest neighbours", "the rows that are closest"),
                ("multiclass", "Multiclass", "many binary classifiers combined"),
                ("multioutput", "Multi-output", "several targets at once"),
                ("tree", "Decision trees", "yes/no questions ending in a leaf"),
                ("forest", "Forests", "many trees, averaged"),
                ("ensemble", "Ensembles", "bagging, boosting, voting, stacking"),
            ),
        ),
        (
            "03",
            "DESCRIBE AND DETECT",
            "what a sample looks like, and what is wrong with it",
            PINK,
            (
                ("proba", "Distributions", "Beta, Gaussian, Multinomial posteriors"),
                ("anomaly", "Anomaly detection", "values unlike their neighbours"),
                ("drift", "Change detection", "where the level of a series shifts"),
                ("vision", "Vision", "frames and tables of features"),
            ),
        ),
    )
    placed = {a for _, _, _, _, cards in groups for a, _, _ in cards}
    assert placed | {"base"} == set(c), (
        sorted(set(c) - placed - {"base"}),
        sorted(placed - set(c)),
    )
    bands = [
        (n, t, note, col, [(name, plural(c[a]), line2) for a, name, line2 in cards])
        for n, t, note, col, cards in groups
    ]
    total = sum(c.values())
    render(
        DOCS / "assets" / "ml" / "map.png",
        "THE MAP OF MACHINE LEARNING",
        f"{len(placed)} families that read a whole table with fit and predict; {total} classes in the catalog",
        bands,
        "counts: classes in the catalog of the library (dense_armor.usable.catalog), drawn by docs/assets/maps/make_area_maps.py",
    )


def neural_map():
    c = by_module(catalog(), "dense_armor.core.deep.neural.", 4)
    pages = nav_pages("Deep learning")
    groups = (
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
    named = [p for _, _, _, _, ps in groups for p in ps]
    assert sorted(named) == sorted(pages), (
        sorted(set(pages) - set(named)),
        sorted(set(named) - set(pages)),
    )
    files = nav_files("Deep learning")
    bands = [
        (
            n,
            t,
            note,
            col,
            [(p, files[p].removesuffix(".md").removeprefix("neural/"), "") for p in ps],
        )
        for n, t, note, col, ps in groups
    ]
    parts = sorted(c.items(), key=lambda kv: -kv[1])
    code = "   ".join(f"{k} {v}" for k, v in parts)
    render(
        DOCS / "assets" / "neural" / "map.png",
        "THE MAP OF DEEP LEARNING",
        f"{len(pages)} pages in three groups; the code is {sum(c.values())} classes in eight sections",
        bands,
        f"in the code: {code}   (classes in the catalog), drawn by docs/assets/maps/make_area_maps.py",
    )


def audio_map():
    n_classes = sum(by_module(catalog(), "dense_armor.core.audio.", 3).values())
    groups = (
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
                    "read a file or a live input and find the moment something happens",
                ),
                (
                    "Sources and checks",
                    "sources",
                    "which paper each primitive comes from, and what was checked",
                ),
            ),
        ),
    )
    text = (DOCS / "audio" / "index.md").read_text(encoding="utf-8")
    for _, _, _, _, cards in groups:
        for _, page, _ in cards:
            assert f"({page}.md)" in text, page
    bands = [
        (
            n,
            t,
            note,
            col,
            [(name, f"audio/{page}", line2) for name, page, line2 in cards],
        )
        for n, t, note, col, cards in groups
    ]
    render(
        DOCS / "assets" / "audio" / "map.png",
        "THE MAP OF AUDIO",
        f"twelve pages in four groups, {n_classes} classes in the catalog; every primitive works on any signal",
        bands,
        "page names and descriptions are those of the table of docs/audio/index.md; drawn by docs/assets/maps/make_area_maps.py",
    )


def applications_map():
    pages = nav_pages("Applications")
    scripts = {
        d.name: sum(1 for f in d.glob("*.py") if f.name != "__init__.py")
        for d in (REPO / "pipelines").iterdir()
        if d.is_dir() and not d.name.startswith("_")
    }
    assert set(pages) == {"Finance", "Drones", "IoT", "Biomedical", "ARC"}, pages
    bands = [
        (
            "01",
            "APPLICATIONS",
            "worked pipelines on a stream, with the numbers they give",
            ORANGE,
            [
                (
                    "Finance",
                    f"{pages['Finance']} pages",
                    f"risk, execution and fraud; {scripts['finance']} scripts in pipelines/finance",
                ),
                (
                    "Drones",
                    f"{pages['Drones']} pages",
                    f"faults, battery, safety, landing; {scripts['drones']} scripts in pipelines/drones",
                ),
                ("IoT", f"{pages['IoT']} page", "industrial telemetry guard"),
                (
                    "Biomedical",
                    f"{pages['Biomedical']} page",
                    "ECG lead guard (a demonstration, not a device)",
                ),
                (
                    "ARC-AGI",
                    f"{pages['ARC']} pages",
                    "grids, objects and programs for the benchmark",
                ),
            ],
        )
    ]
    render(
        DOCS / "assets" / "applications" / "map.png",
        "THE MAP OF THE APPLICATIONS",
        f"{sum(pages.values())} pages in five applications; each one runs a pipeline of the library on a stream",
        bands,
        "page counts come from the nav of mkdocs.yml and script counts from the folder pipelines/; drawn by docs/assets/maps/make_area_maps.py",
    )


MAPS = {
    "ml": ml_map,
    "neural": neural_map,
    "audio": audio_map,
    "applications": applications_map,
}

if __name__ == "__main__":
    for name in sys.argv[1:] or MAPS:
        MAPS[name]()
