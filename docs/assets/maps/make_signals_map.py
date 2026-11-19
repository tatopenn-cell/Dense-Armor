"""The interactive map of the Signals section: the folders of ``core/signals`` and what is in each.

Draws docs/assets/maps/signals.svg with the same drawing as the other section maps (make_tree_map.py). The
classes of a folder are listed on one line (there are 82 in all); the jobs are those of the ontology that
reach the domains of the folders.

Run from the repo root:
    python docs/assets/maps/make_signals_map.py
"""

import json

from make_tree_map import CYAN, HERE, ORANGE, PINK, REPO, draw

from dense_armor.usable.catalog import get_catalog

SIGNALS = REPO / "dense_armor" / "core" / "signals"
FOLDERS = [
    ("Streams", "files, bags and arrays as streams", "stream/index.md", CYAN, "stream"),
    (
        "Datasets",
        "streams with a known change",
        "stream/datasets.md",
        ORANGE,
        "datasets",
    ),
    (
        "Statistics",
        "running moments, quantiles, sketches",
        "stats/index.md",
        PINK,
        "stats",
    ),
    (
        "Preprocessing",
        "scale, select, text, joints",
        "preprocessing/index.md",
        CYAN,
        "preprocessing",
    ),
    ("Vision", "frames, patches, motion", "vision/index.md", ORANGE, "vision"),
]


def short(names, limit=36):
    shown, used = [], 0
    for n in names:
        if used + len(n) + 2 > limit and shown:
            break
        shown.append(n)
        used += len(n) + 2
    rest = len(names) - len(shown)
    return ", ".join(shown) + (f"  +{rest}" if rest else "")


def main():
    catalog = [
        i
        for i in get_catalog(REPO)
        if i.get("kind", "class") == "class"
        and i["module"].startswith("dense_armor.core.signals.")
    ]
    pages, summaries, domains = [], {}, set()
    for title, sub, rel, col, folder in FOLDERS:
        classes = [i["class"] for i in catalog if i["module"].split(".")[3] == folder]
        domains |= {i["domain"] for i in catalog if i["module"].split(".")[3] == folder}
        if classes:
            summaries[title] = f"{len(classes)} classes: {short(classes)}"
        else:
            modules = sorted(
                f.stem for f in (SIGNALS / folder).glob("*.py") if f.stem != "__init__"
            )
            summaries[title] = f"{len(modules)} modules: {short(modules)}"
        pages.append((title, sub, rel, col, classes))
    ontology = json.loads(
        (REPO / "dense_armor" / "usable" / "ontology.json").read_text(encoding="utf-8")
    )
    jobs = [j for j, v in ontology["jobs"].items() if domains & set(v["domains"])]
    draw(
        pages,
        tuple(sorted(domains)),
        "THE MAP OF SIGNALS",
        "the job, the page of each folder of core/signals and what it holds; pass over a page to light it, click to open it",
        HERE / "signals.svg",
        "The map of signals",
        {"Classifier": "C", "Regressor": "R"},
        "C classifier, R regressor; ",
        "docs/assets/maps/make_signals_map.py",
        partial=True,
        jobs=jobs,
        hub="signals",
        summaries=summaries,
    )


if __name__ == "__main__":
    main()
