"""The interactive map of the robot part of the API: the pages of the Robot group of the menu and what is in each.

Draws docs/assets/maps/robot.svg with the same drawing as the other section maps (make_tree_map.py). The
modules of Control and of Robot dynamics are read from the folders ``core/robot/control`` and ``dynamics``; the
job is the ``robot`` job of the ontology and every other domain it reaches is a page of the map, with the number
of its classes in the catalog: a robot can take any of them, and none of them points back to the robot.

Run from the repo root:
    python docs/assets/maps/make_robot_map.py
"""

import json

from make_tree_map import CYAN, HERE, ORANGE, PINK, REPO, draw

from dense_armor.usable.catalog import get_catalog

PARTS = {
    "learn": ("Online learning", "learn/index.md"),
    "tree": ("Trees", "tree/index.md"),
    "forest": ("Forests", "forest/index.md"),
    "ensemble": ("Ensembles", "ml/ensemble.md"),
    "neural": ("Deep learning", "neural/overview.md"),
    "ml": ("Machine learning", "ml/index.md"),
    "timeseries": ("Time series", "timeseries/index.md"),
    "anomaly": ("Anomaly detection", "anomaly/index.md"),
    "drift": ("Drift detectors", "drift/detectors/index.md"),
    "cluster": ("Clustering", "cluster/index.md"),
    "vision": ("Video", "vision/index.md"),
    "audio": ("Audio", "audio/index.md"),
    "stats": ("Statistics", "stats/index.md"),
    "preprocessing": ("Preprocessing", "preprocessing/index.md"),
    "optim": ("Optimizers", "optim/index.md"),
    "graph": ("Ontology", "ontology/index.md"),
    "protect": ("Shields", "shields/engine.md"),
}


def short(names, limit=36):
    shown, used = [], 0
    for n in names:
        if used + len(n) + 2 > limit and shown:
            break
        shown.append(n)
        used += len(n) + 2
    rest = len(names) - len(shown)
    return ", ".join(shown) + (f"  +{rest}" if rest else "")


def modules(folder):
    return sorted(f.stem for f in folder.glob("*.py") if f.stem != "__init__")


def main():
    control = modules(REPO / "dense_armor" / "core" / "robot" / "control")
    dynamics_dir = REPO / "dense_armor" / "dynamics"
    dynamics = modules(dynamics_dir)
    classes = [
        i["class"]
        for i in get_catalog(REPO)
        if i.get("kind", "class") == "class"
        and i["module"].startswith("dense_armor.dynamics.")
    ]
    pages = [
        ("Control", "rate, barrier, path, controller", "control/index.md", CYAN, []),
        (
            "Robot dynamics",
            "a physical model from a URDF",
            "dynamics/index.md",
            ORANGE,
            classes,
        ),
        (
            "Benchmarks",
            "recordings of real robots",
            "benchmarks/index.md",
            PINK,
            [],
        ),
    ]
    onto = json.loads(
        (REPO / "dense_armor" / "usable" / "ontology.json").read_text(encoding="utf-8")
    )
    domains = [d for d in onto["jobs"]["robot"]["domains"] if d in PARTS]
    rows = [i for i in get_catalog(REPO) if i.get("kind", "class") == "class"]
    colours = [CYAN, ORANGE, PINK]
    extra = {}
    for k, d in enumerate(domains):
        title, rel = PARTS[d]
        names = [i["class"] for i in rows if i["domain"] == d]
        if title in extra or not names:
            continue
        pages.append((title, "", rel, colours[k % 3], []))
        extra[title] = f"{len(names)} classes: {short(names)}"
    summaries = {
        "Control": f"{len(control)} modules: {', '.join(control)}",
        "Robot dynamics": f"{len(dynamics)} modules: {', '.join(dynamics)}",
        "Benchmarks": "CASPER on the UR3e",
        **extra,
    }
    draw(
        pages,
        ("dynamics", *domains),
        "THE MAP OF THE ROBOT",
        "the job, the page of each part of the robot group and what it holds; pass over a page to light it, click to open it",
        HERE / "robot.svg",
        "The map of the robot",
        {},
        "",
        "docs/assets/maps/make_robot_map.py",
        partial=True,
        jobs=["robot"],
        hub="robot",
        summaries=summaries,
    )


if __name__ == "__main__":
    main()
