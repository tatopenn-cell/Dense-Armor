"""The interactive map of the Datasets section, with the exact names of `usable`.

Draws docs/assets/maps/datasets.svg with the same drawing as the other section maps (make_tree_map.py).

Run from the repo root:
    python docs/assets/maps/make_datasets_map.py
"""

from make_tree_map import CYAN, HERE, ORANGE, PINK, draw

PAGES = [
    (
        "DriftStream",
        "a concept that changes",
        "datasets/drift_stream.md",
        CYAN,
        ["DriftStream"],
    ),
    (
        "SyntheticArm",
        "a robot arm with a fault",
        "datasets/synthetic_arm.md",
        ORANGE,
        ["SyntheticArm"],
    ),
    ("Casper", "an industrial arm recording", "datasets/casper.md", PINK, ["Casper"]),
]


def main():
    draw(
        PAGES,
        ("datasets",),
        "THE MAP OF DATASETS",
        "the job, the page of each dataset and its class; pass over a page to light it, click to open it",
        HERE / "datasets.svg",
        "The map of datasets",
        {"Classifier": "C", "Regressor": "R"},
        "C classifier, R regressor; ",
        "docs/assets/maps/make_datasets_map.py",
    )


if __name__ == "__main__":
    main()
