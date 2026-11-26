"""The interactive map of the Factorization machines section, with the exact names of `usable`.

Draws docs/assets/maps/facto.svg with the same drawing as the other section maps (make_tree_map.py).

Run from the repo root:
    python docs/assets/maps/make_facto_map.py
"""

from make_tree_map import CYAN, HERE, ORANGE, PINK, draw

PAGES = [
    ("Factorization machine", "a short vector per feature", "facto/fm.md", CYAN, ["FMClassifier", "FMRegressor"]),
    ("Field-aware", "a vector per field", "facto/ffm.md", ORANGE, ["FFMClassifier", "FFMRegressor"]),
    ("Field-weighted", "a weight per pair of fields", "facto/fwfm.md", PINK, ["FwFMClassifier", "FwFMRegressor"]),
    ("Higher-order", "three features at once", "facto/hofm.md", CYAN, ["HOFMClassifier", "HOFMRegressor"]),
]


def main():
    draw(
        PAGES,
        ("facto",),
        "THE MAP OF FACTORIZATION MACHINES",
        "the job, the page of each model and its classes; pass over a page to light its classes, click to open it",
        HERE / "facto.svg",
        "The map of factorization machines",
        {"Classifier": "C", "Regressor": "R"},
        "",
        "docs/assets/maps/make_facto_map.py",
        partial=True,
    )


if __name__ == "__main__":
    main()
