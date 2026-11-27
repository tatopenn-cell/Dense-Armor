"""The interactive map of the Fit section, with the exact names of `usable`.

Draws docs/assets/maps/fit.svg with the same drawing as the other section maps (make_tree_map.py). The classes come from
several domains of `usable`, one for each family that the section shows. LEGO, POLA and OASIS are online metric learners: their unit is a pair or a triplet.

Run from the repo root:
    python docs/assets/maps/make_fit_map.py
"""

from make_tree_map import CYAN, HERE, ORANGE, PINK, draw

PAGES = [
    ("Fit on an estimator", "the same algorithm on the whole table", "fit/estimators.md", CYAN, ["LinearRegression", "StandardScaler"]),
    ("Whole-table algorithms (Machine learning)", "the ones that need all the rows", "ml/fit_batch.md", ORANGE, ["BatchDecisionTreeClassifier", "OnlineBaggingClassifier", "TextVectorizer"]),
    ("LEGO, POLA and OASIS", "online metric learning, a pair or a triplet at a time", "fit/metric_learners.md", PINK, ["LEGO", "POLA", "OASIS"]),
]


def main():
    draw(
        PAGES,
        ("linear", "preprocessing", "tree", "ensemble", "learn"),
        "THE MAP OF FIT",
        "the job, the page of each way to give data and its classes; pass over a page to light its classes, click to open it",
        HERE / "fit.svg",
        "The map of fit",
        {"Classifier": "C", "Regressor": "R"},
        "",
        "docs/assets/maps/make_fit_map.py",
        partial=True,
    )


if __name__ == "__main__":
    main()
