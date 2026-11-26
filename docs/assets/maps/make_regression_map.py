"""The interactive map of the Regression pages, with the exact names of `usable`.

Draws docs/assets/maps/regression.svg with the same drawing as the other section maps (make_tree_map.py).
MeanAbsoluteError, MeanSquaredError and RootMeanSquaredError are not in the catalog of `usable`, so their pages
are drawn with a summary line (``summaries``) and only R2Score is checked against the catalog.

Run from the repo root:
    python docs/assets/maps/make_regression_map.py
"""

import sys

sys.path.insert(0, ".")

from make_tree_map import CYAN, HERE, ORANGE, PINK, draw

PAGES = [
    (
        "Mean absolute error",
        "the average miss",
        "metrics/regression/mae.md",
        CYAN,
        ["MeanAbsoluteError"],
    ),
    (
        "MSE and RMSE",
        "large misses weigh more",
        "metrics/regression/mse_rmse.md",
        ORANGE,
        ["MeanSquaredError", "RootMeanSquaredError"],
    ),
    (
        "R squared",
        "better than the mean?",
        "metrics/regression/r2.md",
        PINK,
        ["R2Score"],
    ),
]

SUMMARIES = {
    "Mean absolute error": "MeanAbsoluteError",
    "MSE and RMSE": "MeanSquaredError, RootMeanSquaredError",
}


def main():
    draw(
        PAGES,
        ("metrics",),
        "THE MAP OF REGRESSION",
        "the job, the page of each metric and its classes; pass over a page to light its classes, click to open it",
        HERE / "regression.svg",
        "The map of regression",
        {},
        "",
        "docs/assets/maps/make_regression_map.py",
        partial=True,
        summaries=SUMMARIES,
    )


if __name__ == "__main__":
    main()
