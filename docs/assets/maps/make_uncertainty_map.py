"""The interactive map of the Uncertainty pages, with the exact names of `usable`.

Draws docs/assets/maps/uncertainty.svg with the same drawing as the other section maps (make_tree_map.py).

Run from the repo root:
    python docs/assets/maps/make_uncertainty_map.py
"""

import sys

sys.path.insert(0, ".")

from make_tree_map import CYAN, HERE, ORANGE, PINK, draw

PAGES = [
    (
        "Interval coverage",
        "is the promise kept?",
        "metrics/uncertainty/coverage.md",
        CYAN,
        ["IntervalCoverage"],
    ),
    (
        "Interval width",
        "how much it costs",
        "metrics/uncertainty/width.md",
        ORANGE,
        ["MeanIntervalWidth"],
    ),
    (
        "Gaussian NLL",
        "a mean and a variance",
        "metrics/uncertainty/gaussian_nll.md",
        PINK,
        ["GaussianNLL"],
    ),
]


def main():
    draw(
        PAGES,
        ("metrics",),
        "THE MAP OF UNCERTAINTY",
        "the job, the page of each metric and its classes; pass over a page to light its classes, click to open it",
        HERE / "uncertainty.svg",
        "The map of uncertainty",
        {},
        "",
        "docs/assets/maps/make_uncertainty_map.py",
        partial=True,
    )


if __name__ == "__main__":
    main()
