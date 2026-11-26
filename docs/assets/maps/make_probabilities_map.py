"""The interactive map of the Probabilities pages, with the exact names of `usable`.

Draws docs/assets/maps/probabilities.svg with the same drawing as the other section maps (make_tree_map.py).

Run from the repo root:
    python docs/assets/maps/make_probabilities_map.py
"""

import sys

sys.path.insert(0, ".")

from make_tree_map import CYAN, HERE, ORANGE, draw

PAGES = [
    (
        "Log loss",
        "punishes a sure mistake",
        "metrics/probabilities/log_loss.md",
        CYAN,
        ["LogLoss"],
    ),
    (
        "Brier score",
        "squared distance",
        "metrics/probabilities/brier_score.md",
        ORANGE,
        ["BrierScore"],
    ),
]


def main():
    draw(
        PAGES,
        ("metrics",),
        "THE MAP OF PROBABILITIES",
        "the job, the page of each metric and its classes; pass over a page to light its classes, click to open it",
        HERE / "probabilities.svg",
        "The map of probabilities",
        {},
        "",
        "docs/assets/maps/make_probabilities_map.py",
        partial=True,
    )


if __name__ == "__main__":
    main()
