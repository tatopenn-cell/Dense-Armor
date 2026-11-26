"""The interactive map of the Classification pages, with the exact names of `usable`.

Draws docs/assets/maps/classification.svg with the same drawing as the other section maps (make_tree_map.py).

Run from the repo root:
    python docs/assets/maps/make_classification_map.py
"""

import sys

sys.path.insert(0, ".")

from make_tree_map import CYAN, HERE, ORANGE, PINK, draw

PAGES = [
    (
        "Precision, recall and F1",
        "an alarm, right or missed",
        "metrics/classification/precision_recall_f1.md",
        CYAN,
        ["Accuracy", "Precision", "Recall", "F1", "FBeta"],
    ),
    (
        "Confusion matrix",
        "every kind of answer counted",
        "metrics/classification/confusion_matrix.md",
        ORANGE,
        ["ConfusionMatrix", "BalancedAccuracy"],
    ),
    (
        "Cohen's kappa",
        "agreement beyond chance",
        "metrics/classification/kappa.md",
        PINK,
        ["CohenKappa"],
    ),
]


def main():
    draw(
        PAGES,
        ("metrics",),
        "THE MAP OF CLASSIFICATION",
        "the job, the page of each metric and its classes; pass over a page to light its classes, click to open it",
        HERE / "classification.svg",
        "The map of classification",
        {},
        "",
        "docs/assets/maps/make_classification_map.py",
        partial=True,
    )


if __name__ == "__main__":
    main()
