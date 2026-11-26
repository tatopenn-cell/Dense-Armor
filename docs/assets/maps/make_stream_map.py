"""The interactive map of the "On a stream" pages, with the exact names of `usable`.

Draws docs/assets/maps/stream.svg with the same drawing as the other section maps (make_tree_map.py).
The functions of the evaluation module (progressive_val_score, best_of, ...) are not classes of the catalog of `usable`, so
their pages are drawn with a summary line (``summaries``); Rolling and RollingAUC are checked against the catalog.

Run from the repo root:
    python docs/assets/maps/make_stream_map.py
"""

import sys

sys.path.insert(0, ".")

from make_tree_map import CYAN, HERE, ORANGE, PINK, draw

PAGES = [
    (
        "Rolling window",
        "only the recent past",
        "metrics/stream/rolling.md",
        CYAN,
        ["Rolling"],
    ),
    (
        "Rolling AUC",
        "exact, on the window",
        "metrics/stream/rolling_auc.md",
        ORANGE,
        ["RollingAUC"],
    ),
    (
        "Predict, then learn",
        "score before learning",
        "metrics/stream/predict_then_learn.md",
        PINK,
        ["progressive_val_score", "progressive_val_proba_score"],
    ),
    (
        "Choosing a model",
        "who is ahead today",
        "metrics/stream/model_selection.md",
        CYAN,
        ["best_of", "OnlineModelSelection"],
    ),
]

SUMMARIES = {
    "Predict, then learn": "progressive_val_score, progressive_val_proba_score",
    "Choosing a model": "best_of, OnlineModelSelection",
}


def main():
    draw(
        PAGES,
        ("metrics",),
        "THE MAP OF THE STREAM PAGES",
        "the job, the page of each tool and its classes; pass over a page to light its classes, click to open it",
        HERE / "stream.svg",
        "The map of the stream pages",
        {},
        "",
        "docs/assets/maps/make_stream_map.py",
        partial=True,
        summaries=SUMMARIES,
    )


if __name__ == "__main__":
    main()
