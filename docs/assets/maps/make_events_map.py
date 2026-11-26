"""The interactive map of the Events pages, with the exact names of `usable`.

Draws docs/assets/maps/events.svg with the same drawing as the other section maps (make_tree_map.py).
The pages that show one method of EventMetrics, and the function evaluate_events, are drawn with a summary line
(``summaries``); EventMetrics, EventWindow and PointAdjustedF1 are checked against the catalog.

Run from the repo root:
    python docs/assets/maps/make_events_map.py
"""

import sys

sys.path.insert(0, ".")

from make_tree_map import CYAN, HERE, ORANGE, PINK, draw

PAGES = [
    (
        "Delay and false alarms",
        "how late, how often for nothing",
        "metrics/events/delays.md",
        CYAN,
        ["EventMetrics", "EventWindow"],
    ),
    (
        "Range precision and recall",
        "how much of the event",
        "metrics/events/range.md",
        ORANGE,
        ["range_precision, range_recall, range_f1"],
    ),
    (
        "The NAB score",
        "early and quiet",
        "metrics/events/nab.md",
        PINK,
        ["nab_score"],
    ),
    (
        "A detector on a stream",
        "all the numbers at once",
        "metrics/events/evaluate_events.md",
        CYAN,
        ["evaluate_events"],
    ),
    (
        "Point-adjusted F1",
        "a warning",
        "metrics/events/point_adjusted_f1.md",
        ORANGE,
        ["PointAdjustedF1"],
    ),
]

SUMMARIES = {
    "Range precision and recall": "EventMetrics: range_precision, range_recall, range_f1",
    "The NAB score": "EventMetrics: nab_score",
    "A detector on a stream": "evaluate_events",
}


def main():
    draw(
        PAGES,
        ("metrics",),
        "THE MAP OF EVENTS",
        "the job, the page of each tool and its classes; pass over a page to light its classes, click to open it",
        HERE / "events.svg",
        "The map of events",
        {},
        "",
        "docs/assets/maps/make_events_map.py",
        partial=True,
        summaries=SUMMARIES,
    )


if __name__ == "__main__":
    main()
