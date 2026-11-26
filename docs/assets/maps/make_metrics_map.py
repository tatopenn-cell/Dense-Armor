"""The interactive map of the Metrics section, with the exact names of `usable`.

Draws docs/assets/maps/metrics.svg with the same drawing as the other section maps (make_tree_map.py).

Run from the repo root:
    python docs/assets/maps/make_metrics_map.py
"""

import sys
sys.path.insert(0, ".")

from make_tree_map import CYAN, HERE, ORANGE, PINK, draw

PAGES = [
    (
        "Metrics: pairs",
        "one pair at a time",
        "metrics/basics.md",
        CYAN,
        ["Metric"],
    ),
    (
        "Classification",
        "precision, recall, F1",
        "metrics/classification/index.md",
        ORANGE,
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "FBeta",
            "BalancedAccuracy",
            "ConfusionMatrix",
            "CohenKappa",
        ],
    ),
    (
        "Probabilities",
        "log-loss, Brier score",
        "metrics/probabilities/index.md",
        PINK,
        ["LogLoss", "BrierScore"],
    ),
    (
        "Regression",
        "error, R2",
        "metrics/regression/index.md",
        CYAN,
        ["R2Score"],
    ),
    (
        "Uncertainty",
        "intervals",
        "metrics/uncertainty/index.md",
        CYAN,
        ["GaussianNLL", "IntervalCoverage", "MeanIntervalWidth"],
    ),
    (
        "Stream",
        "windows, rolling",
        "metrics/stream.md",
        ORANGE,
        ["Rolling", "RollingAUC"],
    ),
    (
        "Events",
        "delay, false alarms",
        "metrics/events.md",
        PINK,
        ["EventMetrics", "EventWindow", "PointAdjustedF1"],
    ),
]


def main():
    draw(
        PAGES,
        ("metrics",),
        "THE MAP OF METRICS",
        "the job, the page of each metric and its classes; pass over a page to light its classes, click to open it",
        HERE / "metrics.svg",
        "The map of metrics",
        {},
        "",
        "docs/assets/maps/make_metrics_map.py",
        partial=True,
    )


if __name__ == "__main__":
    main()