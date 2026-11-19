"""The interactive map of the time series section, with the exact names of `usable`.

Draws docs/assets/maps/timeseries.svg for the first page of the section; it uses the same drawing
as the map of the trees (make_tree_map.py).

Run from the repo root:
    python docs/assets/maps/make_timeseries_map.py
"""

from make_tree_map import CYAN, HERE, ORANGE, PINK, draw

PAGES = [
    (
        "Autoregressive models",
        "the next value of a signal",
        "timeseries/ar.md",
        CYAN,
        ["OnlineAR", "OnlineVAR"],
    ),
    (
        "Kalman filters",
        "a state seen through noise",
        "timeseries/kalman.md",
        ORANGE,
        ["KalmanFilter", "VBAdaptiveKalman", "InnovationAdaptiveKalman"],
    ),
    (
        "Cycle residual",
        "when a rhythm breaks",
        "timeseries/cycle.md",
        PINK,
        ["CycleResidual"],
    ),
    (
        "GARCH",
        "when the chaos rises",
        "timeseries/garch.md",
        CYAN,
        ["GARCH11"],
    ),
]


def main():
    draw(
        PAGES,
        ("timeseries",),
        "THE MAP OF THE TIME SERIES",
        "the job, the page of each tool and its classes; pass over a page to light its classes, click to open it",
        HERE / "timeseries.svg",
        "The map of the time series",
        {},
        "",
        "docs/assets/maps/make_timeseries_map.py",
    )


if __name__ == "__main__":
    main()
