"""The interactive map of the drift detectors section, with the exact names of `usable`.

Draws docs/assets/maps/drift_detectors.svg with the same drawing as the other section maps (make_tree_map.py).

Run from the repo root:
    python docs/assets/maps/make_drift_detectors_map.py
"""

from make_tree_map import CYAN, HERE, ORANGE, PINK, draw

PAGES = [
    ("CUSUM", "its own section", "drift/cusum/index.md", CYAN, ["CUSUMDriftDetector"]),
    ("PageHinkley", "units do not matter", "drift/detectors/page_hinkley.md", ORANGE, ["PageHinkley"]),
    ("ADWIN", "a window that cuts itself", "drift/detectors/adwin.md", PINK, ["ADWIN"]),
    ("KSWIN", "compare the shapes", "drift/detectors/kswin.md", CYAN, ["KSWIN"]),
    ("The four side by side", "on a robot stream", "drift/detectors/compare.md", ORANGE, ["CUSUMDriftDetector", "PageHinkley", "ADWIN", "KSWIN"]),
]


def main():
    draw(
        PAGES,
        ("drift",),
        "THE MAP OF THE DRIFT DETECTORS",
        "the job, the page of each detector and its class; pass over a page to light its class, click to open it",
        HERE / "drift_detectors.svg",
        "The map of the drift detectors",
        {"Classifier": "C", "Regressor": "R"},
        "",
        "docs/assets/maps/make_drift_detectors_map.py",
        partial=True,
    )


if __name__ == "__main__":
    main()
