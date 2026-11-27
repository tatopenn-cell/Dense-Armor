"""The interactive map of the Drift detection section, with the exact names of `usable`.

Draws docs/assets/maps/drift.svg with the same drawing as the other section maps (make_tree_map.py). The offline page is
about the algorithm on a whole series, so its card carries a line of text.

Run from the repo root:
    python docs/assets/maps/make_drift_map.py
"""

from make_tree_map import CYAN, HERE, ORANGE, PINK, draw

PAGES = [
    ("CUSUM and ARL theory", "accumulate small deviations", "drift/cusum/index.md", CYAN, ["CUSUMDriftDetector"]),
    ("Drift detectors", "CUSUM, Page-Hinkley, ADWIN, KSWIN", "drift/detectors/index.md", ORANGE, ["CUSUMDriftDetector", "PageHinkley", "ADWIN", "KSWIN"]),
    ("Offline algorithm", "on a whole series", "drift/offline.md", PINK, ["offline"]),
]

SUMMARIES = {"Offline algorithm": "offline"}


def main():
    draw(
        PAGES,
        ("drift",),
        "THE MAP OF DRIFT DETECTION",
        "the job, the page of each detector and its classes; pass over a page to light its classes, click to open it",
        HERE / "drift.svg",
        "The map of drift detection",
        {"Classifier": "C", "Regressor": "R"},
        "",
        "docs/assets/maps/make_drift_map.py",
        partial=True,
        summaries=SUMMARIES,
    )


if __name__ == "__main__":
    main()
