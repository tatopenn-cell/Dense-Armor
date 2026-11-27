"""The interactive map of the Anomaly detection section, with the exact names of `usable`.

Draws docs/assets/maps/anomaly.svg with the same drawing as the other section maps (make_tree_map.py). The pages of
robust filters and of curvature are about functions, not classes of the catalog, so each of their cards carries a line
of text.

Run from the repo root:
    python docs/assets/maps/make_anomaly_map.py
"""

from make_tree_map import CYAN, HERE, ORANGE, PINK, draw

PAGES = [
    (
        "Streaming",
        "a score for every new sample",
        "anomaly/streaming/",
        CYAN,
        [
            "streamDD",
            "MultistreamDD",
            "streamDS",
            "HampelScorer",
            "TukeyScorer",
            "ChauvenetScorer",
            "SigmaClipScorer",
            "HampelFilter",
        ],
    ),
    ("Robust filters", "four detectors on a whole series", "anomaly/robust_filters.md", ORANGE, ["pressure_valve"]),
    ("Curvature", "a proximity score in [0, 1)", "anomaly/curvature.md", PINK, ["curvature"]),
    ("Online Robust Mahalanobis", "distance from a robust median", "anomaly/mahalanobis.md", CYAN, ["OnlineRobustMahalanobis"]),
    ("LODA", "rare bins in random projections", "anomaly/loda.md", ORANGE, ["LODA"]),
    ("Half-Space Trees", "few reference samples in the region", "anomaly/hst.md", PINK, ["HalfSpaceTrees"]),
    ("One-Class SGD (SONAR)", "distance from a learned boundary", "anomaly/ocsvm.md", CYAN, ["OneClassSGD"]),
    ("Online Isolation Forest", "depth to isolate a point", "anomaly/oiforest.md", ORANGE, ["OnlineIsolationForest"]),
    ("Local Outlier Factor", "density against the neighbours", "anomaly/lof.md", PINK, ["LocalOutlierFactor"]),
    ("Offline algorithms", "five detectors on a whole table", "anomaly/offline.md", CYAN, ["offline"]),
]

SUMMARIES = {"Robust filters": "pressure_valve", "Curvature": "curvature", "Offline algorithms": "offline"}


def main():
    draw(
        PAGES,
        ("anomaly",),
        "THE MAP OF ANOMALY DETECTION",
        "the job, the page of each detector and its classes; pass over a page to light its classes, click to open it",
        HERE / "anomaly.svg",
        "The map of anomaly detection",
        {"Classifier": "C", "Regressor": "R"},
        "",
        "docs/assets/maps/make_anomaly_map.py",
        partial=True,
        summaries=SUMMARIES,
    )


if __name__ == "__main__":
    main()
