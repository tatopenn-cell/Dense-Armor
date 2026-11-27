"""The interactive map of the streaming anomaly detection section, with the exact names of `usable`.

Draws docs/assets/maps/streaming.svg with the same drawing as the other section maps (make_tree_map.py).
The class_seg_multi page is about a function, not a class of the catalog, so its card
carries a line of text via `summaries`. The last card is the Online Robust Mahalanobis page of the
anomaly section, another section of the library, like make_drift_detectors_map.py does for CUSUM.

Run from the repo root:
    python docs/assets/maps/make_streaming_map.py
"""

from make_tree_map import CYAN, DOCS, HERE, ORANGE, PINK, draw

PAGES_ALL = [
    ("Stream", "a score for every new sample", "anomaly/streaming/index.md", CYAN, ["streamDD"]),
    ("Flag", "flag a strange sample", "anomaly/streaming/streaming_deviation_detector.md", CYAN, ["streamDD"]),
    ("Score", "the score itself", "anomaly/streaming/streaming_deviation_scorer.md", ORANGE, ["streamDS"]),
    ("Multi", "all the joints at once", "anomaly/streaming/multi_channel_streaming_deviation_detector.md", PINK, ["MultistreamDD"]),
    ("Spike", "spike versus regime, batch", "anomaly/streaming/class_seg_multi.md", CYAN, ["class_seg_multi"]),
    ("Hampel", "Hampel, one sample at a time", "anomaly/streaming/hampel_scorer.md", ORANGE, ["HampelScorer"]),
    ("Tukey", "Tukey fences, one sample at a time", "anomaly/streaming/tukey_scorer.md", PINK, ["TukeyScorer"]),
    ("Chauvenet", "Chauvenet, one sample at a time", "anomaly/streaming/chauvenet_scorer.md", CYAN, ["ChauvenetScorer"]),
    ("Sigma", "sigma clipping, one sample at a time", "anomaly/streaming/sigma_clip_scorer.md", ORANGE, ["SigmaClipScorer"]),
    ("Clean", "replace an outlier with the median", "anomaly/streaming/hampel_filter.md", PINK, ["HampelFilter"]),
    ("OnlineRobustMahalanobis", "the full page of the anomaly section", "anomaly/mahalanobis.md", CYAN, ["OnlineRobustMahalanobis"]),
]

PAGES = [p for p in PAGES_ALL if (DOCS / p[2]).exists()]

SUMMARIES = {"Spike": "class_seg_multi"} if any(p[0] == "Spike" for p in PAGES) else None


def main():
    draw(
        PAGES,
        ("anomaly",),
        "THE MAP OF STREAMING ANOMALY DETECTION",
        "the job, the page of each part and its classes; pass over a page to light its classes, click to open it",
        HERE / "streaming.svg",
        "The map of streaming anomaly detection",
        {"Classifier": "C", "Regressor": "R"},
        "",
        "docs/assets/maps/make_streaming_map.py",
        partial=True,
        summaries=SUMMARIES,
    )


if __name__ == "__main__":
    main()
