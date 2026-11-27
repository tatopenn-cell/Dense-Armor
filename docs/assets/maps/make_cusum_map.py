"""The interactive map of the CUSUM section, with the exact names of `usable`.

Draws docs/assets/maps/cusum.svg with the same drawing as the other section maps (make_tree_map.py). The pages on the
formula, on the average run length and on the recorded series are about functions or a formula, not classes of the
catalog, so each of their cards carries a line of text.

Run from the repo root:
    python docs/assets/maps/make_cusum_map.py
"""

from make_tree_map import CYAN, HERE, ORANGE, PINK, draw

PAGES = [
    ("CUSUMDriftDetector", "one value per call", "drift/cusum/detector.md", CYAN, ["CUSUMDriftDetector"]),
    ("one_sided_arl", "one direction", "drift/cusum/one_sided_arl.md", ORANGE, ["one_sided_arl"]),
    ("two_sided_arl", "both directions", "drift/cusum/two_sided_arl.md", PINK, ["two_sided_arl"]),
    ("detectability_report", "in the units of the series", "drift/cusum/detectability_report.md", CYAN, ["detectability_report"]),
    ("expected_false_alarm_run", "samples between false alarms", "drift/cusum/expected_false_alarm_run.md", ORANGE, ["expected_false_alarm_run"]),
    ("expected_detection_delay", "samples to detect a shift", "drift/cusum/expected_detection_delay.md", PINK, ["expected_detection_delay"]),
    ("cusum_detector", "a whole recording", "drift/cusum/cusum_detector.md", CYAN, ["cusum_detector"]),
    ("Results", "against Page-Hinkley and ADWIN", "drift/cusum/results.md", ORANGE, ["CUSUMDriftDetector", "PageHinkley", "ADWIN"]),
]

SUMMARIES = {"one_sided_arl": "one_sided_arl", "two_sided_arl": "two_sided_arl", "detectability_report": "detectability_report", "expected_false_alarm_run": "expected_false_alarm_run", "expected_detection_delay": "expected_detection_delay", "cusum_detector": "cusum_detector"}


def main():
    draw(
        PAGES,
        ("drift",),
        "THE MAP OF CUSUM",
        "the job, the page of each part and its classes; pass over a page to light its classes, click to open it",
        HERE / "cusum.svg",
        "The map of CUSUM",
        {"Classifier": "C", "Regressor": "R"},
        "",
        "docs/assets/maps/make_cusum_map.py",
        partial=True,
        summaries=SUMMARIES,
    )


if __name__ == "__main__":
    main()
