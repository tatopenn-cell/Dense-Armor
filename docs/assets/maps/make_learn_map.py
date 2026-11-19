"""The interactive map of the Adaptive learners section, with the exact names of `usable`.

Draws docs/assets/maps/learn.svg for the first page of the section; it uses the same drawing as
the map of the trees (make_tree_map.py).

Run from the repo root:
    python docs/assets/maps/make_learn_map.py
"""

from make_tree_map import CYAN, HERE, ORANGE, PINK, draw

PAGES = [
    (
        "Calibration",
        "honest probabilities",
        "learn/calibration.md",
        CYAN,
        ["OnlinePlattScaling"],
    ),
    (
        "Metric learning",
        "which customers look alike",
        "learn/metric_learning.md",
        ORANGE,
        ["MetricKNNClassifier", "MetricLearner"],
    ),
    (
        "OASIS",
        "a similarity from triplets",
        "learn/oasis.md",
        ORANGE,
        ["OASIS"],
    ),
    (
        "LEGO",
        "a distance from pairs and targets",
        "learn/lego.md",
        ORANGE,
        ["LEGO"],
    ),
    (
        "POLA",
        "similar or different pairs",
        "learn/pola.md",
        ORANGE,
        ["POLA"],
    ),
    (
        "Online classifiers",
        "a state that follows the data",
        "learn/online_classifiers.md",
        PINK,
        ["OnlineGaussianNB", "OnlineSoftmaxRegression", "DriftAdaptiveClassifier"],
    ),
    (
        "Online dynamics",
        "what the formula misses",
        "learn/online_dynamics.md",
        CYAN,
        [
            "RecursiveLeastSquares",
            "ResidualDynamicsLearner",
            "DriftAwareResidualDynamicsLearner",
        ],
    ),
]


def main():
    draw(
        PAGES,
        ("learn",),
        "THE MAP OF THE ADAPTIVE LEARNERS",
        "the job, the page of each tool and its classes; pass over a page to light its classes, click to open it",
        HERE / "learn.svg",
        "The map of the adaptive learners",
        {"Classifier": "C", "Regressor": "R"},
        "C classifier, R regressor; ",
        "docs/assets/maps/make_learn_map.py",
    )


if __name__ == "__main__":
    main()
