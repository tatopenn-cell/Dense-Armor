"""The interactive map of the Adaptive learners section, with the exact names of `usable`.

Draws docs/assets/maps/learn.svg for the first page of the section; it uses the same drawing as
the map of the trees (make_tree_map.py).

Run from the repo root:
    python docs/assets/maps/make_learn_map.py
"""

from make_tree_map import CYAN, HERE, ORANGE, PINK, draw

PAGES = [
    (
        "Platt scaling",
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
        "OnlineGaussianNB",
        "a mean and a variance per class",
        "learn/gaussian_nb.md",
        PINK,
        ["OnlineGaussianNB"],
    ),
    (
        "OnlineSoftmaxRegression",
        "a border that moves",
        "learn/softmax.md",
        PINK,
        ["OnlineSoftmaxRegression"],
    ),
    (
        "DriftAdaptiveClassifier",
        "a substitute on the bench",
        "learn/drift_adaptive.md",
        PINK,
        ["DriftAdaptiveClassifier"],
    ),
    (
        "RecursiveLeastSquares",
        "a line that keeps up",
        "learn/recursive_least_squares.md",
        CYAN,
        ["RecursiveLeastSquares"],
    ),
    (
        "ResidualDynamicsLearner",
        "the torque the model misses",
        "learn/residual_dynamics.md",
        CYAN,
        ["ResidualDynamicsLearner"],
    ),
    (
        "DriftAwareResidualDynamicsLearner",
        "forget faster when it changes",
        "learn/drift_aware.md",
        CYAN,
        ["DriftAwareResidualDynamicsLearner"],
    ),
]


SECTIONS = {
    "learn": (None, "THE MAP OF THE ADAPTIVE LEARNERS"),
    "metric_learning": (
        ("Metric learning", "OASIS", "LEGO", "POLA"),
        "THE MAP OF METRIC LEARNING",
    ),
    "online_classifiers": (
        ("OnlineGaussianNB", "OnlineSoftmaxRegression", "DriftAdaptiveClassifier"),
        "THE MAP OF THE ONLINE CLASSIFIERS",
    ),
    "online_dynamics": (
        (
            "RecursiveLeastSquares",
            "ResidualDynamicsLearner",
            "DriftAwareResidualDynamicsLearner",
        ),
        "THE MAP OF ONLINE DYNAMICS",
    ),
}


def main():
    for name, (titles, title) in SECTIONS.items():
        pages = PAGES if titles is None else [p for p in PAGES if p[0] in titles]
        draw(
            pages,
            ("learn",),
            title,
            "the job, the page of each tool and its classes; pass over a page to light its classes, click to open it",
            HERE / f"{name}.svg",
            title.capitalize(),
            {"Classifier": "C", "Regressor": "R"},
            "C classifier, R regressor; ",
            "docs/assets/maps/make_learn_map.py",
            partial=titles is not None,
        )


if __name__ == "__main__":
    main()
