"""The interactive map of the Ensembles section, with the exact names of `usable`.

Draws docs/assets/maps/ensemble.svg with the same drawing as the other section maps (make_tree_map.py).

Run from the repo root:
    python docs/assets/maps/make_ensemble_map.py
"""

from make_tree_map import CYAN, HERE, ORANGE, PINK, draw

PAGES = [
    (
        "Voting",
        "the majority decides",
        "ensemble/voting.md",
        CYAN,
        ["VotingClassifier"],
    ),
    (
        "Bagging",
        "a different sample for every member",
        "ensemble/bagging.md",
        ORANGE,
        ["OnlineBaggingClassifier"],
    ),
    (
        "Boosting",
        "weight the samples the members miss",
        "ensemble/boosting.md",
        PINK,
        ["OnlineBoostingClassifier"],
    ),
    (
        "Weighted average",
        "a regressor",
        "ensemble/ewa.md",
        CYAN,
        ["ExponentiallyWeightedAverageRegressor"],
    ),
    (
        "Stacking",
        "a second model on top",
        "ensemble/stacking.md",
        ORANGE,
        ["StackingClassifier"],
    ),
]


def main():
    draw(
        PAGES,
        ("ensemble",),
        "THE MAP OF ENSEMBLES",
        "the job, the page of each ensemble and its class; pass over a page to light its class, click to open it",
        HERE / "ensemble.svg",
        "The map of ensembles",
        {"Classifier": "C", "Regressor": "R"},
        "C classifier, R regressor; ",
        "docs/assets/maps/make_ensemble_map.py",
    )


if __name__ == "__main__":
    main()
