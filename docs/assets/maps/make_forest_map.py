"""The interactive map of the Forests section, with the exact names of `usable`.

Draws docs/assets/maps/forest.svg with the same drawing as the other section maps (make_tree_map.py).

Run from the repo root:
    python docs/assets/maps/make_forest_map.py
"""

from make_tree_map import CYAN, HERE, ORANGE, PINK, draw

PAGES = [
    (
        "AdaptiveRandomForestClassifier",
        "replace the trees that go wrong",
        "forest/adaptive_random_forest.md",
        CYAN,
        ["AdaptiveRandomForestClassifier", "SubspaceHoeffdingTree"],
    ),
    (
        "AggregatedMondrianForest",
        "average every depth",
        "forest/aggregated_mondrian_forest.md",
        ORANGE,
        ["AggregatedMondrianForestClassifier", "AggregatedMondrianForestRegressor"],
    ),
    (
        "OnlineExtraTreesRegressor",
        "draw the split at random",
        "forest/online_extra_trees.md",
        PINK,
        ["OnlineExtraTreesRegressor"],
    ),
]


def main():
    draw(
        PAGES,
        ("forest",),
        "THE MAP OF FORESTS",
        "the job, the page of each tool and its classes; pass over a page to light its classes, click to open it",
        HERE / "forest.svg",
        "The map of forests",
        {"Classifier": "C", "Regressor": "R"},
        "C classifier, R regressor; ",
        "docs/assets/maps/make_forest_map.py",
    )


if __name__ == "__main__":
    main()
