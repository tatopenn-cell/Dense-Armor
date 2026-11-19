"""The interactive map of the Clustering section, with the exact names of `usable`.

Draws docs/assets/maps/cluster.svg with the same drawing as the other section maps (make_tree_map.py).

Run from the repo root:
    python docs/assets/maps/make_cluster_map.py
"""

from make_tree_map import CYAN, HERE, ORANGE, PINK, draw

PAGES = [
    (
        "OnlineKMeans",
        "k groups, one point at a time",
        "cluster/kmeans.md",
        CYAN,
        ["OnlineKMeans"],
    ),
    (
        "DenStream",
        "groups of any shape, with outliers",
        "cluster/denstream.md",
        ORANGE,
        ["DenStream"],
    ),
    (
        "Visual vocabulary",
        "image patches as words",
        "cluster/visual_vocabulary.md",
        PINK,
        ["BagOfVisualWords"],
    ),
]


def main():
    draw(
        PAGES,
        ("cluster", "vision"),
        "THE MAP OF CLUSTERING",
        "the job, the page of each tool and its classes; pass over a page to light its classes, click to open it",
        HERE / "cluster.svg",
        "The map of clustering",
        {"Classifier": "C", "Regressor": "R"},
        "C classifier, R regressor; ",
        "docs/assets/maps/make_cluster_map.py",
        partial=True,
    )


if __name__ == "__main__":
    main()
