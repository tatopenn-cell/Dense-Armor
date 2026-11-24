"""The interactive map of the Moments section, with the exact names of `usable`.

Draws docs/assets/maps/moments.svg with the same drawing as the other section maps (make_tree_map.py).

Run from the repo root:
    python docs/assets/maps/make_moments_map.py
"""

from make_tree_map import CYAN, HERE, ORANGE, PINK, draw

PAGES = [
    (
        "RunningMoments",
        "mean, wobble, skew, tails",
        "stats/moments/running_moments.md",
        CYAN,
        ["RunningMoments"],
    ),
    (
        "Merging and many channels",
        "join halves, follow every joint",
        "stats/moments/merge_and_vector.md",
        ORANGE,
        ["RunningMomentsVector"],
    ),
    (
        "EWStats",
        "forget the old, follow the drift",
        "stats/moments/ewstats.md",
        PINK,
        ["EWStats"],
    ),
]


def main():
    draw(
        PAGES,
        ("stats",),
        "THE MAP OF MOMENTS",
        "the job, the page of each tool and its classes; pass over a page to light its classes, click to open it",
        HERE / "moments.svg",
        "The map of moments",
        {},
        "",
        "docs/assets/maps/make_moments_map.py",
        partial=True,
    )


if __name__ == "__main__":
    main()
