"""The interactive map of the Streams section: the readers, the merge and the channels of a sample.

Draws docs/assets/maps/stream.svg with the same drawing as the other section maps (make_tree_map.py). The
readers and the merge are functions, so they are one line each; the classes of the channels are those of
the catalog of `usable`.

Run from the repo root:
    python docs/assets/maps/make_stream_map.py
"""

import json

from make_tree_map import CYAN, HERE, ORANGE, PINK, REPO, draw

PAGES = [
    (
        "CSV files, ROS bags and arrays",
        "the readers",
        "stream/readers.md",
        CYAN,
        [],
    ),
    (
        "Merge by time and shuffle",
        "several streams, one timeline",
        "stream/merge.md",
        ORANGE,
        [],
    ),
    (
        "Choosing and reshaping channels",
        "keep, drop and change channels",
        "stream/channels.md",
        PINK,
        ["Select", "Discard", "FuncTransformer"],
    ),
]
SUMMARIES = {
    "CSV files, ROS bags and arrays": "iter_csv, iter_rosbag, iter_array",
    "Merge by time and shuffle": "merge_by_time, shuffle",
}


def main():
    ontology = json.loads(
        (REPO / "dense_armor" / "usable" / "ontology.json").read_text(encoding="utf-8")
    )
    jobs = [
        j
        for j, v in ontology["jobs"].items()
        if {"compose", "data"} & set(v["domains"])
    ]
    draw(
        PAGES,
        ("compose",),
        "THE MAP OF STREAMS",
        "the job, the page of each tool and what it holds; pass over a page to light it, click to open it",
        HERE / "stream.svg",
        "The map of streams",
        {"Classifier": "C", "Regressor": "R"},
        "C classifier, R regressor; ",
        "docs/assets/maps/make_stream_map.py",
        partial=True,
        jobs=jobs,
        hub="stream",
        summaries=SUMMARIES,
    )


if __name__ == "__main__":
    main()
