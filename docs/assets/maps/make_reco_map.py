"""The interactive map of the Recommenders section, with the exact names of `usable`.

Draws docs/assets/maps/reco.svg with the same drawing as the other section maps (make_tree_map.py). SDFT and Skyline
belong to the domain stats of `usable`, so the map reads both domains.

Run from the repo root:
    python docs/assets/maps/make_reco_map.py
"""

from make_tree_map import CYAN, HERE, ORANGE, PINK, draw

PAGES = [
    ("Baseline", "the mean plus two biases", "reco/baseline.md", CYAN, ["Baseline"]),
    ("Biased matrix factorization", "biases and latent factors", "reco/biased_mf.md", ORANGE, ["BiasedMF"]),
    ("Plain matrix factorization", "latent factors only", "reco/funk_mf.md", PINK, ["FunkMF"]),
    ("Top-k recommendations", "the best items for a user", "reco/rank.md", CYAN, ["Baseline", "BiasedMF", "FunkMF"]),
    ("Two sliding-window structures", "the spectrum and the skyline", "reco/windows.md", ORANGE, ["SDFT", "Skyline"]),
]


def main():
    draw(
        PAGES,
        ("reco", "stats"),
        "THE MAP OF RECOMMENDERS",
        "the job, the page of each model and its classes; pass over a page to light its classes, click to open it",
        HERE / "reco.svg",
        "The map of recommenders",
        {"Classifier": "C", "Regressor": "R"},
        "",
        "docs/assets/maps/make_reco_map.py",
        partial=True,
    )


if __name__ == "__main__":
    main()
