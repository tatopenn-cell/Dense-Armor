"""The interactive map of the Protection section, with the exact names of `usable`.

Draws docs/assets/maps/protect.svg with the same drawing as the other section maps (make_tree_map.py). The two engines
are modules, not classes of the catalog, so their cards carry a line of text.

Run from the repo root:
    python docs/assets/maps/make_protect_map.py
"""

from make_tree_map import CYAN, HERE, ORANGE, PINK, draw

PAGES = [
    ("Arbiter", "label each point, route it", "protect/arbiter.md", CYAN, ["StreamingArbiter", "StreamingHealing"]),
    ("Shields", "one for a series, one for a model", "protect/shields.md", ORANGE, ["Armor", "Orca"]),
    ("Orca (online)", "a shield for a whole model", "protect/orca.md", PINK, ["Orca"]),
    ("Adaptive engine (online)", "follows a slow drift", "protect/engine.md", CYAN, ["engine"]),
    ("Armor (offline)", "a shield for one series", "protect/armor.md", ORANGE, ["Armor"]),
    ("Hybrid engine (offline)", "the core of Armor", "protect/hybrid_engine.md", PINK, ["hybrid_engine"]),
]

SUMMARIES = {"Adaptive engine (online)": "engine", "Hybrid engine (offline)": "hybrid_engine"}


def main():
    draw(
        PAGES,
        ("protect", "core"),
        "THE MAP OF PROTECTION",
        "the job, the page of each tool and its classes; pass over a page to light its classes, click to open it",
        HERE / "protect.svg",
        "The map of protection",
        {"Classifier": "C", "Regressor": "R"},
        "",
        "docs/assets/maps/make_protect_map.py",
        partial=True,
        summaries=SUMMARIES,
    )


if __name__ == "__main__":
    main()
