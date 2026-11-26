"""The interactive map of the Active learning section, with the exact names of `usable`.

Draws docs/assets/maps/active.svg with the same drawing as the other section maps (make_tree_map.py).

Run from the repo root:
    python docs/assets/maps/make_active_map.py
"""

from make_tree_map import CYAN, HERE, ORANGE, PINK, draw

PAGES = [
    ("Random labelling", "ask with a fixed probability", "active/random_strategy.md", CYAN, ["RandomStrategy"]),
    ("Fixed threshold", "ask when unsure", "active/fixed_uncertainty.md", ORANGE, ["FixedUncertainty"]),
    ("Variable threshold", "spend the budget evenly", "active/variable_uncertainty.md", PINK, ["VariableUncertainty"]),
    ("Randomised threshold", "ask a surprise question", "active/randomized_uncertainty.md", CYAN, ["VariableRandomizedUncertainty"]),
    ("The shared methods", "what every strategy inherits", "active/strategy.md", ORANGE, ["ActiveStrategy"]),
]


def main():
    draw(
        PAGES,
        ("active",),
        "THE MAP OF ACTIVE LEARNING",
        "the job, the page of each strategy and its class; pass over a page to light its class, click to open it",
        HERE / "active.svg",
        "The map of active learning",
        {"Classifier": "C", "Regressor": "R"},
        "",
        "docs/assets/maps/make_active_map.py",
        partial=True,
    )


if __name__ == "__main__":
    main()
