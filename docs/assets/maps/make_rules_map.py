"""The interactive map of the Rules section, with the exact names of `usable`.

Draws docs/assets/maps/rules.svg with the same drawing as the other section maps (make_tree_map.py).

Run from the repo root:
    python docs/assets/maps/make_rules_map.py
"""

from make_tree_map import CYAN, HERE, ORANGE, PINK, draw

PAGES = [
    ("Learn a rule set", "cuts, n_min and explain_one", "rules/learn.md", CYAN, ["AMRulesRegressor"]),
    ("When the world changes", "a rule is dropped", "rules/change.md", ORANGE, ["AMRulesRegressor"]),
    ("Options", "how a rule predicts, how many rules", "rules/options.md", PINK, ["AMRulesRegressor"]),
]


def main():
    draw(
        PAGES,
        ("rules",),
        "THE MAP OF RULES",
        "the job, the page of each part of the model and its class; pass over a page to light its class, click to open it",
        HERE / "rules.svg",
        "The map of rules",
        {"Classifier": "C", "Regressor": "R"},
        "",
        "docs/assets/maps/make_rules_map.py",
        partial=True,
    )


if __name__ == "__main__":
    main()
