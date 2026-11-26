"""The interactive map of the Bandits section, with the exact names of `usable`.

Draws docs/assets/maps/bandit.svg with the same drawing as the other section maps (make_tree_map.py). The function
offline_replay is not a class of the catalog, so its card carries its name as a summary line.

Run from the repo root:
    python docs/assets/maps/make_bandit_map.py
"""

from make_tree_map import CYAN, HERE, ORANGE, PINK, draw

PAGES = [
    ("Epsilon-greedy", "explore with a fixed probability", "bandit/epsilon_greedy.md", CYAN, ["EpsilonGreedy"]),
    ("UCB1 and UCB1-Tuned", "the mean plus a bonus", "bandit/ucb.md", ORANGE, ["UCB1", "UCB1Tuned"]),
    ("Thompson sampling", "draw from what you believe", "bandit/thompson.md", PINK, ["ThompsonSampling"]),
    ("Bayes-UCB", "the highest quantile", "bandit/bayes_ucb.md", CYAN, ["BayesUCB"]),
    ("Exp3", "no assumption on the rewards", "bandit/exp3.md", ORANGE, ["Exp3"]),
    ("LinUCB", "use a context", "bandit/linucb.md", PINK, ["LinUCBDisjoint"]),
    ("Offline replay", "judge a policy on a log", "bandit/offline_replay.md", CYAN, ["offline_replay"]),
    ("The shared methods", "what every policy inherits", "bandit/policy.md", ORANGE, ["BanditPolicy"]),
    ("RobotKpSelector", "choose a controller gain", "bandit/robot_kp_selector.md", PINK, ["RobotKpSelector"]),
]

SUMMARIES = {"Offline replay": "offline_replay"}


def main():
    draw(
        PAGES,
        ("bandit",),
        "THE MAP OF BANDITS",
        "the job, the page of each policy and its classes; pass over a page to light its classes, click to open it",
        HERE / "bandit.svg",
        "The map of bandits",
        {"Classifier": "C", "Regressor": "R"},
        "",
        "docs/assets/maps/make_bandit_map.py",
        partial=True,
        summaries=SUMMARIES,
    )


if __name__ == "__main__":
    main()
