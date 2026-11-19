"""The interactive maps of the Optimizers and Linear models sections, with the exact names of `usable`.

Draws docs/assets/maps/optim.svg and linear.svg with the same drawing as the map of the trees
(make_tree_map.py). Only the classes that have a page are on the map.

Run from the repo root:
    python docs/assets/maps/make_optim_linear_map.py
"""

from make_tree_map import CYAN, HERE, ORANGE, PINK, draw

MAPS = {
    "optim": (
        "THE MAP OF THE OPTIMIZERS",
        [
            ("SGD", "the plain gradient step", "optim/sgd.md", CYAN, ["SGD"]),
            ("AdaGrad", "a step per weight", "optim/adagrad.md", ORANGE, ["AdaGrad"]),
            ("Adam", "momentum and scale", "optim/adam.md", PINK, ["Adam"]),
            ("FTRLProximal", "sparse weights", "optim/ftrl.md", CYAN, ["FTRLProximal"]),
            (
                "OnlineNewtonStep",
                "a step that uses curvature",
                "optim/newton.md",
                ORANGE,
                ["OnlineNewtonStep"],
            ),
        ],
    ),
    "linear": (
        "THE MAP OF THE LINEAR MODELS",
        [
            (
                "GLM",
                "a line, a logistic curve",
                "linear/glm.md",
                CYAN,
                ["LinearRegression", "LogisticRegression"],
            ),
            (
                "Bayesian linear regression",
                "a line with its doubt",
                "linear/bayes.md",
                ORANGE,
                ["BayesianLinearRegression"],
            ),
            (
                "Payload estimator",
                "the mass of a payload",
                "linear/payload.md",
                PINK,
                ["PayloadEstimator"],
            ),
        ],
    ),
}


def main():
    for name, (title, pages) in MAPS.items():
        draw(
            pages,
            (name,),
            title,
            "the job, the page of each tool and its classes; pass over a page to light its classes, click to open it",
            HERE / f"{name}.svg",
            title.capitalize(),
            {"Classifier": "C", "Regressor": "R"},
            "C classifier, R regressor; ",
            "docs/assets/maps/make_optim_linear_map.py",
            partial=True,
        )


if __name__ == "__main__":
    main()
