# Trees that learn as the data arrive

A tree is a list of yes/no questions that ends with an answer. The trees in this section keep adding new questions while the data arrive, one sample at a time, but only when the data show that a new question really helps. They can also change their mind when things change, and one family can say how sure it is.

## A first example: card payments

A bank wants to flag card payments that look like fraud. Three thousand payments arrive one at a time. Most are small and made in the afternoon; a few are large and made at night. After each payment the tree is told whether it was fraud, and it learns from it.

```python
import numpy as np
from dense_armor.core.online.tree import HoeffdingTreeClassifier

rng = np.random.default_rng(0)
tree = HoeffdingTreeClassifier(grace_period=100)
for _ in range(3000):
    normal = rng.random() < 0.8
    amount = rng.normal(60, 20) if normal else rng.normal(400, 100)
    hour = rng.normal(14, 3) if normal else rng.normal(3, 2)
    tree.learn_one({"amount": amount, "hour": hour}, int(not normal))
print(tree.predict_one({"amount": 45.0, "hour": 15.0}), tree.predict_one({"amount": 420.0, "hour": 2.0}))
print(tree.explain_one({"amount": 420.0, "hour": 2.0}))
```

```
0 1
[{'feature': 'amount', 'threshold': 232.60942718228952, 'side': 'right'}]
```

A payment of 45 at 15:00 is not flagged (0) and one of 420 at 02:00 is (1). `explain_one` gives the question behind the answer: *is the amount above 232.61?* The tree found that question by itself, halfway between the usual amounts and the suspicious ones, and it did not need the hour, which here separates the two kinds of payment just as well. It did so without keeping the payments: only a few statistics per leaf.

## Gamestory

New Year's Eve in London. A mysterious thief is emptying the most guarded banks in the city.

On his trail there is a young detective. To anticipate the next move, he reasons with a chain of rapid-fire questions: "Are there guards at night? Yes. Is the alarm connected to the police? No. Then he will strike there".

His colleagues use the old method: they stay locked in the office studying old files before acting, but that night the thief does not follow any old rule and they remain stuck.

The detective instead is already on the street and reasons like the trees that grow live. He collects one clue at a time while running through the city: he waits for two or three sure proofs and, as soon as he understands the trick, he adds a new question on the fly to his mental scheme.

In the middle of the night the thief completely changes strategy. The detective does like the adaptive trees: he throws away the old hypotheses now useless and immediately creates a new plan while the chase is still underway.

At four in the morning only two banks remain. He does not guess: like a Mondrian forest, he calculates not only where the thief will go, but also how sure he is of his prediction. He sees that the evidence points 95% to the harbor bank, runs there and catches him red-handed.

The pages below follow the story, from the simplest tree to the one that knows what it does not know.

## The map of the trees

Click a page to open it, or a job on the left (or the button at the top) to go back to the [map by job](../index.md#find-a-tool-by-what-you-want-to-do).

<object type="image/svg+xml" data="../assets/maps/trees.svg" style="width:100%"></object>

## The trees, one page each

| page | classes | what it is for |
|---|---|---|
| [Hoeffding tree](hoeffding.md) | `HoeffdingTreeClassifier`, `HoeffdingTreeRegressor` | the base tree: a split only when the data prove it; a class or a number |
| [Hoeffding Anytime tree](efdt.md) | `HoeffdingAnytimeTreeClassifier` | it can replace a split that is no longer the best |
| [Hoeffding Adaptive tree](adaptive.md) | `HoeffdingAdaptiveTreeClassifier` | it follows a sudden change with a background subtree |
| [Stochastic gradient trees](sgt.md) | `SGTRegressor`, `SGTClassifier` | one rule for numbers and classes, from the gradient of the loss |
| [Mondrian forests](mondrian.md) | `MondrianForestRegressor`, `MondrianForestClassifier` | a prediction with its uncertainty |

The forests built from these trees (the adaptive random forest, the extra trees) are in the
[Forests](../forest/index.md) section; the trees grown on the whole table are in
[Machine learning](../ml/tree.md).

## Which tree for which job

The same streams for every model, 4000 samples each, tested before learning each sample.
Classifiers see two features `a`, `b`; the target `y` is 0 or 1; `a ~ N(3y, 0.5)` and
`b ~ N(0, 0.5)`; at sample 2000 `a` switches to `N(3(1-y), 0.5)`. Regressors see two joint
angles `q1`, `q2 ~ U(-pi, pi)` and target `tau = 9.81 (2 cos q1 + cos(q1+q2))` scaled by 1.0
before sample 2000 and 1.5 after. The script `docs/assets/tree/make_table.py` regenerates the
table below with these streams; `ms per sample` is measured on the machine that runs the
script.

| model | task | stationary | with a change | ms per sample |
|---|---|---|---|---|
| `HoeffdingTreeClassifier` | accuracy | 0.992 | 0.524 | 0.030 |
| `HoeffdingAnytimeTreeClassifier` | accuracy | 0.992 | 0.659 | 0.031 |
| `HoeffdingAdaptiveTreeClassifier` (`ADWIN`) | accuracy | 0.992 | 0.962 | 0.496 |
| `SGTClassifier` | accuracy | 0.842 | 0.773 | 0.061 |
| `MondrianForestClassifier` | accuracy | 0.998 | 0.605 | 1.272 |
| `SGTRegressor` | mean abs. error | 10.218 | 12.392 | 0.035 |
| `MondrianForestRegressor` | mean abs. error | 1.161 | 3.343 | 1.314 |

On a stable stream the three Hoeffding trees and the Mondrian forest classify almost perfectly;
after a change, the adaptive tree recovers best among the trees, at about 16 times the cost per
sample of the plain tree; the Mondrian forest is the most accurate regressor here and the only
model with an uncertainty, at about 1 ms per sample; the stochastic gradient tree is the
cheapest regressor. Accuracy "with a change" counts the whole stream, before and after the
change.

## The robot examples

Further down, the pages use a robot arm as their example.

A decision tree makes choices by asking a chain of questions (for example, "Is the joint torque above 3.5 N·m?"), allowing humans to easily understand the reasoning behind its decisions.
Traditionally, these trees are trained in batches, meaning they require all the data to be available at once. In contrast, the trees described in this section grow dynamically—one single sample at a time—while the robot is actively working. They create a new branch (split a leaf) only when they have gathered enough data to prove the change is statistically accurate.
Furthermore, they can adapt and change their mind if the robot's behavior shifts, and one specific family of trees can even communicate how confident it is in its own predictions.

## Details

- The Mondrian, SGT and FIMT-DD classes and the Hoeffding forests cap their size with `max_nodes` (default 10000); the Hoeffding trees bound the leaves with statistics instead (see the [Hoeffding tree](hoeffding.md)); `budget_s` is about ten times the 99th percentile of `learn_one` measured on a desktop CPU.
- Figures on these pages are generated by `docs/assets/tree/make_figure.py` and `docs/assets/tree/make_mondrian.py`; the table above by `docs/assets/tree/make_table.py`.
