# Fit: the same algorithm on the whole dataset

## The map of fit

Click a page to open it, or the job on the left (or the button at the top) to go back to the [map of the library](../index.md#the-library-folder-by-folder).

<object type="image/svg+xml" data="../assets/maps/fit.svg" style="width:100%"></object>

## The problem, in one line

A desk keeps a table of past orders and wants a model that learns from it. Every online algorithm of the library learns one row at a time with `learn_one`, which is how it works on a live stream. With a table already in hand, you hand it the rows all at once: `fit` feeds them to `learn_one`, so the result is the same as the loop that you would write by hand.

## The words

| Word | What it means here | In the example |
|---|---|---|
| `learn_one` | learn from one row | one order, with its slippage |
| `fit` | learn from a whole table, in one call | the 200 orders of the table |
| epoch | one pass over the whole table | 5 passes |
| shuffle | read the rows in a different order at every pass | `shuffle=True` |
| batch algorithm | one that needs the whole table to start, like a decision tree | `BatchDecisionTreeClassifier` |

## A first example: `fit` and the loop by hand are the same

A model of the slippage is learned twice from the same 200 orders: with a loop of `learn_one`, and with one call of `fit`. The answer says if the two models predict the same number for a new order.

```python
import numpy as np
from dense_armor.core.online.linear.glm import LinearRegression
from dense_armor.core.online.optim.sgd import SGD

rng = np.random.default_rng(0)
A = rng.normal(size=(200, 2))
X = [{"size": float(u), "depth": float(v)} for u, v in A]
y = [float(2 * u - v) for u, v in A]
by_hand = LinearRegression(optimizer=SGD(lr=0.02))
for x, t in zip(X, y):
    by_hand.learn_one(x, t)
by_fit = LinearRegression(optimizer=SGD(lr=0.02)).fit(X, y)
probe = {"size": 1.0, "depth": 1.0}
print(by_hand.predict_one(probe) == by_fit.predict_one(probe))
```

```
True
```

The two give the same prediction, to the last digit: `fit` is the loop, written once on the base class.

## The pages of the section

| Page | Classes | What it does |
|---|---|---|
| [Fit on an estimator](estimators.md) | `LinearRegression`, `StandardScaler` | the same algorithm on the whole table, with passes, shuffle and timestamps |
| [Algorithms that work on the whole table](../ml/fit_batch.md) (in Machine learning) | `BatchDecisionTreeClassifier`, `OnlineBaggingClassifier`, `TextVectorizer` | the three that need the whole table by nature |
| [LEGO, POLA and OASIS](metric_learners.md) | `LEGO`, `POLA`, `OASIS` | the first online algorithms of the library: a distance learned from a pair or a triplet at a time |

## What the numbers say: who has `fit`, and what it does

The classes of the library fall in three groups. The classes of the library fall in three groups. The first number counts the classes that have a `fit` of their own, the second those that use the `fit` of the base class and have `learn_one`, and the third those that use the `fit` of the base class and have no `learn_one`: their unit is not a row, it is a value for `update`, or a pair or a triplet for the metric learners `LEGO`, `POLA` and `OASIS`.

```python
import importlib
from pathlib import Path
from dense_armor.roles import Root
from dense_armor.usable.catalog import get_catalog

own = plain = other = 0
for row in get_catalog(Path(".")):
    if row["kind"] != "class":
        continue
    try:
        cls = getattr(importlib.import_module(row["module"]), row["class"])
    except Exception:
        continue
    if not hasattr(cls, "fit"):
        continue
    if cls.fit is not Root.fit:
        own += 1
    elif hasattr(cls, "learn_one"):
        plain += 1
    else:
        other += 1
print(own, plain, other)
```

```
92 151 31
```

The second group is the one that this section is about. The metric learners are online algorithms whose unit is a pair or a triplet, and the page [LEGO, POLA and OASIS](metric_learners.md) shows them.

## The same problem on a robot

A robot logs a run: the torque that a joint needed for a given speed and a given load. The algorithms are the same, and each page ends with the same code written for a robot.

## Details

### What `fit` does and does not do
`fit` is defined once on the base class `Root`, so every estimator of the library has it. It adds a way in for a whole dataset and changes nothing that exists: `learn_one`, `learn_many` and every algorithm stay as they were. It is a loop over `learn_one`; the classic batch version of each algorithm, written from its original paper, is the next step of the program (row 24 of the master program), one family at a time; the ones that exist today are in section 4.

### Arguments
`X` is a list of rows, `y` a list of labels of the same length, `t` a list of timestamps of the same length, `epochs` an integer of at least 1, `shuffle` a boolean, `seed` an integer. A mismatch of lengths, `epochs < 1`, or `t` with more than one epoch raises `ValueError`; an object without `learn_one` raises `TypeError`.
