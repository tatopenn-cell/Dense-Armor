# Offline: the same algorithm on the whole dataset

Every online algorithm of the library can also learn from a dataset you already have, all at once: you hand it the rows and it learns from them.

## 1. One call: `fit`

An online algorithm learns one row at a time with `learn_one`. `fit` takes all the rows and feeds them to `learn_one` in order, so the result is the same as the loop you would write by hand.

```python
import numpy as np
from dense_armor.utility.linear.glm import LinearRegression
from dense_armor.utility.optim.sgd import SGD

rng = np.random.default_rng(0)
X = [{"a": float(u), "b": float(v)} for u, v in rng.normal(size=(200, 2))]
y = [2 * x["a"] - x["b"] for x in X]
m = LinearRegression(optimizer=SGD()).fit(X, y)
print(round(m.predict_one({"a": 1.0, "b": 1.0}), 2))
```

```
1.0
```

The rule behind the data is `2a - b`, so the answer for `a = 1, b = 1` is 1.0. `fit` returns the estimator itself, so the call can be chained.

## 2. Several passes, in a shuffled order

An offline dataset can be read more than once. `epochs` sets the number of passes, `shuffle` changes the order at every pass, and `seed` makes the order repeatable.

```python
m = LinearRegression(optimizer=SGD()).fit(X, y, epochs=5, shuffle=True, seed=0)
print(round(m.predict_one({"a": 1.0, "b": 1.0}), 2))
```

```
1.0
```

## 3. Algorithms without a label, and timestamps

Without `y` the call feeds `learn_one(x)`, which is what the statistics, the detectors and the transformers need. A list `t` with one timestamp per row is passed row by row and needs a single pass, because the time of a row cannot go back.

## Details

### What `fit` does and does not do
`fit` is defined once on the base class `Root`, so every estimator of the library has it. It adds an offline way in and changes nothing that exists: `learn_one`, `learn_many` and every algorithm stay as they were. It is a loop over `learn_one`; the classic batch version of each algorithm, written from its original paper, is the next step of the program (row 24 of the master program), one family at a time.

### Arguments
`X` is a list of rows, `y` a list of labels of the same length, `t` a list of timestamps of the same length, `epochs` an integer of at least 1, `shuffle` a boolean, `seed` an integer. A mismatch of lengths, `epochs < 1`, or `t` with more than one epoch raises `ValueError`; an object without `learn_one` raises `TypeError`.
