# Classifiers and regressors

A supervised tool learns from a sample and its answer. A classifier learns a label and answers with probabilities; a regressor learns a number and answers with a number.

## 1. A classifier that has learned nothing

Before the first `learn_one`, every classifier answers `None` to `predict_one` and an empty dict to `predict_proba_one`. A label that does not exist is never invented.

```python
from dense_armor.core.online.learn.online_classifiers import OnlineGaussianNB

nb = OnlineGaussianNB()
print(nb.predict_one({"a": 1.0}), nb.predict_proba_one({"a": 1.0}))
```

```
None {}
```

## 2. Learn one sample, then predict

`learn_one(x, y)` takes the dict and its label. `predict_proba_one` returns one probability per class that it has seen, and `predict_one` returns the class with the largest one.

```python
from dense_armor.core.online.learn.online_classifiers import OnlineGaussianNB

nb = OnlineGaussianNB()
for a, y in [(0.1, 0), (0.2, 0), (0.9, 1), (1.0, 1), (0.15, 0), (0.95, 1)]:
    nb.learn_one({"a": a}, y)
p = nb.predict_proba_one({"a": 0.9})
print(nb.predict_one({"a": 0.9}), {k: round(v, 3) for k, v in p.items()})
```

```
1 {0: 0.0, 1: 1.0}
```

`predict_one` is built from `predict_proba_one`. A classifier that cannot give probabilities raises `NotImplementedError` with its name when you ask for them.

## 3. Many samples at once

`learn_many`, `predict_many` and `predict_proba_many` take a list of dicts and call the `_one` method for each. A tool may replace them with a faster version, but the answer is the same.

```python
from dense_armor.core.online.learn.online_classifiers import OnlineGaussianNB

nb = OnlineGaussianNB()
nb.learn_many([{"a": 0.1}, {"a": 0.9}, {"a": 0.2}, {"a": 1.0}], [0, 1, 0, 1])
print(nb.predict_many([{"a": 0.15}, {"a": 0.95}]))
```

```
[0, 1]
```

## 4. A regressor

A regressor answers `predict_one(x)` with a number. The target can also be a vector.

```python
import numpy as np
from dense_armor.core.online.tree.hoeffding_regressor import HoeffdingTreeRegressor

rng = np.random.default_rng(0)
r = HoeffdingTreeRegressor(grace_period=30)
for _ in range(600):
    x = float(rng.random())
    r.learn_one({"x": x}, 5.0 * x + float(rng.normal(0, 0.1)))
print(round(r.predict_one({"x": 0.5}), 2))
```

```
2.34
```

## 5. A number and its spread

With `return_std=True` a regressor that supports it answers a pair: the prediction and an estimate of how much it can be off.

```python
import numpy as np
from dense_armor.core.online.tree.hoeffding_regressor import HoeffdingTreeRegressor

rng = np.random.default_rng(0)
r = HoeffdingTreeRegressor(grace_period=30)
for _ in range(600):
    x = float(rng.random())
    r.learn_one({"x": x}, 5.0 * x + float(rng.normal(0, 0.1)))
m, s = r.predict_one({"x": 0.5}, return_std=True)
print(round(m, 2), round(s, 2))
```

```
2.34 0.16
```

Here the spread is the standard deviation of the target in the leaf that the sample reaches. Each regressor defines it in its own way, and its page says how.

## Details

- `Classifier` has the methods `learn_one`, `predict_proba_one`, `predict_one`, `learn_many`, `predict_many`, `predict_proba_many`. `Regressor` has `learn_one`, `predict_one`, `learn_many`, `predict_many`.
- Every method takes an optional time `t`; see [the clock](base.md).
- The files are `dense_armor/roles/classifier.py` and `dense_armor/roles/regressor.py`.
