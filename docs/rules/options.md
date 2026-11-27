# Options: how a rule predicts, and how big the model can grow

[← Rules](index.md)

A desk can choose how each rule predicts (the mean of its targets, a small linear model, or the one that is doing better), whether the rules learn in order or all together, and how big the model can grow. The defaults are the ones of the paper, except where the page says so. A robot with limited memory sets the size of the model first.

![The mean absolute error of the slippage over 3000 orders, for the largest number of rules from 1 to 20.](../assets/rules/pages/rules_options_3d.png)

## 1. A first example: three ways to predict

The same stream of orders as on the first page, with the three values of `prediction`. The printed number is the mean absolute error over the last 1000 of 3000 orders.

```python
import numpy as np
from dense_armor.core.online.rules import AMRulesRegressor

for mode in ("adaptive", "mean", "perceptron"):
    rng = np.random.default_rng(0)
    model = AMRulesRegressor(n_min=100, prediction=mode, seed=0)
    err = []
    for _ in range(3000):
        size = float(rng.uniform(-1.0, 1.0))
        row = {"size": size, "spread": float(rng.uniform(-1.0, 1.0)), "vol": float(rng.uniform(-1.0, 1.0))}
        y = 5.0 * (size > 0.0) + 0.1 * float(rng.normal())
        err.append(abs(model.predict_one(row) - y))
        model.learn_one(row, y)
    print(mode, round(float(np.mean(err[-1000:])), 3))
```

```
adaptive 0.266
mean 0.266
perceptron 0.284
```

On this stream the three ways are close: the target is a step, and a rule that predicts the mean of its targets is already right.

![The mean absolute error for the three ways a rule predicts.](../assets/rules/pages/rules_options_finance.png)

Over 3 streams, the median error in the blocks ending at 100, 500, 1000 and 3000 orders reads 2.217, 0.355, 0.136 and 0.355 for `adaptive` (the default), 2.552, 0.355, 0.136 and 0.355 for `mean`, and 2.218, 3.824, 0.344 and 0.502 for `perceptron`: the linear model is much worse in the first 500 orders. On the surface, the largest number of rules `max_rules` of 1 gives 0.537, 0.684 and 1.02 in the blocks ending at 500, 1000 and 3000 orders, 5 gives 0.355, 0.136 and 0.834, and 20 gives 0.355, 0.136 and 0.355: a model that is allowed too few rules keeps its error up.

| class or function | what it does |
|---|---|
| `AMRulesRegressor` | the same model, with the options below |

## 2. What it does

- **Ordered or unordered.** With `ordered_set=True` the first rule that covers a sample learns and predicts; by default all covering rules learn and their predictions are averaged.
- **How a rule predicts.** `prediction="mean"` uses the mean of its targets, `"perceptron"` a linear model on the standardised features, `"adaptive"` (default) the one with the lower fading mean absolute error.
- **Anomalies.** With `anomaly_threshold` set (the paper uses -0.75), a rule ignores a sample whose features are unusual for it, once it has seen `m_min` samples. The score is the mean over features, so it flags a sample that is unusual in most features, not in one of them.
- **Size.** `max_rules`, `max_literals` and `max_split_points` bound the model.

**Source:** Duarte, Gama, Bifet (2016), Algorithms 1 and 2, Sections 3.1 to 3.5.

## 3. The parameters

| Parameter | Meaning | Default |
|---|---|---|
| `ordered_set` | `True` learns and predicts with the first rule that covers a sample; `False` with all of them (a choice of the library) | `False` |
| `prediction` | `"adaptive"`, `"mean"` or `"perceptron"` | `"adaptive"` |
| `learning_rate` | step of the delta rule of the linear models | 0.01 |
| `anomaly_threshold` | a sample below the threshold is ignored by the rule that finds it anomalous; `None` turns the test off (a choice of the library) | `None` |
| `m_min` | samples a rule needs before it checks anomalies | 30 |
| `max_split_points` | split values kept per feature and rule | 50 |
| `max_rules` | largest number of rules | 100 |
| `max_literals` | largest number of conditions of a rule | 20 |

The constructor raises `ValueError` on a non-positive `n_min`, `learning_rate`, `ph_lambda`, `max_rules`, `max_literals` or `max_split_points`, on `delta` or `alpha` outside (0, 1), a negative `tau`, `ph_phi` or `m_min`, or an unknown `prediction`. `budget_s` is 0.01 s and `memory_class` is `O(window)`.

## 4. The same on a robot

A robot has little memory, so it limits the model: at most 5 rules of at most 2 conditions. It predicts the torque of a joint from its speed, its load and its temperature.

```python
import numpy as np
from dense_armor.core.online.rules import AMRulesRegressor

rng = np.random.default_rng(1)
model = AMRulesRegressor(n_min=100, max_rules=5, max_literals=2, seed=0)
for _ in range(1500):
    speed = float(rng.uniform(-1.0, 1.0))
    row = {"speed": speed, "load": float(rng.uniform(-1.0, 1.0)), "temp": float(rng.uniform(-1.0, 1.0))}
    model.learn_one(row, 5.0 * (speed > 0.0) + 0.1 * float(rng.normal()))
print(round(model.predict_one({"speed": 0.5, "load": 0.0, "temp": 0.0}), 1), round(model.predict_one({"speed": -0.5, "load": 0.0, "temp": 0.0}), 1))
```

```
5.0 -0.0
```

The small model gives the same answers as the large one on this stream.

## API reference

::: dense_armor.core.online.rules.amrules.AMRulesRegressor

---

## Details

### Choices

- **The figures** are drawn with the library by `docs/assets/rules/make_rules_style_figures.py`: 3000 orders, `n_min` = 100, `max_rules` or `prediction` on the figure, the error averaged over blocks of 100 orders, median over 3 seeds.
