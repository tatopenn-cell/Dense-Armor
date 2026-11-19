# Linear models: first-order online learning for regression and classification

These are prediction tools that learn from data one example at a time, without a huge offline database. One predicts a real number by minimizing squared errors, another gives the probability of one of two answers, and a third also says how sure it is; how each one learns depends on the optimizer you plug into it.

## The map of the linear models

Click a page to open it, or the job on the left (or the button at the top) to go back to the [map of the library](../index.md#the-library-folder-by-folder).

<object type="image/svg+xml" data="../../assets/maps/linear.svg" style="width:100%"></object>

## A first example: the fee of a payment

A bank charges a fee on every payment: 2 for every 100 of the amount, plus a fixed 0.50 (the amounts are in hundreds). Nobody tells the model the rule: it sees the payments one at a time, each with its fee, and learns the line. After 500 payments it is asked the fee of a payment of 250 (the real fee is 5.50).

```python
import numpy as np
from dense_armor.core.online.linear.glm import LinearRegression
from dense_armor.core.online.optim import SGD

rng = np.random.default_rng(0)
reg = LinearRegression(optimizer=SGD(lr=0.05))
for _ in range(500):
    amount = rng.uniform(0.1, 5.0)
    fee = 2.0 * amount + 0.5 + rng.normal(0, 0.05)
    reg.learn_one({"amount": amount}, fee)
print(round(reg.predict_one({"amount": 2.5}), 2))
```

```
5.5
```

The model is `LinearRegression`: a weighted sum of the inputs, with a constant term that learns the fixed 0.50. The learning rate is not in the model but in the optimizer (`SGD`).

## The pages of the section

| Page | What it is for |
|---|---|
| [GLM](glm.md) | `LinearRegression` (a real number) and `LogisticRegression` (the probability of a class), with any optimizer |
| [The input shape](glm_input_shape.md) | what a sample looks like: the order of the features, the prediction before the update, the missing values |
| [The learning rate](glm_learning_rate.md) | why the learning rate belongs to the optimizer, and what happens when it is too large |
| [Bayesian linear regression](bayes.md) | a line with its doubt: a prediction and how sure it is |
| [Payload estimator](payload.md) | the mass of an object that a robot arm is carrying, while it moves |

The optimizers have their own section: [Optimizers](../optim/index.md).
