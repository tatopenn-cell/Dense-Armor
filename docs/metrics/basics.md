# Metrics: pairs, classification and probabilities

## 1. A metric, one pair at a time

A metric reads one pair `(y_true, y_pred)` with `update` and answers `get()` at any moment.

```python
from dense_armor.utility.metrics import Accuracy

m = Accuracy()
for y, p in [(0, 0), (1, 1), (0, 1), (1, 1)]:
    m.update(y, p)
print(m.get(), m.n)
```

```
0.75 4
```

Three of the four predictions are right, so the accuracy is 3/4 = 0.75. The metric keeps two
running sums (hits and total), not the list of pairs, so its memory stays the same after a
million samples. Every metric on this page has the same members:

- `update(y_true, y_pred, t=None, w=1.0)` adds a pair, with an optional timestamp and weight;
- `revert(y_true, y_pred)` removes a pair added before (used by the rolling windows of step 6);
- `merge(other)` combines two metrics computed on different parts of the stream;
- `n` counts the pairs used, `n_missing` the pairs skipped because one side was `None` or `NaN`;
- `works_with(model)` tells whether the metric fits the model's role (classifier or regressor).

## 2. Classification: precision, recall, F1

When the model says "positive", how often is it right (**precision**)? Of the real positives,
how many did it find (**recall**)? **F1** is their harmonic mean.

```python
from dense_armor.utility.metrics import F1, Precision, Recall

y = [1, 0, 1, 1, 0, 1]
p = [1, 1, 1, 0, 0, 1]
ms = [Precision(positive=1), Recall(positive=1), F1(positive=1), F1(average="macro")]
for a, b in zip(y, p):
    for m in ms:
        m.update(a, b)
print([round(m.get(), 3) for m in ms])
```

```
[0.75, 0.75, 0.75, 0.625]
```

Count by hand for class 1: true positives TP = 3 (positions 0, 2, 5), one false positive FP
(position 1), one false negative FN (position 3).

$$P = \frac{TP}{TP + FP} = \frac{3}{4}, \qquad R = \frac{TP}{TP + FN} = \frac{3}{4}, \qquad
F_1 = \frac{2PR}{P + R} = 0.75.$$

`average="macro"` computes F1 for every class and takes the plain mean: class 0 has TP = 1,
FP = 1, FN = 1, so F1 = 0.5, and the macro F1 is (0.75 + 0.5) / 2 = 0.625. `"micro"` pools the
counts of all classes, `"weighted"` weights each class by how often it occurs; `FBeta(beta=2)`
weights recall more than precision.

## 3. Agreement beyond chance and probabilities

Accuracy can look good by chance when one class is common. **Cohen's kappa** subtracts the
agreement expected by chance.

```python
from dense_armor.utility.metrics import CohenKappa

m = CohenKappa()
for a, b in zip([1, 0, 1, 1, 0, 1], [1, 1, 1, 0, 0, 1]):
    m.update(a, b)
print(round(m.get(), 3))
```

```
0.25
```

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

where $p_o$ is the observed agreement (4 of 6) and $p_e$ the agreement two independent guessers
with the same label frequencies would reach: both say 1 four times out of six and 0 twice, so
$p_e = (4 \cdot 4 + 2 \cdot 2)/36 = 0.556$ and $\kappa = (0.667 - 0.556)/(1 - 0.556) = 0.25$.

When the model gives a probability instead of a label, score the probability itself.

```python
from dense_armor.utility.metrics import BrierScore, LogLoss

ll, br = LogLoss(), BrierScore()
for y, p in [(1, 0.9), (0, 0.2), (1, 0.6)]:
    ll.update(y, p)
    br.update(y, p)
print(round(ll.get(), 4), round(br.get(), 4))
```

```
0.2798 0.07
```

The **log-loss** is the mean of $-\ln p$ given to what happened: $-(\ln 0.9 + \ln 0.8 + \ln 0.6)/3
= 0.2798$. The **Brier score** is the mean squared distance between probability and outcome:
$(0.1^2 + 0.2^2 + 0.4^2)/3 = 0.07$. Both are lower for better forecasts.
