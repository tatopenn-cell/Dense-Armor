# MetricLearner: the base of the online metric learners

[← Metric learning](metric_learning.md)

A bank wants to know which past customers look like a new one. Looking alike is a distance, and one customer on the table says nothing about a distance: to measure one, there must be at least two. So a metric learner does not read rows `(x, y)` and does not predict a class. Its unit is a **pair** of customers (near or far) or a **triplet** (a customer, one like it and one unlike it), and what it learns is a matrix that gives the distance between two customers, for another algorithm to use, a k-NN for example. `MetricLearner` is the base class of the three algorithms of this section, `OASIS`, `LEGO` and `POLA`: it fixes what they share, the way they take their data and the way they give the distance.

## 1. A first example: two customers alike, one different

The customers have an age and a monthly spending. `a` and `b` are alike (about 30 years old, about 500 of spending), and `c` is different (55 years old). `LEGO` learns from pairs with a target squared distance: a small one (0.05) for `a` with `b`, a large one (4.0) for `a` with `c`. `POLA` learns from pairs labelled +1 for alike and -1 for different. Both read the same pairs 20 times, one pair at a time.

```python
from dense_armor.core.online.learn.metric_learning import LEGO, POLA

a = {"age": 30.0, "spend": 500.0}
b = {"age": 31.0, "spend": 520.0}
c = {"age": 55.0, "spend": 480.0}
lego, pola = LEGO(), POLA()
print("before", round(lego.distance(a, b), 1), round(lego.distance(a, c), 1))
for _ in range(20):
    lego.learn_pair(a, b, 0.05)
    lego.learn_pair(a, c, 4.0)
    pola.learn_pair(a, b, +1)
    pola.learn_pair(a, c, -1)
print("LEGO", round(lego.distance(a, b), 3), round(lego.distance(a, c), 3))
print("POLA", round(pola.distance(a, b), 3), round(pola.distance(a, c), 3))
```

```
before 20.0 32.0
LEGO 0.731 2.0
POLA 0.186 1.414
```

Before the learning, the distance is the plain Euclidean one, dominated by the spending: `a` is 20 from `b` and 32 from `c`. After 20 rounds, `LEGO` puts `a` at 0.731 from `b` and at 2.0 from `c`, and `POLA` at 0.186 and 1.414: both have moved the pair that is alike near and kept the pair that is different far. The target of 0.05 is not reached in 20 rounds: the distance moves toward the targets one pair at a time.

| class or function | what it does |
|---|---|
| `MetricLearner` | the base of the three: the pair, the triplet and the distance |
| `learn_pair(x, x2, y)` | one update from a pair, with a label or a target distance |
| `learn_triplet(x, x_pos, x_neg)` | one update from a triplet: a sample, one like it and one unlike it |
| `distance(x, x2)` | the learned distance between two samples |

## 2. What it does

A metric learner is trained from pairs or triplets and exposes a `distance(x, x2)` method. Unlike the `Classifier`, `Regressor` and `Transformer`, it learns from relative information (two or three samples at a time) and does not predict labels or transform features directly. Subclasses implement `learn_triplet`, `learn_pair` and `distance`. Features are dicts; the first call fixes the feature order (sorted keys), and any feature not seen at that time counts as 0 in later calls.

There is no `learn_one(x, y)`, and it is not an omission: a single sample does not contain the relation that a distance is learned from, and there is no prediction that would pair with it, because a metric learner updates a matrix and not a class. A stream of rows can still feed a metric learner, with the pairs or the triplets built from a window of the latest samples: that is what [`MetricKNNClassifier`](metric_knn.md) does.

**Source:** the base class has no paper of its own; the algorithms are in the pages [OASIS](oasis.md), [LEGO](lego.md) and [POLA](pola.md).

## 3. The parameters

`MetricLearner` takes no parameter. Each algorithm has its own, on its page: `C` for `OASIS`, `eta` for `LEGO` and `b_init` for `POLA`.

## 4. The data of the toy example

The older example of the page, kept as it was: two groups of two-dimensional samples.

A stream of samples, one per customer. Each sample is a 2-D vector, with the label of its group, A or B.

```python
import numpy as np

rng = np.random.default_rng(42)
group_a = rng.normal([1.0, 0.0], 0.1, (200, 2))
group_b = rng.normal([0.0, 1.0], 0.1, (200, 2))
X = np.vstack([group_a, group_b])
y = np.array(["A"] * 200 + ["B"] * 200)
```

Euclidean distance on this data works, because the clusters are far apart. On a real
real data they are often not.

## 5. The same on a robot

A robot has two joint angles, `q1` and `q2`. Two free-moving states are alike, a contact state is different. The code is the same, with the names of the robot.

```python
from dense_armor.core.online.learn.metric_learning import LEGO

free = {"q1": 0.10, "q2": 0.20}
free2 = {"q1": 0.12, "q2": 0.21}
contact = {"q1": 0.90, "q2": 0.80}
lego = LEGO()
for _ in range(20):
    lego.learn_pair(free, free2, 0.05)
    lego.learn_pair(free, contact, 4.0)
print(round(lego.distance(free, free2), 3), round(lego.distance(free, contact), 3))
```

```
0.044 2.0
```

The two free states are near and the contact state is far.

## API reference

::: dense_armor.core.online.learn.metric_learning.MetricLearner

---

## Details

### Choices

- **The numbers** of the first example are printed by the code above, with the default parameters of `LEGO` and `POLA`.
- **The detail that was on the old page** (the toy data) is kept in section 4.
