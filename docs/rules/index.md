# Rules that learn a number

## The map of rules

Click a page to open it, or the job on the left (or the button at the top) to go back to the [map of the library](../index.md#the-library-folder-by-folder).

<object type="image/svg+xml" data="../assets/maps/rules.svg" style="width:100%"></object>

## Gamestory

### The Lighthouse

Once upon a time there was a lighthouse keeper in the middle of the sea who had to guess how high the water level would be every morning (AMRulesRegressor), looking only at the clouds and the wind.

At first he knew nothing and had written on a sheet of paper a single giant rule that applied to every day (default rule). But after observing the sea for a hundred mornings in a row (n_min), he noticed a trick that was one hundred percent sure (thanks to a special compass called Hoeffding bound that prevented him from guessing at random): when the wind blew from the north, the water always rose by five meters! So he added that new rule to his sheet, becoming very good at making predictions.

One day, however, a sudden storm changed the sea currents forever. The keeper's old rule began to fail badly, but luckily on the table an alarm bell rang immediately (Page-Hinkley test), which warned every time a rule no longer worked. Without losing a second, the keeper tore up the old sheet and wrote a new one for the new currents. And in order never to be confused by the craziest days, he used one last secret: he ignored the waves that were too strange (anomaly_threshold) and kept on the table only very few easy-to-read slips of paper (max_rules), thus guessing the height of the sea every single day.

## A first example: the slippage of an order

A desk predicts the slippage of an order, in basis points, from its size and the spread, with rules that it can read.

```python
import numpy as np
from dense_armor.core.online.rules import AMRulesRegressor

r = np.random.default_rng(0)
m = AMRulesRegressor(n_min=100, seed=0)
for _ in range(1500):
    s = float(r.uniform(-1, 1))
    m.learn_one({"size": s, "spread": float(r.uniform(-1, 1))}, 5.0 * (s > 0))
print(round(m.predict_one({"size": 0.5, "spread": 0.0}), 1), round(m.predict_one({"size": -0.5, "spread": 0.0}), 1))
print(m.explain_one({"size": 0.5, "spread": 0.0}))
```

```
5.0 0.0
[[{'feature': 'size', 'op': '>', 'value': -5.435268394649029e-05}]]
```

The slippage is 5 when the size is above 0 and 0 otherwise; the spread says nothing. The model answers 5.0 for a positive size and 0.0 for a negative one. `explain_one` lists the conditions of the rule that covers the first order: one condition, on `size`, close to 0. The spread is not in it.

## The pages of the section

| Page | What it shows |
|---|---|
| [Learn a rule set](learn.md) | the default rule, `n_min`, the margin of certainty, the new rule and who answers |
| [When the world changes](change.md) | the Page-Hinkley alarm that removes a rule |
| [Options](options.md) | the strange orders, the number of rules, the way a rule predicts, the parameters |

## The words

| Word | What it means |
|---|---|
| rule | conditions on the features, and the number to answer when they hold |
| default rule | the rule with no conditions: it learns the orders that no other rule covers |
| `n_min` | orders a rule sees before it asks for a cut |
| Page-Hinkley | the test that removes a rule whose error keeps rising |
| `anomaly_threshold` | the score under which an order is ignored by the rule that covers it |

## The same on a robot

A robot predicts the torque of a joint, in newton-metres, from its speed and its load.

```python
import numpy as np
from dense_armor.core.online.rules import AMRulesRegressor

r = np.random.default_rng(1)
m = AMRulesRegressor(n_min=100, seed=0)
for _ in range(1500):
    v = float(r.uniform(-1, 1))
    m.learn_one({"speed": v, "load": float(r.uniform(-1, 1))}, 5.0 * (v > 0))
print(round(m.predict_one({"speed": 0.5, "load": 0.0}), 1), round(m.predict_one({"speed": -0.5, "load": 0.0}), 1))
print(m.explain_one({"speed": 0.5, "load": 0.0}))
```

```
5.0 0.0
[[{'feature': 'speed', 'op': '>', 'value': -0.0030808348674327135}, {'feature': 'speed', 'op': '>', 'value': 0.005520971026271582}]]
```

The torque is 5 when the speed is above 0 and 0 otherwise; the load says nothing. The model answers 5.0 and 0.0, and the conditions that cover the first sample are both on `speed`.

## Details

### References

- Duarte, J., Gama, J., Bifet, A. (2016). Adaptive model rules from high-speed data streams. ACM Transactions on Knowledge Discovery from Data 10(3), article 30. Algorithms 1 and 2 (pages 6 and 7 of the PDF), Sections 3.3 to 3.5 (pages 8 to 10), parameters on pages 11 and 12.

### Where the paper and the code differ

- **The expansion test.** Algorithm 2 prints `VR(Xa) - VR(Xb) > eps`; the text on page 7 compares the ratio of the two variance ratios with `1 - eps`. The ratio is implemented.
- **The delta rule.** The paper writes `w <- w + eta (y_hat - y) x`, which moves the weights up the error. The descent step with a minus sign is implemented, and the linear model converges on a plane (mean absolute error below 0.2 on `3a - 2b`).
- **Choices the paper leaves open.** `learning_rate = 0.01`; the side of a split with the smaller target variance becomes the condition, and the other side seeds the new default rule; the split statistics are a sorted list of at most 50 values per feature and rule, a sample joining the nearest value once it is full; probabilities of the anomaly score are clipped to `[1e-12, 1 - 1e-12]`; the anomaly test is off by default; `max_rules = 100` and `max_literals = 20` bound the model.

### Cost and memory

p99 of one `learn_one` plus one `predict_one` over 3000 samples with four features, measured on a desktop CPU: 0.31 ms (median 0.10 ms); `budget_s` is 0.01 s. A model has at most `max_rules` rules, each with at most `max_literals` conditions and `max_split_points` stored values per feature, so the state is constant in the length of the stream. There is no pure `init_state` / `step` form: the set of rules changes shape.

### Checks

- `check_estimator` passes with no check skipped; the tests cover every line of the module.
- The variance ratio of a hand case (targets 0, 2, 4, 6 split after the second) is 0.8, and 1.0 for a perfect split.
- Two identical features tie: the tie threshold `tau` decides the split only once `n` is large enough that the Hoeffding bound falls below it (3224 samples for the defaults, `ln(1/delta) / (2 tau^2) = 3223.6`).

### The text of the older page

Suppose an AI assistant must predict how many seconds a request will take from a few numbers about it. A rule could read: *if the prompt length is above 300 and the queue is shorter than 4, the answer is about 2.1 seconds*. The part after "if" is a list of conditions on the features; the part after "then" is a small model that predicts the target for the requests the rule covers.

`AMRulesRegressor` (Duarte, Gama and Bifet 2016) keeps such a list and learns it one sample at a time.

- **Ordered or unordered.** With `ordered_set=True` the first rule that covers a sample learns and predicts; by default all covering rules learn and their predictions are averaged.
- **How a rule predicts.** `prediction="mean"` uses the mean of its targets, `"perceptron"` a linear model on the standardised features, `"adaptive"` (default) the one with the lower fading mean absolute error.
- **Anomalies.** With `anomaly_threshold` set (the paper uses -0.75), a rule ignores a sample whose features are unusual for it, once it has seen `m_min` samples. The score is the mean over features, so it flags a sample that is unusual in most features, not in one of them.
- **Size.** `max_rules`, `max_literals` and `max_split_points` bound the model.
