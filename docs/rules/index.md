# Rules that learn a number

## The map of rules

Click a page to open it, or the job on the left (or the button at the top) to go back to the [map of the library](../index.md#the-library-folder-by-folder).

<object type="image/svg+xml" data="../assets/maps/rules.svg" style="width:100%"></object>

## Gamestory

### The Lighthouse

Once upon a time there was a lighthouse keeper in the middle of the sea who had to guess how high the water level would be every morning (AMRulesRegressor), looking only at the clouds and the wind.

At first he knew nothing and had written on a sheet of paper a single giant rule that applied to every day (default rule). But after observing the sea for a hundred mornings in a row (n_min), he noticed a trick that was one hundred percent sure (thanks to a special compass called Hoeffding bound that prevented him from guessing at random): when the wind blew from the north, the water always rose by five meters! So he added that new rule to his sheet, becoming very good at making predictions.

One day, however, a sudden storm changed the sea currents forever. The keeper's old rule began to fail badly, but luckily on the table an alarm bell rang immediately (Page-Hinkley test), which warned every time a rule no longer worked. Without losing a second, the keeper tore up the old sheet and wrote a new one for the new currents. And in order never to be confused by the craziest days, he used one last secret: he ignored the waves that were too strange (anomaly_threshold) and kept on the table only very few easy-to-read slips of paper (max_rules), thus guessing the height of the sea every single day.

## The problem, in one line

A rule says "if these conditions hold, the answer is about this number". A set of rules is a model you can read: each rule names the features it looks at and the range where it applies. These rules start with one rule that covers everything, split it into narrower rules as the data shows where the answer changes, and throw away a rule whose predictions start to fail.

A desk that wants to predict the slippage of an order wants a model it can read. A **rule** says "if these conditions hold, the answer is about this number", and a set of rules is a model that names the features it looks at and the range where each rule applies. These rules start with one default rule that covers everything and learn one sample at a time, so they follow the stream. `AMRulesRegressor` (Duarte, Gama and Bifet 2016) is the class that does it.

## The words

| Word | What it means here | In the example |
|---|---|---|
| rule | conditions on the features, and the number to answer when they hold | "if the size is above 0, about 5" |
| default rule | the rule that covers every sample, before any cut | the answer is the mean slippage |
| expand | add a condition to a rule, after `n_min` samples | the rule is cut at a size of 0 |
| Page-Hinkley | the test that signals when the error of a rule rises | the rule is dropped when the market changes |

## A first example: the slippage of an order

The slippage is 5 basis points when the size of the order is above 0 and 0 otherwise; the spread and the volatility say nothing. The model sees 1500 orders, and then answers two orders of opposite size and says which conditions its rule found.

```python
import numpy as np
from dense_armor.core.online.rules import AMRulesRegressor

rng = np.random.default_rng(0)
model = AMRulesRegressor(n_min=100, seed=0)
for _ in range(1500):
    size = float(rng.uniform(-1.0, 1.0))
    row = {"size": size, "spread": float(rng.uniform(-1.0, 1.0)), "vol": float(rng.uniform(-1.0, 1.0))}
    model.learn_one(row, 5.0 * (size > 0.0) + 0.1 * float(rng.normal()))
print(round(model.predict_one({"size": 0.5, "spread": 0.0, "vol": 0.0}), 1), round(model.predict_one({"size": -0.5, "spread": 0.0, "vol": 0.0}), 1))
print(model.explain_one({"size": 0.5, "spread": 0.0, "vol": 0.0}))
```

```
5.0 -0.0
[[{'feature': 'size', 'op': '>', 'value': 0.03862887620925903}]]
```

The answer is 5.0 for the positive size and -0.0, which is zero, for the negative one, with a cut at a size of about 0.04 (the slippage changes at 0) and no mention of the spread or the volatility. The pages below take each part in turn.

## The pages of the section

| Page | What it shows |
|---|---|
| [Learn a rule set](learn.md) | how the rules are cut, `n_min`, and `explain_one`, with the toy stream of the old page |
| [When the world changes](change.md) | the Page-Hinkley test that drops a rule when its error rises, with the older figure |
| [Options](options.md) | how a rule predicts, how many rules, and the parameters |

## The same problem on a robot

A robot that predicts the torque of a joint from its speed, its load and its temperature reads its rules in the same way, and each page ends with the same code written for a robot.

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
