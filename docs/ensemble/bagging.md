# Bagging: a different sample for every member

[← Ensembles](index.md)

`OnlineBaggingClassifier` follows Oza and Russell (2001), Section 2. Every member sees every
sample, but for each arriving sample it draws `k` from a `Poisson(1)` distribution and trains on
the sample `k` times. On average a sample reaches each member once, and a few samples reach it
twice or not at all, which is what a bootstrap sample of the batch algorithm does in the limit.

```python
import numpy as np
from dense_armor.core.online.ensemble import OnlineBaggingClassifier
from dense_armor.core.online.tree.hoeffding import HoeffdingTreeClassifier

rng = np.random.default_rng(0)
bag = OnlineBaggingClassifier(
    lambda: HoeffdingTreeClassifier(grace_period=20), n_models=5, seed=1)
for _ in range(500):
    x = float(rng.normal(0.0 if rng.random() < 0.5 else 4.0, 0.5))
    bag.learn_one({"x": x}, int(x > 2.0))
print(bag.predict_one({"x": 0.1}), bag.predict_one({"x": 4.0}))
```

```
0 1
```

The Poisson draw is what makes the members differ. On the drifted stream of the cost section of the [index](index.md#what-the-ensembles-cost) the bag of
five trees answers correctly more often than one tree in the median of 20 seeds (0.947 against
0.942, paired p = 0.003). The gain is small on this stream: the members share the same data,
and the diversity the Poisson draw gives is limited.

---

## Details

### Source

Oza, N. C., Russell, S. (2001). Online bagging and boosting. In Artificial Intelligence and Statistics 2001, 105-112. Section 2 for the Poisson sampling.

The
bagging counts are the part for which the paper proves that the expected counts of the online
and batch algorithms are equal.

### Choices

- **The Poisson draw of bagging.** The paper draws `Poisson(1)` per member per sample. The class
  draws one `k` per member and calls `learn_one` `k` times, which is the same rule.
