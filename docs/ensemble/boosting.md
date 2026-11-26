# Boosting: weight the samples the members miss

[← Ensembles](index.md)

`OnlineBoostingClassifier` follows the same paper, Section 3 and Figure 2. Each member has two
counters, one for the samples it classified correctly and one for the wrong ones. The weight of
the incoming sample for the next member grows with the fraction of its mistakes, so a sample a
member got wrong reaches the next member more often. The final vote weights each member by
`log(1 / beta_m)`, with `beta_m = eps_m / (1 - eps_m)` and `eps_m` the error rate of that member.

```python
import numpy as np
from dense_armor.core.online.ensemble import OnlineBoostingClassifier
from dense_armor.core.online.tree.hoeffding import HoeffdingTreeClassifier

rng = np.random.default_rng(0)
boost = OnlineBoostingClassifier(
    lambda: HoeffdingTreeClassifier(grace_period=20), n_models=5, seed=1)
for _ in range(500):
    x = float(rng.normal(0.0 if rng.random() < 0.5 else 4.0, 0.5))
    boost.learn_one({"x": x}, int(x > 2.0))
print(boost.predict_one({"x": 0.1}), boost.predict_one({"x": 4.0}))
```

```
0 1
```

Boosting is the strongest of the four ensembles of this page on the drifted stream: 0.982
against 0.942 for the single tree, on 19 seeds out of 20 (the cost section of the [index](index.md#what-the-ensembles-cost)). The price is the highest
cost per sample of the four, because every member trains on every sample with a weight, and the
weight depends on the mistakes of the member before it.

---

## Details

### Source

Oza, N. C., Russell, S. (2001). Online bagging and boosting. In Artificial Intelligence and Statistics 2001, 105-112. Section 3 and Figure 2 for the boosting weight update.

The boosting update is the one of Figure 2 of Oza and Russell: the weight of the sample for the
next member is multiplied by `N / (2 * lambda_s)`, with `N` the number of samples seen so far
and `lambda_s` the accumulator of the member for the samples it classified correctly or wrongly.
The paper gives no equivalence and no convergence rate for the boosting weights: it builds the
algorithm to correspond to batch AdaBoost.M1 and then shows, with experiments, that online
boosting starts worse and approaches AdaBoost as the number of examples grows (Figure 4).
