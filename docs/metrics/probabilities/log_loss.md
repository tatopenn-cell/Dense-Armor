# Log loss

For a trading model that quotes the default probability of a corporate bond, log loss tells you whether the quoted probability is honest before you lose money. If the model says 5% default risk and the bond defaults, log loss penalizes heavily.

A first example: bond default probability.

```python
from dense_armor.core.evaluation.metrics import LogLoss

bond_default = [0, 1, 0, 0, 1]  # 0 = no default, 1 = default
bond_prob = [0.05, 0.20, 0.03, 0.02, 0.15]  # quoted default probability

ll = LogLoss()
for y, p in zip(bond_default, bond_prob):
    ll.update(y, p)
print(round(ll.get(), 4))
```

```
0.7217
```

Log loss penalizes confident wrong predictions. It accumulates $-\ln p$ when the true label is 1 and $-\ln(1-p)$ when it is 0, then returns the mean.

```python
from dense_armor.core.evaluation.metrics import LogLoss

ll = LogLoss()
for y, p in [(1, 0.9), (0, 0.2), (1, 0.6)]:
    ll.update(y, p)
print(round(ll.get(), 4))
```

```
0.2798
```

The log-loss is the mean of $-\ln p$ given to what happened: $-(\ln 0.9 + \ln 0.8 + \ln 0.6)/3 = 0.2798$. Lower is better.

![Log loss as the probability given to the true class goes from 0.01 to 0.99.](../../assets/metrics/probabilities_log_loss_3d.png)

## Details

- **Base implementation:** `LogLoss` inherits from `Metric` and tracks the sum of scores and the count `n`.
- **Merge:** Two `LogLoss` metrics with sums `s1, n1` and `s2, n2` merge to `s1+s2, n1+n2`.
- **Revert:** When `revert(y, p)` is called, the score is subtracted from the sum. Used by `Rolling` windows.
