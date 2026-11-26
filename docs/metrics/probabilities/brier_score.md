# Brier score

The Brier score is the mean squared distance between probability and outcome. It accumulates $(1-p)^2$ when the true label is 1 and $p^2$ when it is 0, then returns the mean.

~~~python
from dense_armor.core.evaluation.metrics import BrierScore

br = BrierScore()
for y, p in [(1, 0.9), (0, 0.2), (1, 0.6)]:
    br.update(y, p)
print(round(br.get(), 4))
~~~

```
0.07
```

The Brier score is the mean squared distance between probability and outcome: $(0.1^2 + 0.2^2 + 0.4^2)/3 = 0.07$. Lower is better.

## Details

- **Base implementation:** `BrierScore` inherits from `Metric` and tracks the sum of scores and the count `n`.
- **Merge:** Two `BrierScore` metrics with sums `s1, n1` and `s2, n2` merge to `s1+s2, n1+n2`.
- **Revert:** When `revert(y, p)` is called, the score is subtracted from the sum. Used by `Rolling` windows.
