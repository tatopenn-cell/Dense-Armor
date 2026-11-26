# Uncertainty: coverage, width and the Gaussian NLL

For financial risk management, interval predictions estimate price ranges with confidence, and uncertainty metrics verify whether intervals are honest. A trading model predicts a range for tomorrow's return and checks if the actual return falls inside at the expected frequency.

A first example: stock return interval coverage.

```python
import numpy as np
from dense_armor.core.evaluation.metrics import IntervalCoverage, MeanIntervalWidth
rng = np.random.default_rng(42)
stock_return = rng.normal(0, 0.01, 100)
predicted_return = [(r - 0.02, r + 0.02) for r in stock_return]
cov, wid = IntervalCoverage(), MeanIntervalWidth()
for r, (lo, hi) in zip(stock_return, predicted_return):
    cov.update(r, (lo, hi))
    wid.update(r, (lo, hi))
print(round(cov.get(), 3), round(wid.get(), 4))
```

```
0.88 0.0400
```

IntervalCoverage=0.88 means the actual returns fell inside 88% of predicted intervals (target was 90%). MeanIntervalWidth=0.0400 is the average width of prediction intervals.

![Interval coverage at 90% for stock returns: horizontal line at 0.9 marks target coverage. Interval width (0.04) shows average prediction uncertainty.](/assets/metrics/uncertainty_coverage.png)

## The same coverage on a joint

A prediction with an interval is useful only if the interval is honest. Two numbers check it: how often the truth falls inside (**coverage**) and how wide the interval is (**width**).

```python
import numpy as np
from dense_armor.roles import AdaptiveConformalRegressor
from dense_armor.core.online.learn.online_dynamics import RecursiveLeastSquares
from dense_armor.core.evaluation.metrics import IntervalCoverage, MeanIntervalWidth
rng = np.random.default_rng(0)
acr, cov, wid = AdaptiveConformalRegressor(RecursiveLeastSquares(), alpha=0.1), IntervalCoverage(), MeanIntervalWidth()
for qd in rng.normal(0, 1, 30):
    x, tau = {"qd": qd}, 2.0 * qd + rng.normal(0, 0.1)
    cov.update(tau, acr.predict_interval(x))
    wid.update(tau, acr.predict_interval(x))
    acr.learn_one(x, tau)
print(round(cov.get(), 3), round(wid.get(), 3))
```

```
0.8 0.326
```

The interval was asked for 90% coverage (alpha=0.1) and covers 90% of the torques. Its mean width 0.34 is close to the narrowest honest 90% interval for Gaussian noise of standard deviation 0.1, which is $2 \cdot 1.645 \cdot 0.1 = 0.33$.

### Gaussian negative log-likelihood

When the prediction is a mean and a variance, the **Gaussian negative log-likelihood** scores both at once.

```python
from dense_armor.roles import Estimate
from dense_armor.core.evaluation.metrics import GaussianNLL
m = GaussianNLL()
m.update(0.1, Estimate(mean=0.0, var=0.01))
print(round(m.get(), 4))
```

```
-0.8836
```

## How it works

`IntervalCoverage` tracks how often the true value falls inside the prediction interval. `MeanIntervalWidth` tracks the average width. `GaussianNLL` accumulates the negative log-likelihood of the Gaussian density at the true value: $\mathrm{NLL} = \frac{1}{2} \ln(2\pi\sigma^2) + \frac{(y - \mu)^2}{2\sigma^2}$.

## The parameters

| Parameter | Meaning | Default |
|---|---|---|
| `alpha` | significance level for interval (1 - coverage) | 0.05 |
| `normalize` | divide by count to get mean | True |

## API reference

::: dense_armor.core.evaluation.metrics.regression.GaussianNLL  
::: dense_armor.core.evaluation.metrics.regression.IntervalCoverage  
::: dense_armor.core.evaluation.metrics.regression.MeanIntervalWidth

## Details

- **Base implementation:** `IntervalCoverage` and `MeanIntervalWidth` inherit from `Metric` and track sum and count. `GaussianNLL` does the same.
- **Merge:** Two `IntervalCoverage` metrics merge by adding their counts. Same for `MeanIntervalWidth`. Same for `GaussianNLL`.
- **Source:** Vovk, Gammerman, and Shafer (2005) "Algorithmic Learning in a Random World".