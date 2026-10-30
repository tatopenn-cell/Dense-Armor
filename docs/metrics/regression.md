# Metrics: regression and uncertainty

## 4. Regression on a joint

A regressor predicts a number, for example the torque a joint needs at a given velocity. Here a
recursive least-squares model learns $\tau = 2\,\dot q$ from noisy samples, and three metrics
score it before each update.

```python
import numpy as np
from dense_armor.utility.learn.online_dynamics import RecursiveLeastSquares
from dense_armor.utility.metrics import MeanAbsoluteError, R2Score, RootMeanSquaredError

rng = np.random.default_rng(0)
rls, ms = RecursiveLeastSquares(), [MeanAbsoluteError(), RootMeanSquaredError(), R2Score()]
for qd in rng.normal(0, 1, 500):
    tau = 2.0 * qd + rng.normal(0, 0.1)
    for m in ms:
        m.update(tau, rls.predict_one({"qd": qd}))
    rls.learn_one({"qd": qd}, tau)
print([round(m.get(), 3) for m in ms])
```

```
[0.077, 0.098, 0.998]
```

The **MAE** is the mean of $|\tau - \hat\tau|$, the **RMSE** the square root of the mean of
$(\tau - \hat\tau)^2$, both in the units of the torque. They sit near the noise level: for noise
with standard deviation 0.1 the RMSE tends to 0.1 and the MAE to $0.1\sqrt{2/\pi} = 0.080$.
$R^2 = 1 - \sum(\tau - \hat\tau)^2 / \sum(\tau - \bar\tau)^2$ compares the model with always
predicting the mean: 0.998 means the model explains almost all the variation.

A robot arm has several joints. Pass the vector and the metric reports each joint and the mean.

```python
from dense_armor.utility.metrics import MeanAbsoluteError

m = MeanAbsoluteError()
m.update([1.0, 2.0], [1.5, 2.5])
m.update([2.0, 4.0], [2.5, 3.0])
print(m.get())
```

```
{'per_joint': [0.5, 0.75], 'mean': 0.625}
```

Joint 0 is off by 0.5 both times; joint 1 by 0.5 and then 1.0, mean 0.75.

## 5. Scoring the uncertainty

A prediction with an interval is useful only if the interval is honest. Two numbers check it:
how often the truth falls inside (**coverage**) and how wide the interval is (**width**).

```python
import numpy as np
from dense_armor.roles import AdaptiveConformalRegressor
from dense_armor.utility.learn.online_dynamics import RecursiveLeastSquares
from dense_armor.utility.metrics import IntervalCoverage, MeanIntervalWidth

rng = np.random.default_rng(0)
acr, cov, wid = AdaptiveConformalRegressor(RecursiveLeastSquares(), alpha=0.1), IntervalCoverage(), MeanIntervalWidth()
for qd in rng.normal(0, 1, 2000):
    x, tau = {"qd": qd}, 2.0 * qd + rng.normal(0, 0.1)
    cov.update(tau, acr.predict_interval(x))
    wid.update(tau, acr.predict_interval(x))
    acr.learn_one(x, tau)
print(round(cov.get(), 3), round(wid.get(), 3))
```

```
0.9 0.34
```

The interval was asked for 90 % coverage (`alpha=0.1`) and covers 90 % of the torques. Its mean
width 0.34 is close to the narrowest honest 90 % interval for Gaussian noise of standard
deviation 0.1, which is $2 \cdot 1.645 \cdot 0.1 = 0.33$.

When the prediction is a mean and a variance, the **Gaussian negative log-likelihood** scores both
at once.

```python
from dense_armor.roles import Estimate
from dense_armor.utility.metrics import GaussianNLL

m = GaussianNLL()
m.update(0.1, Estimate(mean=0.0, var=0.01))
print(round(m.get(), 4))
```

```
-0.8836
```

$$\mathrm{NLL} = \tfrac12 \ln(2\pi v) + \frac{(y - \mu)^2}{2v} = \tfrac12 \ln(0.0628) + \tfrac12
= -0.8836.$$

A variance that is too small makes the second term explode; one that is too large makes the
first term grow. The minimum is at the true spread.

## float32

`R2Score(dtype=numpy.float32)` keeps the Welford state and the
`ss_res` sum in float32 and applies the shifted-data algorithm with
``K = y_true[0]`` (Healy 2022, Algorithm 2) online: TwoSum on ``s1``,
``s2`` and ``ss_res``, two-term formulas for the mean and for ``M2``.
Relative error against the float64 reference on N(1e3, 0.01), 100,000
samples, seed 0: ``M2`` 1.7e-8, ``ss_res`` 3.7e-8. On a constant target
``get()`` returns 0.0. With this mode ``w`` must be 1. `R2Score` is a
`Metric`, not an `Estimator`: `check_estimator` does not apply.

Cost, median of 7 runs of one update over 20,000 samples, one CPU thread: default 0.001300 ms (IQR 0.000012), float32 0.009759 ms (IQR 0.000077). The claim that the float32 path is at
most as slow as the default is not met: on a scalar loop the numpy
float32 arithmetic is slower than the Python float64 arithmetic. The
float32 speed of these classes is not measured; no class in this batch
has a `learn_many` on which the float32 blocks of the T3 kernel could
be timed.
        `R2Score.merge` with `dtype=numpy.float32` on either side raises
        `NotImplementedError`: the pairwise formula runs on the stored
        scalars, whose shift differs between the two partial states, so
        the combined value would carry the accumulated float32 error
        (about 5.5e-5 on a split N(1e3, 0.01) stream) instead of the
        single-pass 1.7e-8. The default path merges without this limit.

