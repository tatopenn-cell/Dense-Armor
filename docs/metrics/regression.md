# Regression: MAE, MSE, RMSE and R²

For financial predictions like stock prices or revenue forecasts, regression metrics track how well models estimate continuous values over time without storing past data. A trading model can use MAE, RMSE, and R² to evaluate its price predictions as new market data arrives.

A first example: stock price prediction.

```python
import numpy as np
from dense_armor.core.evaluation.metrics import MeanAbsoluteError, RootMeanSquaredError, R2Score
rng = np.random.default_rng(42)
stock_price = 100 + np.cumsum(rng.normal(0, 1, 100))
predicted_price = stock_price + rng.normal(0, 0.5, 100)
mae, rmse, r2 = MeanAbsoluteError(), RootMeanSquaredError(), R2Score()
for p, yp in zip(stock_price, predicted_price):
    mae.update(yp, p); rmse.update(yp, p); r2.update(yp, p)
print([round(mae.get(), 3), round(rmse.get(), 3), round(r2.get(), 3)])
```

```
[0.391, 0.487, 0.954]
```

Count: MAE is the mean of $|p - \hat{p}|$, RMSE is $\sqrt{\text{mean of } (p - \hat{p})^2}$. For prices near 100 with prediction error about 0.5, MAE ≈ 0.39 and RMSE ≈ 0.49. $R^2 = 0.954$ means 95.4% of variance is explained.

![Stock price prediction: true prices (blue) vs predicted prices (red). MAE=0.391, RMSE=0.487, R²=0.954.](/assets/metrics/regression_finance.png)

## The same metrics on a joint

A regressor predicts a number, for example the torque a joint needs at a given velocity. Here a recursive least-squares model learns $\tau = 2\,\dot q$ from noisy samples, and three metrics score it.

```python
import numpy as np
from dense_armor.core.online.learn.online_dynamics import RecursiveLeastSquares
from dense_armor.core.evaluation.metrics import MeanAbsoluteError, R2Score, RootMeanSquaredError
rng = np.random.default_rng(0)
rls, ms = RecursiveLeastSquares(), [MeanAbsoluteError(), RootMeanSquaredError(), R2Score()]
for qd in rng.normal(0, 1, 500):
    tau = 2.0 * qd + rng.normal(0, 0.1)
    for m in ms: m.update(tau, rls.predict_one({"qd": qd}))
    rls.learn_one({"qd": qd}, tau)
print([round(m.get(), 3) for m in ms])
```

```
[0.077, 0.098, 0.998]
```

The **MAE** is the mean of $|\tau - \hat\tau|$, the **RMSE** the square root of the mean of
$(\tau - \hat\tau)^2$, both in the units of the torque. They sit near the noise level. 
$R^2 = 1 - \sum(\tau - \hat\tau)^2 / \sum(\tau - \bar\tau)^2$ compares the model with always
predicting the mean: 0.998 means the model explains almost all the variation.

A robot arm has several joints. Pass the vector and the metric reports each joint and the mean.

```python
from dense_armor.core.evaluation.metrics import MeanAbsoluteError
m = MeanAbsoluteError()
m.update([1.0, 2.0], [1.5, 2.5])
m.update([2.0, 4.0], [2.5, 3.0])
print(m.get())
```

```
{'per_joint': [0.5, 0.75], 'mean': 0.625}
```

Joint 0 is off by 0.5 both times; joint 1 by 0.5 and then 1.0, mean 0.75.

### Mean squared error

The **MSE** (mean squared error) is the mean of $(\tau - \hat\tau)^2$ without the square root.
It is the quantity that RMSE takes the square root of: `RMSE = sqrt(MSE)`. MSE is useful when
you need the raw squared error for gradient-based optimisation or for combining with other
squared-error terms.

```python
from dense_armor.core.evaluation.metrics import MeanSquaredError, RootMeanSquaredError
mse, rmse = MeanSquaredError(), RootMeanSquaredError()
for a, b in [(1.0, 1.5), (2.0, 2.5), (3.0, 2.5)]:
    mse.update(a, b)
    rmse.update(a, b)
print(f"MSE: {mse.get()}, RMSE: {rmse.get()}")
```

```
MSE: 0.25, RMSE: 0.5
```

Count by hand:
$(1.0-1.5)^2 = 0.25$, $(2.0-2.5)^2 = 0.25$, $(3.0-2.5)^2 = 0.25$.
MSE $= (0.25+0.25+0.25)/3 = 0.25$. RMSE $= \sqrt{0.25} = 0.5$.

## float32

`R2Score(dtype=numpy.float32)` keeps the Welford state and the
`ss_res` sum in float32 and applies the shifted-data algorithm with
`K = y_true[0]` (Healy 2022, Algorithm 2) online: TwoSum on `s1`,
`s2` and `ss_res`, two-term formulas for the mean and for `M2`.
Relative error against the float64 reference on N(1e3, 0.01), 100,000
samples, seed 0: `M2` 1.7e-8, `ss_res` 3.7e-8. On a constant target
`get()` returns 0.0. With this mode `w` must be 1. `R2Score` is a
`Metric`, not an `Estimator`: `check_estimator` does not apply.

Cost, median of 7 runs of one update over 20,000 samples, one CPU thread: default 0.002044 ms (IQR 0.000635), float32 0.009688 ms (IQR 0.000079). The claim that the float32 path is at most as slow as the default is not met: on a scalar loop the numpy
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

## How it works

MAE accumulates $|\tau - \hat\tau|$; MSE accumulates $(\tau - \hat\tau)^2$; R² accumulates sum of squared residuals and total sum of squares to compute $1 - \text{SS}_\text{res}/\text{SS}_\text{tot}$.

## The parameters

| Parameter | Meaning | Default |
|---|---|---|
| `dtype` | numpy dtype for accumulators | float64 |
| `normalize` | divide by count to get mean | True |

## API reference

::: dense_armor.core.evaluation.metrics.regression.MeanAbsoluteError  
::: dense_armor.core.evaluation.metrics.regression.MeanSquaredError  
::: dense_armor.core.evaluation.metrics.regression.RootMeanSquaredError  
::: dense_armor.core.evaluation.metrics.regression.R2Score

## Details

- **Base implementation:** `MeanAbsoluteError` tracks sum of absolute errors and count; `MeanSquaredError` tracks sum of squared errors and count; `RootMeanSquaredError` takes the square root; `R2Score` tracks SS_res and SS_tot.
- **Merge:** Two `MeanAbsoluteError` metrics with sums `s1, n1` and `s2, n2` merge to `s1+s2, n1+n2`. Same for MSE and RMSE. For `R2Score`, the merged value would carry float32 error; the default path merges without this limit.
- **Sources:** Hyndman and Brien (1999) "An Introduction to Statistical Learning"; Montgomery et al. (2017) "Introduction to Linear Regression Analysis".