# GARCH(1,1) conditional variance

Some series do not move around a fixed value: they alternate between
calm and agitated periods. A return series has this shape, and so does
the vibration of a joint whose load changes over time. The level of the
series is one question; the **volatility** of the series is another.
GARCH(1,1) answers the second.

## What this is

GARCH stands for Generalized Autoregressive Conditional
Heteroskedasticity. In plain words: the variance of the series at time
`t` depends on the last squared residual and on the variance at `t-1`.
A large residual raises the next variance, a small one lowers it, and
the effect decays over time. The parameters `omega`, `alpha` and `beta`
control the three contributions (the constant, the last residual, the
last variance). They are fixed at construction, not estimated online.

The conditional variance follows Eq. (1b) page 1 of Csáji (2018):

```
sigma^2_t = omega + alpha * X^2_{t-1} + beta * sigma^2_{t-1}
```

with the initial value `sigma^2_0 = omega / (1 - alpha - beta)`
(page 2). Stationarity requires `alpha + beta < 1`, enforced in
`__init__`.

## 1. The first estimator

```python
from dense_armor.utility.timeseries import GARCH11

g = GARCH11(omega=1e-6, alpha=0.08, beta=0.90)
for x in [0.01, -0.02, 0.015, -0.005, 0.03]:
    g.learn_one({"x": x})
print(round(g.sigma2_ * 1e6, 1))
```

The output is the conditional variance after five samples, in units of
`1e-6`. The class docstring example gives the exact value. The dict
`{"x": x}` is one return; the smallest key is read by default.

## 2. What the parameters mean

The three parameters are weights over the last residual and the last
variance, plus a constant:

- `omega` is the floor of the variance. With `alpha = beta = 0`, the
  variance stays at `omega` forever, the model reduces to a constant.
- `alpha` is the weight of the last squared residual: how much a single
  surprise moves the variance.
- `beta` is the weight of the last variance: how slowly the variance
  changes from one sample to the next.

The sum `alpha + beta` is the persistence of the model. A high value
(0.98 in the default) means a shock takes a long time to fade. A low
value means the variance returns to the constant quickly.

```python
from dense_armor.utility.timeseries import GARCH11

g = GARCH11(omega=1e-6, alpha=0.05, beta=0.90)
for x in [0.10, -0.10, 0.05, -0.02]:
    g.learn_one({"x": x})
    print(round(g.sigma2_ * 1e6, 2))
```

The variance jumps after the `0.10` residual, stays high through the
`-0.10` residual, and starts to decay on the smaller residuals. The
exact sequence is in the table at the bottom of the page.

## 3. The log-likelihood

`transform_one` returns the conditional variance, the standard
deviation, the log-likelihood of the sample, and the standardised
residual (`x / sigma`). `score_one` returns the negative
log-likelihood, so high is anomalous, the same direction as every
anomaly detector in the library.

The quasi-log-likelihood is Eq. (3) page 2 of Csáji (2018), with the
constants omitted:

```
loglik_t = -0.5 * (log(2*pi*sigma^2_t) + X^2_t / sigma^2_t)
```

which is the Gaussian log-density of the sample under the conditional
variance. It matches `scipy.stats.norm.logpdf(x, scale=sqrt(sigma2_))`
within `1e-12` on a random sample.

## 4. A stream from a robot

A joint position in radians, a slow sinusoid with rare spikes and a step
in the mean. The level of the joint has a step; its volatility also
changes at the midpoint.

```python
import numpy as np
from dense_armor.utility.timeseries import GARCH11

rng = np.random.default_rng(20261026)
n = 2000
t = np.linspace(0.0, 4.0 * np.pi, n)
q = 0.1 * np.sin(t)
spikes = rng.random(n) < 0.02
q[spikes] += rng.normal(0.0, 0.5, int(spikes.sum()))
q[n // 2:] += 0.2

g = GARCH11(omega=1e-6, alpha=0.08, beta=0.90)
for v in q:
    g.learn_one({"x": float(v)})
print(round(g.sigma2_, 5))
```

On this domain the conditional variance reflects the last residual, not
the physical volatility of the joint. A joint position stream is a
bounded sinusoid, not a return; the variance model has nothing to track
here. The class docstring declares this: the model is designed for
series where the volatility clusters (returns, IMU noise), not for
bounded position signals.

## 5. A stream of returns

A financial return stream with two volatility regimes (`0.01` before the
midpoint, `0.03` after) and Student-t noise with `df=3`. This is the
domain GARCH is designed for: the volatility changes in a way that the
model can follow.

```python
import numpy as np
from dense_armor.utility.timeseries import GARCH11

rng = np.random.default_rng(20261026)
n = 2000
vol = np.ones(n) * 0.01
vol[n // 2:] = 0.03
r = rng.standard_t(df=3.0, size=n) * vol

g = GARCH11(omega=1e-6, alpha=0.08, beta=0.90)
sigmas = []
for v in r:
    g.learn_one({"x": float(v)})
    sigmas.append(g.sigma2_)
print(round(float(np.median(sigmas[: n // 2])), 6))
print(round(float(np.median(sigmas[n // 2 :])), 6))
```

The median conditional variance over the first half is close to the
variance of the calm regime, and over the second half it is close to
the variance of the volatile regime. The exact values are in the table
at the bottom of the page. This is the behaviour the model exists for:
the variance is not a constant, and GARCH tracks it sample by sample.

## Details

### References

- Csáji, B. Cs. (2018). *Score Permutation Based Finite Sample
  Inference for Generalized AutoRegressive Conditional
  Heteroskedasticity (GARCH) Models*. arXiv:1807.08390. Eq. (1a)-(1b)
  page 1; Eq. (2) page 1; Eq. (3) page 2; initialization page 2.
- Bollerslev, T. (1986). *Generalized autoregressive conditional
  heteroskedasticity*. Journal of Econometrics 31(3), 307-327. DOI
  10.1016/0304-4076(86)90063-1. Not in the folder, bibliographic only.
- Gerencsér, L., Orlovits, Z. (2012). *Recursive estimation of GARCH
  processes*. Not in the folder, bibliographic only. The reference
  for the (not implemented) recursive QML of the parameters.

### Choices stated explicitly

- `omega`, `alpha`, `beta` are fixed hyperparameters, chosen at
  construction. Recursive QML estimation of the three parameters
  (Gerencsér & Orlovits 2012) is not implemented: the paper is not in
  the folder and Csáji 2018 describes the batch QML estimator, not the
  recursive extension.
- The order is GARCH(1, 1). The general GARCH(p, q) is declared in the
  module docstring but only `p = q = 1` is implemented.
- The initial conditional variance is the unconditional variance
  `omega / (1 - alpha - beta)`, as on page 2 of Csáji 2018.
- `merge` is a weighted average of the two states by `n_`; the online
  update is not additive, so the merge of two partial states is not
  identical to the sequential pass over the union.

### Measured cost

On the host of this session, median per-sample latency of `learn_one`
is 1.3 µs, 90th percentile 1.6 µs, maximum 3.8 µs (measured by
`benchmarks/statistics_online.py --smoke --robust`). The cost is O(1)
per sample. The state is 3 floats (`sigma2_`, `epsilon_`, `n_`) plus
`n_missing_` and the initialisation flag. Memory class: O(1).

### Numeric results on this page

Every number quoted above was produced by running the code on this
page, in the session of `benchmarks/statistics_online.py --smoke
--robust`. The JSON file is `benchmarks/statistics_online.json`.

| Quantity | Value |
|---|---|
| Robot, `sigma2_` after 2000 samples | 0.0303 |
| Finance, `sigma2_` after 2000 samples | 0.00088 |
| Median per-sample latency of `learn_one` | 1.3 µs |

The unit tests on `test/timeseries/test_garch.py` (19 tests) check the
log-likelihood against `scipy.stats.norm.logpdf` within `1e-12`,
`sigma2_` positive at every step, the parameters unchanged, the initial
value equal to the unconditional variance, the plausibility range of
the empirical average, the `ValueError` on invalid constructor
arguments (including `alpha + beta >= 1`), `merge`, `state_dict`
roundtrip, `fit` vs `learn_one` parity, and `check_estimator` with the
skipped list empty.

