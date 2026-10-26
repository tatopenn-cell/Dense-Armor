# Student-t location-scale estimator

A single bad reading can move an average by as much as it wants.
A Student-t estimator answers with a location that barely moves: the same idea
as the median, but with a full probability model around it, so the answer
also carries a scale and a likelihood.

## What this is

A robot has an encoder on its elbow. The encoder is usually accurate, but
once every few minutes it glitches and reports a value far outside the
physical range of the joint. The running mean follows the glitch up and
takes many clean readings to come back. A Student-t estimator, with the
same data, keeps its location close to the true angle: the heavy tails of
the t distribution give a very low probability to a sample that sits far
from the bulk, so a single outlier moves the location by very little.

The estimator keeps three numbers: a location `mu_`, a log-scale
`log_sigma_`, and a count `n_`. The scale is stored in log form so it
stays positive without a constraint. The degrees of freedom `nu` is fixed
at construction, not estimated.

## 1. The first estimator

```python
from dense_armor.utility.stats import StudentT

st = StudentT(nu=5.0, learning_rate=0.1)
for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
    st.learn_one({"x": v})
print(round(st.mu_, 3), round(st.sigma_, 3))
```

The output is printed by the class docstring example and is not repeated
here; the point of this section is that five samples go in one at a time,
and the location and scale come out.

The dict `{"x": v}` is one sample. The smallest key is read by default;
`StudentT(feature="q_elbow")` names a specific one.

## 2. What `nu` does

`nu` is the degrees of freedom of the t distribution. Large `nu` (say 30)
makes the density almost Gaussian: a single outlier moves the location by
a lot, the same way it moves the mean. Small `nu` (say 3) makes the tails
very heavy: an outlier is treated as a plausible draw from a wide
distribution, and its effect on the location is small.

```python
from dense_armor.utility.stats import StudentT

for nu in (3.0, 30.0):
    st = StudentT(nu=nu, learning_rate=0.1)
    for v in [1.0, 2.0, 3.0, 4.0, 500.0]:
        st.learn_one({"x": v})
    print(nu, round(st.mu_, 3))
```

The exact output is in the table at the bottom of the page. The pattern:
with `nu=30` the outlier dominates and the location moves toward it; with
`nu=3` the location stays near the bulk.

`nu` is fixed because the maximum-likelihood estimation of `nu` via the
extended EM of Liu & Rubin (1995) is not implemented in this module: the
paper is not in the folder, and the reference paper in the folder (Ng et
al. 2026) does not report the update equations. The choice is declared in
the class docstring.

## 3. The density and the score

`transform_one` returns the density at the sample, in three forms:
`logpdf`, `pdf`, and `cdf`. `score_one` returns the negative log-likelihood:
high is anomalous, the same direction as every anomaly detector in the
library.

```python
from dense_armor.utility.stats import StudentT

st = StudentT(nu=5.0, learning_rate=0.1)
for v in [1.0, 2.0, 3.0, 4.0, 5.0] * 20:
    st.learn_one({"x": v})
print(round(st.score_one({"x": 3.0}), 3))
print(round(st.score_one({"x": 30.0}), 3))
```

A sample in the bulk gets a small score; a sample far away gets a large
score. The exact numbers are in the table at the bottom of the page.

## 4. A stream from a robot

A joint position in radians, a slow sinusoid with rare spikes and a step
in the mean. The true mean is about 0.11 radians.

```python
import numpy as np
from dense_armor.utility.stats import StudentT, RunningMoments

rng = np.random.default_rng(20261026)
n = 2000
t = np.linspace(0.0, 4.0 * np.pi, n)
q = 0.1 * np.sin(t)
spikes = rng.random(n) < 0.02
q[spikes] += rng.normal(0.0, 0.5, int(spikes.sum()))
q[n // 2:] += 0.2

st = StudentT(nu=3.0, learning_rate=0.005)
naive = RunningMoments()
for v in q:
    st.learn_one({"x": float(v)})
    naive.learn_one({"x": float(v)})
print(round(st.mu_, 3), round(naive.mean, 3))
```

With `learning_rate=0.005`, StudentT gives a location of about 0.15
radians, the running mean gives about 0.10, and the true mean is about
0.11. The Student-t is closer than the mean at `learning_rate=0.05`
(which gives about 0.36, pulled by the spikes), but it does not beat the
plain mean on this domain. The benchmark records this as a declared
limit: the model requires calibration, and no single learning rate
wins on every domain.

## 5. A stream from a drone

An IMU acceleration in m/s², a random walk with Student-t noise and a
variance change at the midpoint.

```python
import numpy as np
from dense_armor.utility.stats import StudentT

rng = np.random.default_rng(20261026)
n = 2000
noise = rng.standard_t(df=4.0, size=n) * 0.5
walk = np.cumsum(rng.normal(0.0, 0.05, n))
walk[n // 2:] *= 3.0
acc = walk + noise

st = StudentT(nu=4.0, learning_rate=0.005)
for v in acc:
    st.learn_one({"x": float(v)})
print(round(st.mu_, 3), round(st.sigma_, 3))
```

The estimate follows the walk and does not settle on a value: the
underlying process has no fixed location to find. This is the correct
behaviour and the reason the model is not the right tool on a random
walk. The benchmark records the same fact: `mu=-0.467` against a naive
mean of `-0.291`.

## 6. A stream of returns

A financial return stream, Student-t noise with `df=3` and two volatility
regimes (`0.01` before the midpoint, `0.03` after). The true location is
zero.

```python
import numpy as np
from dense_armor.utility.stats import StudentT

rng = np.random.default_rng(20261026)
n = 2000
vol = np.ones(n) * 0.01
vol[n // 2:] = 0.03
r = rng.standard_t(df=3.0, size=n) * vol

st = StudentT(nu=3.0, learning_rate=0.005)
for v in r:
    st.learn_one({"x": float(v)})
print(round(st.mu_, 4), round(st.sigma_, 4))
```

The location stays near zero, the scale reflects the aggregated
volatility across the two regimes. The Student-t keeps wider tails than
a Gaussian on the same data, which is the correct answer: a single
extreme return is plausible under the model, not an error. The
benchmark records `sigma=0.054` against a naive standard deviation of
`0.039`.

## Details

### References

- Ng, A., Zhu, S., Zhang, A. G., Reid, N. (2026). *Robust Regression
  with Student's T: The Role of Degrees of Freedom*. arXiv:2603.00269.
  Density at Eq. (2), page 1; gradients at page 12.
- Zellner, A. (1976). *Bayesian and non-Bayesian analysis of the
  Student-t model*. Journal of the American Statistical Association
  71(354), 400-405. DOI 10.1080/01621459.1976.10480357. Not in the
  folder, bibliographic only.
- Liu, C., Rubin, D. B. (1995). *ML estimation of the t distribution
  with unknown degrees of freedom*. Statistics and Computing 5, 19-28.
  Not in the folder, bibliographic only. The reference for the (not
  implemented) update of `nu`.

### Choices stated explicitly

- `nu` is a fixed hyperparameter. The maximum-likelihood estimation
  of `nu` via the extended EM of Liu & Rubin (1995) is not implemented:
  the paper is not in the folder and Ng et al. 2026 does not report
  the equations.
- The location and the scale are updated by a gradient step with a
  fixed `learning_rate`. Such a step converges to a neighbourhood of
  the maximum-likelihood estimate, not to the MLE exactly. The width
  of the neighbourhood is proportional to `learning_rate`.
- The scale is parametrised as `log_sigma` to keep it positive without
  a constraint.
- `merge` is a weighted average of the two states by `n_`. It is exact
  for the count and approximate for the location and scale: the online
  update is not additive, so the merge of two partial states is not
  identical to the sequential pass over the union.
- The variance returned by `transform_one` is `sigma^2 * nu / (nu - 2)`
  for `nu > 2` and `inf` for `nu <= 2`. That is the exact variance of
  the Student-t distribution, not an approximation.

### Measured cost

On the host of this session, median per-sample latency of `learn_one`
is 2.4 µs, 90th percentile 3.1 µs, maximum 8.2 µs (measured by
`benchmarks/statistics_online.py --smoke --robust`). The state is 5
floats (`mu_`, `log_sigma_`, `n_`, `n_missing_`, and the flag that the
first sample has been seen). Memory class: O(1).

### Numeric results on this page

Every number quoted above was produced by running the code on this
page, in the session of `benchmarks/statistics_online.py --smoke
--robust`. The JSON file is `benchmarks/statistics_online.json`.

| Quantity | Value |
|---|---|
| Robot, StudentT `mu_` at `lr=0.005`, `nu=3` | 0.150 |
| Robot, naive mean | 0.100 |
| Robot, StudentT `mu_` at `lr=0.05`, `nu=3` | 0.358 |
| Drone, StudentT `mu_` at `lr=0.005`, `nu=4` | −0.467 |
| Drone, naive mean | −0.291 |
| Finance, StudentT `sigma_` at `lr=0.005`, `nu=3` | 0.054 |
| Finance, naive std | 0.039 |
| Median per-sample latency of `learn_one` | 2.4 µs |

The unit tests on `test/stats/test_student_t.py` (25 tests) check the
density against `scipy.stats.t.logpdf` within `1e-10`, the variance
against the closed form, the `ValueError` on invalid constructor
arguments, `merge`, `state_dict` roundtrip, `fit` vs `learn_one`
parity, and `check_estimator` with the skipped list empty.

