# Autoregressive models

**In plain words:** These tools predict the next value of a signal from its last few values, which works when the signal has memory: where it was a moment ago tells you where it goes next.

`OnlineAR` and `OnlineVAR` predict the next value of a signal from its own
recent past. They are the simplest time-series models: a weighted sum of the
last `order` samples, with the weights learned online.

## 1. What autoregression is

A signal has *memory* when its next value depends on its last values. The
torque of a joint that swings back and forth does: if the joint is moving
one way, the next reading is likely to keep moving that way. A first-order
autoregressive model writes this as

    x[t] = a * x[t-1] + noise

and an `order`-th model uses the last `order` samples:

    x[t] = a_1 * x[t-1] + ... + a_order * x[t-order] + noise

The model does not know `a` in advance. It learns it from the stream, one
sample at a time, by recursive least squares: each new sample nudges the
coefficients toward the value that would have predicted it best.

## 2. Learning an AR(1) coefficient

The smallest possible model: `order=1`, one coefficient. The stream is a
first-order autoregression with a true coefficient of `0.8`, generated
inside the page for reproducibility.

```python
import numpy as np
from dense_armor.utility.timeseries import OnlineAR

rng = np.random.default_rng(0)
x = 0.0
ar = OnlineAR(order=1, lam=1.0)
for _ in range(500):
    x = 0.8 * x + 0.3 * rng.normal()
    ar.learn_one({"x": float(x)})
print(round(float(ar.coef_[0]), 4))
```

```
0.783
```

The true coefficient is `0.8`; the learned one is `0.78` after 500 samples.
The `lam` parameter is the forgetting factor, in `(0, 1]`. `lam = 1.0`
weights every sample equally, so the coefficients converge to the batch
least-squares solution. A smaller `lam` makes the model forget the older
samples faster, useful when the true coefficient changes over the day.
With `order > 1` the coefficients in `coef_` run from the oldest lag to the
newest.

## 3. Predicting and residual

`learn_one` updates the state; `predict_one` returns the forecast of the
**next** sample from the current history. The method ignores its argument
`x` when the model is an autoregression: the prediction depends only on the
samples already seen, never on a value the caller passes in.

```python
import numpy as np
from dense_armor.utility.timeseries import OnlineAR

rng = np.random.default_rng(0)
x = 0.0
ar = OnlineAR(order=2, lam=0.99)
for _ in range(200):
    x = 0.8 * x + 0.3 * rng.normal()
    ar.learn_one({"x": float(x)})

forecast = ar.predict_one({"x": float(x)})
print(round(forecast, 4))
```

```
-0.1165
```

`transform_one` returns the difference between a sample and the forecast of
the model. Call it **before** `learn_one` on the same sample: the forecast is
then the one the model made before seeing it. After `learn_one`, the forecast
already refers to the next sample. That residual is what a control loop
watches: a small residual means the model understands the signal, a jump in
the residual means the signal broke the pattern.

```python
import numpy as np
from dense_armor.utility.timeseries import OnlineAR

ar = OnlineAR(order=1, lam=1.0)
for v in [0.1, 0.2, 0.15, 0.18]:
    ar.learn_one({"x": float(v)})
out = ar.transform_one({"x": 0.17})
print(round(out["residual"], 4))
```

```
-0.0212
```

## 4. Several channels at once: OnlineVAR

When several signals move together, a **vector autoregression** predicts
every channel from the recent past of every channel:

    y[t] = A_1 y[t-1] + ... + A_order y[t-order] + noise

with `y` a vector of `N` channels and each `A_p` an `N x N` matrix. The
model is the multivariate extension of `OnlineAR`, and it recovers the
matrices online with the same recursive least squares rule. On two coupled
channels (the diagonal terms are the self-lags, the off-diagonal terms the
cross-channel couplings):

```python
import numpy as np
from dense_armor.utility.timeseries import OnlineVAR

rng = np.random.default_rng(0)
var = OnlineVAR(order=1, lam=0.99, group_lambda=0.0,
                feature_keys=["a", "b"])
a, b = 0.0, 0.0
for _ in range(500):
    a_new = 0.8 * a + 0.2 * b + 0.1 * rng.normal()
    b_new = 0.3 * a + 0.7 * b + 0.1 * rng.normal()
    a, b = a_new, b_new
    var.learn_one({"a": float(a), "b": float(b)})
print([round(float(v), 2) for v in var._a[0]])
print([round(float(v), 2) for v in var._a[1]])
```

```
[0.83, 0.18]
[0.37, 0.64]
```

The true matrices are `[[0.8, 0.2], [0.3, 0.7]]`; the learned ones are
`[[0.83, 0.18], [0.37, 0.64]]` after 500 samples. The cross-channel terms
(0.18 and 0.37) are recovered with the same accuracy as the self-lags
(0.83 and 0.64): the model sees that `b` influences `a`, and vice versa.

## 5. Sparse coefficients: the group-lasso penalty

When only a few cross-channel couplings matter, the `group_lambda` parameter
adds a group-lasso penalty on each `N x N` block of `A_1`. Setting `n`
blocks to zero shrinks them; the diagonal blocks (a channel's own lags) are
never penalized, so the model keeps the internal memory of every channel.
The rule is equation (25) of the source paper: the `1{n != n'}` indicator
sets the penalty to zero on the diagonal.

```python
import numpy as np
from dense_armor.utility.timeseries import OnlineVAR

rng = np.random.default_rng(0)
var = OnlineVAR(order=1, lam=0.99, group_lambda=0.1,
                feature_keys=["a", "b"])
a, b = 0.0, 0.0
for _ in range(500):
    a_new = 0.9 * a + 0.05 * rng.normal()
    b_new = 0.9 * b + 0.05 * rng.normal()
    a, b = a_new, b_new
    var.learn_one({"a": float(a), "b": float(b)})
print([round(float(v), 4) for v in var._a[0]])
print([round(float(v), 4) for v in var._a[1]])
```

```
[0.8542, 0.0]
[0.0009, 0.7666]
```

The two channels are independent (no cross-coupling), and the penalty drives
both off-diagonal terms to zero (the second is at `9e-4`, effectively zero)
while keeping the diagonal self-lags at `0.85` and `0.77`. Without the
penalty the same stream gives off-diagonal terms around `0.05`, ten times
larger.

## 6. On a real arm

A `SyntheticArm` from a real Panda URDF gives `q`, `qd`, and `tau` at 50 Hz.
The next-sample residual of joint 1 is a natural "normal" for the joint:
the torque of a healthy arm repeats with the task cycle, so the residual
stays small; a payload shifts the pattern, and the residual grows.

```python
from pathlib import Path
from dense_armor.utility.datasets import SyntheticArm
from dense_armor.utility.timeseries import OnlineAR

ds = SyntheticArm(
    Path("test/fixtures/urdf/panda.urdf"),
    period_s=1.0, rate_hz=50.0, n_cycles=4,
    fault_at_s=2.0, fault="payload", payload_mass=5.0, seed=0,
)
ar = OnlineAR(order=1, lam=0.99)
res = []
for sig, _ in ds.stream():
    out = ar.transform_one({"x": float(sig["tau_1"])})
    res.append(out["residual"])
```

The residual is not a clean indicator on this stream: a first-order model on
the torque of joint 1 tracks the task cycle (period 1 s), so it cannot
distinguish a payload from a healthy cycle within the same period. The
estimator that *does* see the change on this stream is `CycleResidual`, and
[the section page](index.md) uses it there. The `OnlineAR` example above is
here for the mechanics; the use case is on the [cycle page](cycle.md).

## Details

### References

Zaman, B., Lopez Ramos, L. M., Romero, D., Beferull-Lozano, B. (2019).
*Online topology identification from vector autoregressive time series*.
arXiv:1904.01864. Equation (19) is the exponentially weighted loss, equation
(21a) the accumulator `Phi[t]` and (21b) the cross-correlation `r[t]`, equation
(24) the loss with the group-lasso term, equation (25) the shrinkage rule with
the indicator `1{n != n'}`. The algorithm is Procedure 2 (page 7) and starts
from `Phi[P - 1] = sigma^2 I`. Theorem 3 (page 8) bounds the static regret by
`O(L B^2 sqrt(T))` for the step `O(1 / sqrt(t0 2^(m-1)))` in doubling windows;
Theorem 4 (page 8) gives a logarithmic bound for `alpha_t = 1 / (beta t)`,
with `beta` a lower bound of `lambda_min(Phi[t])`. Both need the assumptions
A1, A2, A3 (bounded samples, bounded minimum and maximum eigenvalue of
`Phi`). The adaptive step `alpha_t = c / lambda_max(Phi[t])` is discussed on
page 9, at the end of section IV-C.

### Choices stated explicitly

- **The ridge is `sigma2`, default `0.0`.** Procedure 2 starts from
  `Phi[P - 1] = sigma^2 I`. `OnlineVAR` has the argument `sigma2` for it and
  starts from zero when it is `0.0`.
- **Three differences from Procedure 2 are kept.** `Phi` and `r` are
  accumulated without the factor `mu = 1 - gamma` of equations (21a) and
  (21b), so `lam = 1.0` is allowed; the recursion starts at the first sample
  with a zero history, while the paper starts at `t = P`; the step is
  `alpha / (1 + t)`.
- **The step `alpha / (1 + t)` has no regret bound.** Theorem 3 needs the
  doubling-window step and Theorem 4 needs `alpha_t = 1 / (beta t)`.
- **`alpha = 1.0` needs data of variance below 2.** The step is stable while
  `alpha * lambda_max(Phi) / (1 + t) < 2`. With independent channels of
  standard deviation 1.5, `alpha = 1.0` diverges (values of order `1e265`
  after 2000 samples) and `alpha = 0.1` stays bounded up to a standard
  deviation of 3. Scale the channels or lower `alpha`.
- **Feature order from `feature_keys`.** Every sample is a dict; the
  model fixes the channel order the first time it sees a dict, either
  from the explicit `feature_keys` or the sorted keys.

### Measured cost

`budget_s = 1e-3 s` per sample on a CPU. With `N` channels and order `P`
the update is `O(N^2 P^2)`: on `N = 2, P = 1` a few dozen microseconds;
on `N = 10, P = 3` (the paper's largest example) about 1 ms. Memory is
`O(N^2 P^2)` for `Phi` and `O(N^2 P)` for `r`, both fixed at construction.

### Development diary

The first delivery had a `delta * I` ridge in `OnlineVAR._init`, copied from
`RecursiveLeastSquares`; a review removed it with a test on 400 samples of an
AR(1) stream. A later review found that Procedure 2 starts from
`sigma^2 I` and added the argument `sigma2` (default `0.0`). The same review
made `OnlineAR` use `RecursiveLeastSquares`, and measured the divergence of
`alpha = 1.0` on data of large variance.
