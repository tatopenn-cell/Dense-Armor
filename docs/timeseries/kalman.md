# Kalman filters

**In plain words:** A Kalman filter works out where something really is, and how fast it is moving, from sensor readings that are noisy. It updates its estimate with every new reading.

`KalmanFilter` estimates the hidden state of a system from noisy
measurements, one sample at a time. It is not the same kind of estimator as
`OnlineAR`: instead of predicting the next value of a signal, it predicts
the state the system is *in* — a position and a velocity, a pose and a bias.

## 1. What a Kalman filter does

Suppose a robot moves with constant velocity. Two numbers describe its
state: a position and a velocity. The dynamics model says where the state
will be in the next step:

    x[t] = F x[t-1] + process noise

The measurements from a sensor only see part of the state (usually the
position), through a matrix `H`:

    z[t] = H x[t] + measurement noise

`KalmanFilter` combines the two sources: it predicts the next state with
the dynamics, then corrects the prediction with the measurement, weighting
the two by their uncertainties `Q` (process) and `R` (measurement). If the
sensor is noisy, the model trusts the dynamics more; if the dynamics are
uncertain, it trusts the sensor more.

## 2. Tracking a constant-velocity target

A target moves at a constant speed of `0.1` units per step, and the
position is read with a noise of `0.1`. The state vector holds position and
velocity; the transition matrix `F = [[1, 1], [0, 1]]` advances the position
by the velocity at every step.

```python
import numpy as np
from dense_armor.utility.timeseries import KalmanFilter

F = np.array([[1.0, 1.0], [0.0, 1.0]])
H = np.array([[1.0, 0.0]])
Q = 1e-4 * np.eye(2)
R = np.array([[1e-2]])
kf = KalmanFilter(F=F, H=H, Q=Q, R=R)

rng = np.random.default_rng(0)
for i in range(100):
    pos = 1.0 + 0.1 * i + 0.1 * rng.normal()
    kf.learn_one({"pos": float(pos)})
est = kf.predict_one({})
print([round(float(v), 4) for v in est])
```

```
[10.8041, 0.0778]
```

The true position is `10.9` and the true velocity `0.1`; the estimated ones
are `10.80` and `0.078`. Both are inside the noise band. The output of
`predict_one` is the full state vector, not a scalar: the caller can read
whichever component they need.

## 3. Adaptive noise: two variants

The fixed `KalmanFilter` assumes `Q` and `R` are known and constant. In
practice the sensor noise changes: a robot that moves from a smooth floor
to a carpet picks up more vibration, and `R` grows. Two classes follow the
noise from the data.

`VBAdaptiveKalman` puts an inverse-Wishart prior on the measurement
covariance `R` and updates its parameters `nu` and `V` at every sample by a
fixed-point iteration, following Algorithm 1 of Mbalawata et al. (2013). On a target
whose noise jumps from `0.1` to `1.0` halfway through, the estimate of `R`
follows:

```python
import numpy as np
from dense_armor.utility.timeseries import VBAdaptiveKalman

F = np.array([[1.0, 1.0], [0.0, 1.0]])
H = np.array([[1.0, 0.0]])
Q = 1e-4 * np.eye(2)
R = np.array([[1e-2]])
kf = VBAdaptiveKalman(F=F, H=H, Q=Q, R=R)

rng = np.random.default_rng(0)
for i in range(400):
    std = 0.1 if i < 200 else 1.0
    pos = 1.0 + 0.1 * i + std * rng.normal()
    kf.learn_one({"pos": float(pos)})
print(round(float(kf.R_estimate[0, 0]), 4))
```

```
0.787
```

The true `R` after the jump is `1.0`; the estimate is `0.79`, which is what
the filter has seen in the last samples. The fixed filter would still
believe `R = 0.01` and would over-trust the noisy measurements.

`InnovationAdaptiveKalman` adapts both `Q` and `R` from the innovation and
residual of the filter, following Popescu et al. (2026). The innovation
covariance and the residual covariance are computed over a sliding window
of `window` samples.

```python
import numpy as np
from dense_armor.utility.timeseries import InnovationAdaptiveKalman

F = np.array([[1.0, 1.0], [0.0, 1.0]])
H = np.array([[1.0, 0.0]])
Q = 1e-4 * np.eye(2)
R = np.array([[1e-2]])
kf = InnovationAdaptiveKalman(F=F, H=H, Q=Q, R=R, window=50)

rng = np.random.default_rng(0)
for i in range(400):
    std = 0.1 if i < 200 else 1.0
    pos = 1.0 + 0.1 * i + std * rng.normal()
    kf.learn_one({"pos": float(pos)})
print(round(float(kf.R_estimate[0, 0]), 4))
```

```
0.8264
```

The two adaptive filters give the same order of magnitude (`0.79` and
`0.83`) on the same stream. They differ in what they adapt: `VBAdaptiveKalman`
adapts only `R`, `InnovationAdaptiveKalman` adapts both `Q` and `R`.

## 4. On a real arm

A `SyntheticArm` from a real Panda URDF gives `q`, `qd`, and `tau` at 50 Hz.
Joint 1 follows a sinusoid with a period of 1 s. The measurement of `q_0`
carries noise of `0.01`; a Kalman filter can recover a smoother estimate of
the true position, if the state model matches the dynamics.

A robot that moves in a known periodic pattern has an **harmonic model**:
the state `[q, qd]` rotates in phase space at the angular frequency
`omega = 2 * pi / period_s`. The transition matrix is that rotation over
one sample of `dt = 1/50 s`.

```python
import numpy as np
from pathlib import Path
from dense_armor.utility.datasets import SyntheticArm
from dense_armor.utility.timeseries import KalmanFilter

dt = 0.02
omega = 2 * np.pi
F = np.array([
    [np.cos(omega * dt), np.sin(omega * dt) / omega],
    [-omega * np.sin(omega * dt), np.cos(omega * dt)],
])
H = np.array([[1.0, 0.0]])
Q = 1e-6 * np.eye(2)
R = np.array([[1e-4]])
kf = KalmanFilter(F=F, H=H, Q=Q, R=R)
ds = SyntheticArm(
    Path("test/fixtures/urdf/panda.urdf"),
    period_s=1.0, rate_hz=50.0, n_cycles=2,
    fault_at_s=None, fault=None, seed=0,
)
rng = np.random.default_rng(0)
err = []
for sig, _ in ds.stream():
    q_true = float(sig["q_0"])
    q_meas = q_true + 0.01 * rng.normal()
    kf.learn_one({"q": float(q_meas)})
    est = kf.predict_one({})
    err.append(abs(float(est[0]) - q_true))
print(round(float(np.mean(err)), 4))
```

```
0.0034
```

The raw measurement has a mean absolute error of `0.0080`; the filtered
estimate has `0.0034`, a factor of `2.3`. The lesson is in the comparison
with the earlier sections: a Kalman filter with a *wrong* state model
(a constant, a constant-velocity, or a constant-acceleration model on this
stream) does worse than the raw measurement, because none of them can follow
a sinusoid. The harmonic model works because it is the true dynamics of the
joint.

## Details

### References

Mbalawata, I. S., Sarkka, S., Vihola, M., Haario, H. (2013). *Adaptive
Metropolis algorithm using variational Bayesian adaptive Kalman filter*.
arXiv:1308.5875 - Algorithm 1 with equation (22), page 9, gives the
fixed-point iteration on `nu` and `V` used by `VBAdaptiveKalman`; the
inverse-Wishart distribution is equation (19) on page 8.

Popescu, M., Mronga, D., Kumar, S., Kirchner, F. (2026). *Residual-based
adaptive Kalman filtering for legged robot state estimation*.
arXiv:2608.02316 - equations (13) to (16): the residual covariance over a
sliding window gives `R`, the innovation covariance gives the scale of `Q`.
The window of one second is the paper's choice (section 5.3.2).

Kalman, R. E. (1960). *A new approach to linear filtering and prediction
problems*. Journal of Basic Engineering, 82(1), 35-45. The standard form of
the prediction and correction equations used by the fixed `KalmanFilter`.

### Choices stated explicitly

- **The fixed `KalmanFilter` takes `F`, `H`, `Q`, `R` at construction and
  never changes them.** No auto-tuning, no default model. The caller
  chooses the dynamics.
- **The two adaptive variants adapt different things.** `VBAdaptiveKalman`
  adapts only `R` via an inverse-Wishart prior with parameters `nu` and `V`;
  `InnovationAdaptiveKalman` adapts both `Q` and `R` from the innovation
  and residual covariances over a sliding window of `window` samples.
- **`VBAdaptiveKalman` uses a fixed-point iteration with `n_iter`
  passes** (default `5`, the number the paper used). With `rho = 1` it
  equals Algorithm 1 to `1e-9`; the test compares it with a literal
  implementation. For `rho < 1` (default `0.99`) equation (22) as printed
  weights the previous covariance by `nu_prev - d - 1` and the code by
  `rho * (nu_prev - d - 1)`: on this page's stream the printed equation
  gives `2.18` and the code `0.79` for a true `R` of `1.0`. That
  comparison is this page's own computation, not a statement of the paper.
- **`InnovationAdaptiveKalman` differs from the paper in four points.**
  `R` is forced diagonal (the paper keeps the full matrix), `alpha` is
  clipped to `[0.1, 10]`, the whole `Q` is scaled (the paper scales only the
  IMU terms), and the rotation and the average over the legs of equations
  (14) and (16) are left out because they belong to a legged-robot filter.
  The adapted `Q` and `R` are read with `Q_estimate` and `R_estimate`; the
  constructor arguments keep their values, so `clone()` returns the starting
  filter. `step` carries `Q` and `R` in its state and equals `learn_one`.
- **The `q_0` example uses a harmonic transition matrix**, physically
  correct for the specific SyntheticArm motion (`period_s = 1.0`). The
  period is a property of the trajectory, not of the robot.

### Measured cost

| estimator | per sample | state memory |
|---|---|---|
| `KalmanFilter` | 71 us | `O(n^2)` with `n` the state size |
| `VBAdaptiveKalman` | 309 us (5 iterations) | `O(n^2)` |
| `InnovationAdaptiveKalman` | 161 us | `O(n^2 + window)` |

All measured as the p99 of `predict_one` plus `learn_one` over 3000 samples
on a Windows desktop CPU, with `n = 2` and one measured value.
`budget_s = 1e-3 s` for every class.

### Development diary

The first attempt for section 4 used a constant-velocity model on `q_0`
(`F = [[1, dt], [0, 1]]`). That model gave a position error of `0.31`
against a measurement error of `0.008`: it was *worse* than doing nothing,
because the true joint motion is a sinusoid, not a constant velocity. The
same happened with `F = [[1]]` (constant) and with a constant-acceleration
model (`[q, qd, qdd]`). The harmonic model was the first to beat the raw
measurement, by a factor of `2.3`. The lesson is worth stating: Kalman
filtering works only when the state model is the real dynamics; a generic
polynomial model can hurt.
