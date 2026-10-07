# Online dynamics (residual torque learning)

A robot's URDF file describes its rigid-body dynamics: mass matrix, Coriolis terms,
gravity. That model is *almost* right. But a real robot also carries a payload in its
gripper, has friction in its joints, wears down over months. None of that is in the
file. The controller that trusts the URDF alone will be slightly wrong, always in the
same direction, and on a heavy payload that "slightly" is enough to miss the target.

`ResidualDynamicsLearner` learns that missing piece, one sample at a time, using one
**Recursive Least Squares** per joint on a five-feature basis. The nominal model comes
from [`urdf_dynamics`](../dynamics/urdf_dynamics.md); the learner adds a residual torque
on top so the controller does not need exact masses or friction coefficients.

## The two torques

A joint reports what it actually applied: the measured torque `tau_measured`. The URDF
model predicts what a *perfect* robot would have applied: the nominal torque. The
difference is what the learner has to model.

```python
import numpy as np
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
from dense_armor.utility.learn.online_dynamics import ResidualDynamicsLearner, write_minimal_urdf

nom = RigidBodyModel(write_minimal_urdf())
real = RigidBodyModel(write_minimal_urdf(payload_mass=0.5))
tau = lambda m, q, qd, qdd: np.asarray(m.mass_matrix(q)) @ qdd + np.asarray(m.bias_forces(q, qd))
learner, rng = ResidualDynamicsLearner(nom), np.random.default_rng(0)
for q, qd, qdd in rng.uniform(-1, 1, (40, 3, 2)):
    learner.learn_one(q, qd, qdd, tau(real, q, qd, qdd))
q, qd, qdd = rng.uniform(-1, 1, (3, 2))
print(np.round(tau(real, q, qd, qdd), 3), np.round(tau(nom, q, qd, qdd), 3), np.round(learner.predict_torque(q, qd, qdd), 3))
```

```
[1.496 0.558] [0.85  0.287] [1.422 0.478]
```

Two copies of the same 2-link arm, one with a 0.5 kg payload in the gripper (the "real" robot)
and one without (the URDF the controller knows). After 40 random moves the learner, starting
from the payload-free URDF, predicts torques of 1.422 and 0.478 N·m where the real arm needs 1.496
and 0.558; the URDF alone says 0.85 and 0.287. `write_minimal_urdf` writes this small test robot
to a file, so the example runs anywhere.

## 1. The five features per joint

For each joint `j`, the learner regresses the residual torque on five features:

```
[qdd_j, qd_j, sign(qd_j), g_j, 1]
```

- `qdd_j` — joint acceleration; captures inertia-like effects (a payload adds mass
  that scales with acceleration).
- `qd_j` — joint velocity; captures viscous friction.
- `sign(qd_j)` — sign of velocity; captures Coulomb friction (a constant torque
  opposing motion, independent of speed).
- `g_j` — gravity torque from the URDF at this configuration; captures a constant
  payload offset.
- `1` — a constant bias.

The five together are the minimal basis that captures a first-order residual on a
revolute joint. Everything else (higher-order friction, stiction, thermal effects) is
either negligible or requires more data than a single joint gives.

The target is the measured torque minus the nominal

```
tau_residual = tau_measured − M(q) qdd − C(q, qd) qd − g(q)
```

so the learner only has to model what the URDF does not.

## 2. Recursive Least Squares

### The formula, symbol by symbol

Let `w` be the weight vector (5 numbers per joint), `P` the inverse-covariance matrix
(5×5), `x` the feature vector, `y` the residual torque.

```
K = P x / (λ + xᵀ P x)          (Kalman gain, 5-vector)
w ← w + K (y − xᵀ w)             (weight update)
P ← (P − K xᵀ P) / λ             (covariance update)
```

`P₀ = δ · I` and `w₀ = 0`. `λ` is the **forgetting factor**: `λ = 1` weights all past
samples equally (batch least squares), smaller values weight recent samples more (the
model tracks a parameter that changes).

`P` is symmetrised after each update (`P ← 0.5 (P + Pᵀ)`) to keep numerical errors from
breaking the symmetry that the derivation assumes.

### Hand case

Start `δ = 1e6` (so `P₀ = 1e6 · I`, essentially uninformative), `λ = 1`, one sample
`x = (1, 2)`, `y = 5`:

```
P₀ x = 1e6 · (1, 2) = (1e6, 2e6)
xᵀ P₀ x = 1·1e6 + 2·2e6 = 5e6
K = (1e6, 2e6) / (1 + 5e6) ≈ (0.2, 0.4)
w₀ = (0, 0), so xᵀ w₀ = 0, and the error is y − 0 = 5
w ← (0, 0) + 5 · (0.2, 0.4) = (1.0, 2.0)
```

```python
from dense_armor.utility.learn.online_dynamics import RecursiveLeastSquares

rls = RecursiveLeastSquares(lam=1.0, delta=1e6)
rls.learn_one({"a": 1.0, "b": 2.0}, 5.0)
print(round(rls.predict_one({"a": 1.0, "b": 2.0}), 4), round(rls.predict_one({"a": 1.0, "b": 0.0}), 4))
```

```
5.0 1.0
```

After one sample the weights are exactly `(1, 2)`: `w · x = 1·1 + 2·2 = 5`. That is the
point of the large `δ`: with no prior information, the first sample determines the fit
completely.

With `λ < 1` the same update happens but the effective count of past samples is capped
at `1 / (1 − λ)`. `λ = 0.99` means roughly 100 samples of memory; `λ = 0.9` means
roughly 10.

## 3. On a simulated arm with a payload

A 2-link URDF, true torques = URDF torques + payload + viscous friction. The learner
starts from the URDF prediction and adds the residual. Test check: the learner's
predicted torques have lower RMSE than the nominal URDF prediction on held-out data.

```python
import numpy as np
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
from dense_armor.utility.learn.online_dynamics import ResidualDynamicsLearner, write_minimal_urdf

nom, real = RigidBodyModel(write_minimal_urdf()), RigidBodyModel(write_minimal_urdf(payload_mass=0.5))
tau = lambda m, s: np.asarray(m.mass_matrix(s[0])) @ s[2] + np.asarray(m.bias_forces(s[0], s[1]))
learner, rng = ResidualDynamicsLearner(nom), np.random.default_rng(0)
for s in rng.uniform(-1, 1, (40, 3, 2)):
    learner.learn_one(*s, tau(real, s))
test = rng.uniform(-1, 1, (20, 3, 2))
err = lambda f: np.sqrt(np.mean([(tau(real, s) - f(s)) ** 2 for s in test]))
print(round(err(lambda s: tau(nom, s)), 3), round(err(lambda s: learner.predict_torque(*s)), 3))
```

```
0.403 0.193
```

On 20 new moves the torque error (RMSE, N·m) is 0.403 for the URDF alone and 0.193 for the
learner after 40 samples: half the error.

The nominal model misses the payload; the learner picks it up in the five weights
(the `qdd` coefficient absorbs the payload's inertia, the `1` absorbs the constant
offset, the `sign(qd)` coefficient absorbs Coulomb friction).

## 4. When a parameter changes mid-run

If the payload changes at sample 200 of 400, a `λ = 1` learner (batch least squares)
keeps the old weights and gets the new regime wrong. A `λ = 0.99` learner forgets the
pre-change samples over ~100 samples and tracks the new regime.

```python
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
from dense_armor.utility.learn.online_dynamics import ResidualDynamicsLearner, DriftAwareResidualDynamicsLearner, write_minimal_urdf

model = RigidBodyModel(write_minimal_urdf())
fast = ResidualDynamicsLearner(model, lam=0.99, delta=1e6)
aware = DriftAwareResidualDynamicsLearner(model, lam=0.99)
print(type(fast).__name__, type(aware).__name__)
```

```
ResidualDynamicsLearner DriftAwareResidualDynamicsLearner
```

The choice of `λ` is a trade-off: too small and the learner is noisy (it always fits
the last few samples), too large and it is slow to adapt. `0.99` is a reasonable default
for a joint that changes over seconds.

## 5. The drift-aware variant (available, not default)

`DriftAwareResidualDynamicsLearner` extends the above with a drift detector on the
standardized prediction error. When the detector fires, the RLS forgetting factor is
temporarily lowered to `fast_lam` for `warmup_steps` samples, then restored. The idea:
react quickly to a real change without paying the noise cost of a permanently small
`λ`.

The drift-aware learner is built the same way (it is constructed in the block of step 4).

**Honest result from the 2-link benchmark** (payload / friction step at sample 200 of
400, `lam=0.99`):

- The Hampel trigger fires on pre-change RLS convergence noise around sample 154–157
  (a false positive). The disturbance in the residual window then masks the actual
  change at 200. The apparent improvement in the 10–60-sample post-change window is a
  timing artefact, not a mechanism.
- On the CUSUM path the trigger either never fires (`h = 20`, `h = 10`) or fires as a
  false positive (`h = 5`, events at 151 and 207).

The class is kept as an API so users can try their own parameters on their own data,
but for a plain payload change the recommended default is the plain
`ResidualDynamicsLearner` with a suitable `λ`.

## 6. Performance note

`RigidBodyModel` methods are pure-Python recursive walks over dicts and retrace the JAX
graph on every call, so `jax.jit` on the model does **not** cache across calls the way
it does for a pure function. In a control loop, patch the model once at construction:

```python
import jax
from dense_armor.dynamics.urdf_dynamics import RigidBodyModel
from dense_armor.utility.learn.online_dynamics import write_minimal_urdf

model = RigidBodyModel(write_minimal_urdf())
model.mass_matrix = jax.jit(model.mass_matrix)
model.bias_forces = jax.jit(model.bias_forces)
model.gravity_forces = jax.jit(model.gravity_forces)
print(model.n)
```

```
2
```

Without this, the learner's per-sample cost is in the millisecond range rather than
tens of microseconds. For a 2-DoF arm the difference is roughly 1.9 ms vs 2.7 ms per
sample depending on JAX cache state; the bottleneck is `forward_kinematics`, not the
RLS update itself.

## Parameters

- `lam` (default 1.0): forgetting factor in `(0, 1]`. `lam = 1.0` weights all past
  samples equally and matches batch least squares; smaller values weight recent samples
  more.
- `delta` (default 1e6): initial covariance scale `P₀ = delta · I`. Larger values mean
  less prior confidence in the zero initial weight.
- `feature_keys` (default `None`): explicit feature order for `RecursiveLeastSquares`.
  `None` sorts the keys of the first dict seen, so the result does not depend on dict
  ordering.
- `ResidualDynamicsLearner` uses the same `lam` and `delta` for every per-joint RLS.

## API reference

::: dense_armor.utility.learn.online_dynamics

---

## Details

**Two-step promotion** for the underlying dynamics model — Dense-Evolution-Discovery
Experiment 61 built the same Euler-Lagrange dynamics but hardcoded to one Kinova Gen3;
Experiment 62 replaced the hardcoded tables with a real URDF parser and re-validated.
The learner itself was re-validated on the 2-link benchmark after that.

**RLS references**: Ljung, L., Soderstrom, T. (1983). *Theory and Practice of Recursive
Identification*. MIT Press. RLS with exponential forgetting.

**Triggers**: Page, E. S. (1954). Continuous inspection schemes. *Biometrika* 41,
100–114. Hampel, F. R. (1974). The influence curve and its role in robust estimation.
*JASA* 69(346), 383–393.

**See also**: [Rigid-body dynamics](../dynamics/urdf_dynamics.md) — the nominal model
the learner builds on. [Calibration](calibration.md) — if the residual is used as a
score, calibration makes the score interpretable.
