# DriftAwareResidualDynamicsLearner: forget faster when the weight changes

`DriftAwareResidualDynamicsLearner` is a [ResidualDynamicsLearner](residual_dynamics.md) with an alarm. It keeps a slow forgetting factor (`lam`) in steady state; when a detector sees the prediction error jump, it lowers the factor to `fast_lam` for a few samples and then restores it. The idea is to react quickly to a real change without paying the noise of a factor that is always small. It is available but it is not the default: on the 2-link benchmark the plain learner with a suitable `lam` was the recommended choice (see below). The overview is on [Online dynamics](online_dynamics.md).

This class needs a rigid-body model of a robot, so its example is a robot arm; the idea of forgetting on money is on the [RecursiveLeastSquares](recursive_least_squares.md) page.

## How it works

`DriftAwareResidualDynamicsLearner` extends the above with a drift detector on the
standardized prediction error. When the detector fires, the RLS forgetting factor is
temporarily lowered to `fast_lam` for `warmup_steps` samples, then restored. The idea:
react quickly to a real change without paying the noise cost of a permanently small
`λ`.

The drift-aware learner is built the same way as the plain one (see the [ResidualDynamicsLearner](residual_dynamics.md) page).

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

## The parameters

| Parameter | Meaning | Default |
|---|---|---|
| `model` | the `RigidBodyModel` of the robot | (required) |
| `lam` | the slow (steady-state) forgetting factor | 0.99 |
| `delta` | the initial covariance scale of every per-joint RLS | 1e6 |
| `fast_lam` | the fast forgetting factor, used during `warmup_steps` after a drift | 0.9 |
| `warmup_steps` | the number of samples after a drift during which `fast_lam` is used | 20 |
| `trigger` | `"hampel"` (default) or `"cusum"` | `"hampel"` |
| `hampel_radius`, `hampel_n_sigmas` | the parameters of the Hampel trigger | 20, 3.0 |
| `drift_radius`, `drift_ref_mult`, `drift_k`, `drift_h`, `drift_two_sided`, `drift_reference` | the parameters of the CUSUM trigger | 10, 3, 0.5, 20.0, `True`, `"adaptive"` |
| `detector_warmup` | the number of initial samples during which the trigger is not fed | 0 |

## API reference

::: dense_armor.core.online.learn.online_dynamics.DriftAwareResidualDynamicsLearner

---

## Details

- Triggers: Page, E. S. (1954). Continuous inspection schemes. *Biometrika* 41, 100–114. Hampel, F. R. (1974). The influence curve and its role in robust estimation. *JASA* 69(346), 383–393.
