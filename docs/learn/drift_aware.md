# DriftAwareResidualDynamicsLearner: forget faster when the weight changes

`DriftAwareResidualDynamicsLearner` is a [ResidualDynamicsLearner](residual_dynamics.md) with an alarm. It keeps a slow forgetting factor (`lam`) in steady state; when a detector sees the prediction error jump, it lowers the factor to `fast_lam` for a few samples and then restores it. The idea is to react quickly to a real change without paying the noise of a factor that is always small. It is available but it is not the default: on the 2-link benchmark the plain learner with a suitable `lam` was the recommended choice (see below). The overview is on [Online dynamics](online_dynamics.md).

This class needs a rigid-body model of a robot, so its example is a robot arm; the idea of forgetting on money is on the [RecursiveLeastSquares](recursive_least_squares.md) page.

## Gamestory

London Theater, deep night. Third night of the show. Everything is going smoothly: the puppeteer and his backstage assistant have now gotten the measure of the heavy apple and make the puppet move to perfection.

But halfway through the show comes a plot twist! In the story on stage, the puppet puts down the apple and catches a super-heavy brick on the fly! Suddenly the arm drops down: the numbers written up until a second ago on the assistant's little sheet are worth nothing anymore! To react right away to surprises, the assistant uses the DriftAwareResidualDynamicsLearner trick:

- **1. An alarm bell for mistakes (trigger):** On the table backstage, the assistant has placed an alarm bell (which can work in two ways, called Hampel or CUSUM). As soon as he sees that the puppet's arm is missing its mark badly and the mistakes become huge, RING! The bell goes off!
- **2. "Forget Everything Right Now" mode (fast_lam and warmup_steps):** As soon as the bell rings, the assistant erases the memory of the old apple in a flash. For a small number of moves (called warmup_steps), he uses a super-short memory (fast_lam) to learn only the weight of the new brick on the fly!
- **3. Returning to calm (lam):** Once those few emergency moves are over, the bell stops and the assistant goes back to his normal, steady pace (lam), continuing the show without swerving anymore. (A little theater secret: sometimes this bell is a bit too jumpy and rings even for a tiny wobble of the string, so often the puppeteer still prefers using the normal assistant with an already slightly short memory!)

In the end, what is DriftAwareResidualDynamicsLearner? It is the "alarm-bell" version of the puppeteer's assistant: an algorithm that keeps learning the extra strength for the robot, but when it realizes that weight or friction has suddenly changed, it activates a turbo mode to forget the past and get used to the new situation in a flash!

![Timeline of the alarms of the drift-aware learner on a 2-link arm whose payload goes from 0.5 to 1.5 kg at sample 200, for the Hampel trigger and for CUSUM with h 20, 10 and 5: Hampel rings at samples 4 and 15 and ten times between 201 and 221; CUSUM h 20 never rings; h 10 rings at 4, 130 and 210; h 5 rings nine times, including 127 and 138 before the change.](../assets/learn/drift_alarms.png)

Each tick is a sample at which the trigger rang, on a 2-link arm whose payload goes from 0.5 to 1.5 kg at sample 200. The Hampel trigger rings at samples 4 and 15, while the learner is still converging, and ten times between samples 201 and 221, right after the change. `CUSUM` with `h = 20` never rings. With `h = 10` it rings at 4, 130 and 210; with `h = 5` it rings nine times, including at 127 and 138, before the change. This is why the page says that the trigger is a parameter to try on your own data.

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
    options:
      docstring_style: numpy

---

## Details

- Triggers: Page, E. S. (1954). Continuous inspection schemes. *Biometrika* 41, 100–114. Hampel, F. R. (1974). The influence curve and its role in robust estimation. *JASA* 69(346), 383–393.
