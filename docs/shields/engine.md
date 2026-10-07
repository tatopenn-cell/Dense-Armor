# Adaptive engine (Orca Stage 1)

A signal that drifts slowly is not a spike. It is a sequence of samples, each of them
plausible on its own, that together move the average. A purely local detector misses
this by construction: each sample is within the local window's amplitude, so nothing
fires. `Orca`'s Stage 1 is the module that handles this case: it does not judge each
sample against a fixed baseline, it continuously **re-estimates the baseline** and
damps toward it with a smooth curve.

The module is `AdaptiveSignalStabilizer`, a causal recursive filter written as a
`jax.lax.scan` so it can run inside a JAX pipeline without leaving the JIT.

The examples use a 100 Hz channel with a spike at sample 1500 and a slow ramp from 0.5 to 0.7
starting at sample 2000.

## 1. Filter the series

```python
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from dense_armor.core.engine import AdaptiveSignalStabilizer

x = 0.5 + 0.1 * np.random.default_rng(42).standard_normal(3000)
x[1500] = 2.5
x[2000:] += 0.2 * (np.arange(1000) / 1000)

stabilizer = AdaptiveSignalStabilizer()
clean = stabilizer.filter_data_stream(np.asarray(x))
print(clean.shape, float(clean[1500]))
```

```
(3000,) 0.759195119486936
```

`clean` is the same series with the spike removed and the ramp followed but slightly
smoothed — the filter does not fight a sustained change, it damps the transient part of
it. At the spike position the value 2.5 comes out as 0.76: the filter damps the spike strongly
but does not replace it with the baseline.

## 2. What "adaptive" means here

The filter measures the **local noise** of the signal (a rolling volatility) and turns it into a
damping gain with a sigmoid:

$$k(n) = k_{min} + \frac{k_{max} - k_{min}}{1 + e^{-(n - 2)}}, \qquad n = \mathrm{clip}(\text{local noise},\, 0,\, 5).$$

- Low local noise: the gain stays near $k_{min}$ and the signal passes almost untouched.
- High local noise: the gain goes towards $k_{max}$ (about 8.78 by default) and the filter
  purifies strongly.
- Samples beyond a dynamic threshold (`anomaly_sigma_mult` times the rolling volatility) are
  damped selectively, preserving the L2 norm of the series so the damping cannot create local
  explosions.

The sigmoid is why the filter is smooth: there is no single value at which it switches from
"pass" to "reject".

## 3. Presets

Four calibrated parameter sets ship with the library:

| preset | when to use |
|---|---|
| `balanced_v2` | default, good across signals |
| `cifar10_best_v1` | tuned for image-grid-like inputs |
| `pure_1d_time_v1` | reactive; follows fast changes more closely |
| `cifar10_hardened_lyapunov` | hardened; damps more aggressively |

```python
import numpy as np
from dense_armor.core.preset import SIGNAL_STABILIZER_PRESETS
from dense_armor.core.engine import AdaptiveSignalStabilizer

x = 0.5 + 0.1 * np.random.default_rng(42).standard_normal(3000)
x[1500] = 2.5

stabilizer = AdaptiveSignalStabilizer(**SIGNAL_STABILIZER_PRESETS["pure_1d_time_v1"])
clean = stabilizer.filter_data_stream(np.asarray(x))
print(round(float(np.var(np.asarray(x) - clean)), 6))
```

```
0.003074
```

The presets behave differently on the same series: `pure_1d_time_v1` leaves a residual variance
of 0.0031 against 0.0061 for `balanced_v2` — it changes the signal about half as much, following
it more closely.

## 4. Where Orca uses it

`Orca.protect_and_forward` calls the stabilizer twice: once on the input tensor before
the model, once on the response after. The four-phase shield described on the
[Orca](orca.md) page is exactly this: purify, forward, purify, report trust scores.
Stage 2 (`Orca`'s Collatz-based gate) sits on top and decides how much of the damped
value to use versus how much of the raw value to trust on the input side; on the output
side the check is against the model's own response to the clean reference.

If you use `AdaptiveSignalStabilizer` on its own, you get the same soft damping without
the input/output distinction: a single cleaned stream.

## 5. Batch and multi-scenario use

When you have a whole batch of signals (or several scenarios side by side), the
stabilizer exposes batch methods that vmap over rows:

```python
import numpy as np
from dense_armor.core.engine import AdaptiveSignalStabilizer

batch = np.random.default_rng(0).standard_normal((4, 500))

s = AdaptiveSignalStabilizer()
clean_batch = s.filter_batch_scenarios(np.asarray(batch))
print(clean_batch.shape)
```

```
(4, 500)
```

`filter_batch_scenarios` uses a shared calibration for the whole batch.
`filter_batch_scenarios_independent` gives each row its own thresholds — useful when
the rows come from different sensors with different amplitudes.

## API reference

::: dense_armor.core.engine

---

**See also**: [`Orca`](orca.md), which uses `AdaptiveSignalStabilizer` as Stage 1,
followed by a Collatz-based gate (Stage 2) deciding how much to damp toward the clean
reference. This is also the engine exercised directly by the adversarial test suite
(`test/test_boundA-E.py`) behind the
[robustness numbers on the home page](../index.md).
