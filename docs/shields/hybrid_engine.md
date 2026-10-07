# Hybrid engine (Armatura's core)

`Armatura` is the front door for a 1D series: it takes an array, returns a cleaned
array. Behind it sits `hybrid_engine`, the module that actually decides, sample by
sample, whether a value is a genuine reading or a bad one — and if bad, what to put
in its place.

Each sample is in one of two states: **dynamic** (a genuine movement of the signal, kept as
measured) or **static** (replaced by the median of the previous window). A binary trigger decides,
from how the sample relates to the recent past.

## 1. Call the engine

```python
import numpy as np
from dense_armor.core.hybrid_engine import hybrid_shield

rng = np.random.default_rng(42)
x = 0.5 + 0.1 * rng.standard_normal(300)
x[100], x[200] = 9999.0, float("nan")
clean, trigger, info = hybrid_shield(x)
print(round(float(clean[100]), 3), trigger[100], round(float(trigger.mean()), 2), info)
```

```
0.463 0.0 0.88 {'fallback_triggered': True, 'adaptive_radius_used': 20}
```

The engine returns the cleaned series, the trigger per sample (1 = dynamic, kept; 0 = static,
replaced) and a small dict: whether any sample fell back, and the window radius used,
`min(20, max(3, n // 3))` (20 here). The spike at 100 has trigger 0 and becomes 0.463.

## 2. The trigger, symbol by symbol

For sample $x_i$, with $b_i$ the mean of the previous `R` samples (or `riferimento[i]` if a
reference series is given), $s_i$ the local scale (standard deviation of the differences in the
window) and $g_i$ the sign of the previous step $x_{i-1} - x_{i-2}$:

$$\Phi_i = \mathrm{clip}\Big(0.6\,\tfrac{1 + \mathrm{sign}(x_i - b_i)\,g_i}{2} + 0.4\,\big(1 - \tfrac{|x_i - b_i|}{s_i}\big),\; 0,\; 1\Big),$$

$$v_i = 5\,\mathrm{clip}\big(\ln(|x_i| / |b_i|),\, -5,\, 5\big)\,\Phi_i, \qquad \text{trigger}_i = \big[\,|v_i| > 0.01\,\big].$$

- $\Phi_i$ mixes **alignment** (does the change continue the direction of the last step?) and
  **coherence** (is the change small compared with the local scale?).
- $v_i$, the *dynamic vector*, is the logarithmic change of magnitude weighted by $\Phi_i$.
- A huge spike has $|x_i - b_i| \gg s_i$, so the coherence term drives $\Phi_i$ to 0, $v_i = 0$,
  trigger 0: the sample is replaced by the window median.

The first two samples are never tested (the step direction needs two previous points).

## 3. What the trigger also replaces

On the example the trigger is 1 on 88 % of the samples. Besides the spike and the NaN, it also
classifies as static ordinary samples that hardly differ from the baseline ($v_i$ below 0.01),
and replaces them with the window median:

```python
import numpy as np
from dense_armor.core.hybrid_engine import hybrid_shield

rng = np.random.default_rng(42)
x = 0.5 + 0.1 * rng.standard_normal(300)
x[100], x[200] = 9999.0, float("nan")
clean, trigger, info = hybrid_shield(x)
d = np.abs(clean - x)[np.isfinite(x) & (np.arange(300) != 100)]
print(int((d > 1e-12).sum()), round(float(d.max()), 4))
```

```
35 0.2935
```

35 ordinary samples out of 298 change, by up to 0.29 (about three times the noise level), and they
are not counted as anomalies by `Armatura`. Keep this in mind on smooth signals.

## 4. Non-finite values

`Inf` is turned into NaN and every NaN is filled from its local neighbourhood **before** the
trigger runs, so the windows never contain a non-finite value.

## API reference

::: dense_armor.core.hybrid_engine

---

## Details

The engine generalises the logic already verified in Dense-Evolution's own
`ia_utils.vector_healing.enhanced_dense_healing_hybrid`. The trigger and
the NaN handling are the same as there; this version computes all samples in one vectorised
call (`jax.vmap`).

**See also**: [`Armatura`](armatura.md) — the wearable class built on this engine.
