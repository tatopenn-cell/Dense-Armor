# Robust filters (standalone detectors)

The four classic anomaly detectors of the previous page give one score per sample; on a
**recorded** series you can also run them as batch filters, and combine them into a
single verdict. `pressure_valve` does that: it takes the four scores and returns one
corrected value, one flag, and one effective threshold, using the physical units of the
input.

This page is for off-line cleanup. The causal streaming versions on
[Streaming](streaming.md) are the real-time counterparts.

## The signal

The same joint velocity used everywhere: 100 Hz, one collision spike, one slow drift.

```python
import numpy as np
rng = np.random.default_rng(42)
fs = 100
t = np.arange(3000) / fs
v = 0.5 + 0.1 * rng.standard_normal(len(t))
v[1500] = 2.5
v[2000:] += 0.2 * (np.arange(1000) / 1000)
```

## 1. Hampel filter

The batch Hampel filter uses a window centred on each point (it sees the future) and
replaces outliers with the window median. It is the off-line counterpart of the
[HampelScorer](streaming.md).

```python
import numpy as np
rng = np.random.default_rng(42)
fs = 100
t = np.arange(3000) / fs
v = 0.5 + 0.1 * rng.standard_normal(len(t))
v[1500] = 2.5
v[2000:] += 0.2 * (np.arange(1000) / 1000)
from dense_armor.anomaly.robust_filters import hampel_filter

clean = hampel_filter(v, radius=15, n_sigmas=3.0)
```

On the running example, `clean` removes the collision spike (index 1500 goes back to
about 0.5) and leaves the drift untouched, because the drift is a slow change of the
average and the local median follows it.

## 2. The other three

`tukey_fences`, `chauvenet_criterion` and `sigma_clip` in the same module do the same for
the other three classic rules. Each returns a cleaned series and the indices flagged.

```python
import numpy as np
rng = np.random.default_rng(42)
fs = 100
t = np.arange(3000) / fs
v = 0.5 + 0.1 * rng.standard_normal(len(t))
v[1500] = 2.5
v[2000:] += 0.2 * (np.arange(1000) / 1000)
from dense_armor.anomaly.robust_filters import hampel_filter

clean = hampel_filter(v, radius=15, n_sigmas=3.0)
from dense_armor.anomaly.robust_filters import (
    tukey_fences, chauvenet_criterion, sigma_clip,
)

clean_t, flag_t = tukey_fences(v, radius=15)
clean_c, flag_c = chauvenet_criterion(v, radius=15)
clean_s, flag_s = sigma_clip(v, radius=15, n_sigmas=3.0)
```

## 3. pressure_valve

`pressure_valve` combines the four detectors into one. It is a *minimum-variance*
combination (weights derived from each detector's residual variance), with a dynamic
threshold modulated by the Jensen–Shannon divergence between the four detectors'
verdicts. When the four detectors agree, the threshold tightens; when they disagree, it
loosens, so the filter is more conservative on ambiguous samples.

```python
import numpy as np
rng = np.random.default_rng(42)
fs = 100
t = np.arange(3000) / fs
v = 0.5 + 0.1 * rng.standard_normal(len(t))
v[1500] = 2.5
v[2000:] += 0.2 * (np.arange(1000) / 1000)
from dense_armor.anomaly.robust_filters import hampel_filter

clean = hampel_filter(v, radius=15, n_sigmas=3.0)
from dense_armor.anomaly.robust_filters import (
    tukey_fences, chauvenet_criterion, sigma_clip,
)

clean_t, flag_t = tukey_fences(v, radius=15)
clean_c, flag_c = chauvenet_criterion(v, radius=15)
clean_s, flag_s = sigma_clip(v, radius=15, n_sigmas=3.0)
from dense_armor.anomaly.robust_filters import pressure_valve

clean, flags, pressure, threshold = pressure_valve(v)
```

`pressure` is the per-sample pressure the four detectors exert; `threshold` is the
effective threshold at each sample. The four-detector combination beats every single
detector on the running example's collision spike, because it does not rely on any one
window's MAD being well-behaved.

## API reference

::: dense_armor.anomaly.robust_filters

---

## Details

The Jensen–Shannon modulation is what makes `pressure_valve` useful when the four
detectors disagree: on a genuinely ambiguous sample (spike or regime change?) the
threshold rises, so the filter errs on the side of not flagging. On a clear sample, the
threshold falls. This is why the "effective threshold" is a per-sample array and not a
single number.

The weights are the Lagrange-multiplier solution of the minimum-variance combination:
each detector gets a weight inversely proportional to its own residual variance on the
window. On quiet signals the four weights end up close to 1/4; on a signal where one
detector's residual is much larger (Chauvenet on a signal whose mean is itself drifting,
for example), that detector gets a smaller weight and the combination is dominated by
the other three.
