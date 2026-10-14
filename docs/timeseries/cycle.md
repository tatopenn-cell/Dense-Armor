# Cycle residual

**In plain words:** When a robot repeats the same movement, its signals repeat too. This tool learns the usual shape of one repeat and tells you how much the current reading differs from it: small when all is normal, large when something breaks the pattern.

`CycleResidual` returns the difference between the current sample of a
periodic signal and the value the template predicts for the current phase.
When the motion repeats, the residual is small; when something breaks the
pattern, the residual jumps.

## 1. What a cycle residual is

A robot that repeats the same motion every second produces signals that
repeat with it. If the joint torque during a pick-and-place cycle looks the
same every cycle, then the *shape* of the torque is the normal, and any
difference from that shape is worth watching.

`CycleResidual` builds the shape in two steps. First it finds the period of
the stream, in samples, by autocorrelation of the recent buffer. Then it
keeps a **median template** of the last `n_cycles` periods: at each phase
inside the cycle, the median of the samples seen at that phase in the last
cycles. For every new sample the residual is

    r[t] = x[t] - template[t mod period]

The template is a running median, so a single outlier does not corrupt it,
and the model adapts when the cycle shape changes slowly.

## 2. Finding the period of a sine

A sine with a true period of `50` samples. `period=None` lets the estimator
find the period from the stream; after `1000` samples it has seen enough
cycles to be sure.

```python
import numpy as np
from dense_armor.utility.timeseries import CycleResidual

cr = CycleResidual(period=None, max_period=200, n_cycles=3)
for i in range(1000):
    cr.transform_one({"x": float(np.sin(2 * np.pi * i / 50))})
print(cr.period)
```

```
50
```

Once the period is known and the template is filled, the residual on a
stream that repeats exactly is zero:

```python
import numpy as np
from dense_armor.utility.timeseries import CycleResidual

cr = CycleResidual(period=50, n_cycles=3)
res = []
for i in range(500):
    out = cr.transform_one({"x": float(np.sin(2 * np.pi * i / 50))})
    res.append(out["residual"])
print(round(float(np.mean(np.abs(res[-100:]))), 4))
```

```
0.0
```

The residual is zero because the signal is the same every cycle. On a real
joint the residual is not zero even when healthy: it is a small number
that reflects the noise of the sensor and the small variation between
cycles. The *change* in the residual is what matters.

## 3. On a real arm

A `SyntheticArm` from a real Panda URDF gives `q`, `qd`, and `tau` at 50 Hz.
The arm repeats its motion every second, and a payload of 5 kg appears on
the last link at `t = 12 s`. The period of the joint-1 torque on this stream
is `200` samples (`4` base cycles of `50` samples each), so `period=200` and
`n_cycles=2` build the template from the first two periods.

```python
import numpy as np
from pathlib import Path
from dense_armor.utility.datasets import SyntheticArm
from dense_armor.utility.timeseries import CycleResidual

ds = SyntheticArm(
    Path("test/fixtures/urdf/panda.urdf"),
    period_s=1.0, rate_hz=50.0, n_cycles=16,
    fault_at_s=12.0, fault="payload", payload_mass=5.0, seed=0,
)
cr = CycleResidual(period=200, n_cycles=2)
res = []
for sig, _ in ds.stream():
    out = cr.transform_one({"x": float(sig["tau_1"])})
    res.append(out["residual"])
before = float(np.mean(np.abs(np.asarray(res[400:590]))))
after = float(np.mean(np.abs(np.asarray(res[700:800]))))
print(round(before, 4))
print(round(after, 4))
print(round(after / before, 1))
```

```
0.0477
12.6819
265.7
```

![Cycle residual of joint 1 on a SyntheticArm payload fault](../assets/timeseries/cycle_residual.png)

Before the fault the mean absolute residual is `0.048 N m`: the healthy
cycle repeats within the sensor noise. After the fault it is `12.68 N m`,
a factor of `266`. A control loop that watches this residual sees the
change at once, without knowing the payload model.

## Details

### References

The estimator is not taken from a single paper. The median-template idea
comes from the `benchmarks/anomaly_streaming.py` script that used a cycle
residual on the CASPER dataset, and the period detection by autocorrelation
is the standard approach for a periodic signal.

The median (instead of the mean) for the template is the choice from the
original benchmark script, kept because it resists a single bad sample
inside a cycle.

### Choices stated explicitly

- **The period can be given or found.** `period=200` fixes it at
  construction (fast, useful when the cycle is known). `period=None`
  finds it from the stream after `2 * max_period` samples, by
  autocorrelation of the buffer.
- **The template is the median of the last `n_cycles` periods, not the
  mean.** One outlier in one cycle does not corrupt the template.
- **The buffer is bounded.** Before the period is known the raw buffer is
  capped at `2 * max_period`; after the template is built the cap is
  `n_cycles * period + 2 * max_period`. The class declares
  `memory_class = "O(window)"`; the first delivery did not cap the buffer
  and grew with the stream, the cap was added in the same review round.
- **The residual is the raw difference**, not a score. A downstream
  detector (CUSUM, threshold) decides whether the residual is an anomaly.

### Measured cost

Per-sample latency is below `1 ms` (`budget_s = 1e-3 s`) on a desktop CPU,
measured as the p99 of `transform_one`. Memory is `O(window)`:
`n_cycles * period + 2 * max_period` floats in the buffer, plus a template
of `period` floats. For `period=200, n_cycles=2, max_period=200` the buffer
holds `2060` samples, and `_raw_memory_usage` does not grow after `10000`
samples.

### Development diary

The first delivery appended every sample to `_buf` and never pruned it,
so the memory class `O(window)` was declared but the actual behaviour was
`O(n_samples)`. A review measured `68346 B` after `5000` samples and
`18930 B` after `10000` (the drop is the buffer being capped), and the
class was given a `_prune` method called at the end of every
`transform_one`. The check on `check_estimator` verifies the buffer stays
at the cap; the test suite runs `2000` samples and reads `len(_buf)`.
