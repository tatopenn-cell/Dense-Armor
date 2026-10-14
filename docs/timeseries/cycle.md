# Cycle residual

**In plain words:** When a robot repeats the same movement, its signals repeat too. This tool learns the usual shape of one repeat and tells you how much the current reading differs from it: small when all is normal, large when something breaks the pattern.

`CycleResidual` returns the difference between the current sample of a
periodic signal and the value the template predicts for the current phase.
When the motion repeats, the residual is small; when something breaks the
pattern, the residual jumps. Until the template exists the residual is `nan`
and the key `ready` is `0.0`.

## 1. What a cycle residual is

A robot that repeats the same motion every second produces signals that
repeat with it. If the joint torque during a pick-and-place cycle looks the
same every cycle, then the *shape* of the torque is the normal, and any
difference from that shape is worth watching.

`CycleResidual` builds the shape in two steps. First it finds the period of
the stream, in samples, by autocorrelation of its first `2 * max_period`
samples (or takes the `period` you give). Then it keeps the last `n_cycles`
complete cycles and takes, at each phase inside the cycle, the **median** of
the samples seen at that phase. For every new sample the residual is

    r[t] = x[t] - template[phase of t]

The template is recomputed every time a cycle is complete: the oldest cycle
leaves and the new one enters. A single outlier does not corrupt the median,
and the template follows a signal whose shape changes. With `search > 0` each
complete cycle is compared with the template shifted by up to `search`
samples and stored back aligned, so a motion that starts a few samples late
keeps the same template.

## 2. Finding the period of a sine

A sine with a true period of `50` samples. `period=None` lets the estimator
find the period from the stream; it looks at the first `2 * max_period`
samples, here `400`.

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
12.6829
265.8
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
  construction. `period=None` finds it after `2 * max_period` samples by
  autocorrelation; the default `max_period = 1000` therefore needs 2000
  samples, and on the 800-sample stream of section 3 it finds no period
  (with `max_period = 250` it finds exactly `200`, at sample 499).
- **The template is the median of the last `n_cycles` complete cycles, not
  the mean.** One outlier in one cycle does not corrupt the template.
- **Before the template exists the residual is `nan`**, with `ready = 0.0`;
  `ready` is `1.0` on every sample whose residual is a number. The first
  residual appears after `n_cycles * period` samples when the period is
  given.
- **Memory is one array of `n_cycles * period` values.** The raw samples
  held to find the period are dropped as soon as the period is known.
- **The phase shift is searched once per complete cycle** and applied to the
  next one; it is off by default (`search = 0`).
- **The residual is the raw difference**, not a score. A downstream
  detector (CUSUM, threshold) decides whether the residual is an anomaly.

### Measured cost

Per-sample latency is `3` microseconds median and `40` microseconds at the
99th percentile (`budget_s = 1e-3 s`), for `period=50, n_cycles=2`, over 3000
samples on a Windows desktop CPU. Memory is `n_cycles * period` floats
(`400` for `period=200, n_cycles=2`) plus the template of `period` floats,
constant after the period is known.

### Development diary

The first delivery appended every sample to a buffer and never pruned it.
The second capped the buffer but built the template once and never updated
it, returned `0.0` before the template existed (a CUSUM on the residual rang
three times before the fault, at samples 444, 447 and 453), and had no phase
search. A review replaced it with the cycle store of this page: a rolling
median, `nan` plus `ready` before the template, and the `search` argument.
On the stream of section 3, with the period found, there are no alarms
before the fault and the first alarm is at sample 600, `12.0 s`.
