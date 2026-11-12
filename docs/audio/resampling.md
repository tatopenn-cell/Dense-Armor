# Resampling

This page changes the sample rate of a sound, and then makes the pitch glide without a click. The algorithm is the one of Falkenberg (2026), a preprint by one author.

## 1. A constant ratio

The `Resampler` reads the input at a step `q` per output sample. `q = 2` reads at twice the speed: half as many output samples, an octave up. `q = Fin / Fout` converts a sample rate. Between two input samples it interpolates linearly (Algorithm 3), so the stopband is not that of a windowed FIR filter.

```python
import numpy as np
from dense_armor.core.audio import Audio, Resampler, resample

t = np.arange(8000) / 8000
a = Audio(np.sin(2 * np.pi * 1000 * t), 8000)
y = resample(a, 2.0)
print(y.sr, y.n, round(y.duration, 3))
print(Resampler(1.5).update_many(np.arange(7.0)).round(2).tolist())
```

```text
4000 4000 1.0
[0.0, 1.5, 3.0, 4.5]
```

`resample(audio, ratio)` converts a whole mono `Audio`. `Resampler(q)` takes input in blocks, keeps the read position between calls, and gives the same output as the whole signal at once. `Resampler(q, gain=G)` multiplies the output by the gain `G` of Algorithm 3.

## 2. A ratio that changes

`set_ratio(q)` changes the step at any time and the read position stays continuous, so there is no click. Algorithm 4 gives the path of a pitch change: `pitch_ratio_transition(p_start, p_target, fin, fout, steps, mode)` returns the steps `q`, linear in the ratio (`"ratio-linear"`) or linear in cents (`"cents-linear"`, which is how the ear hears an interval). Give each step to `set_ratio`.

```python
from dense_armor.core.audio import pitch_ratio_transition, rational_ratio

q = pitch_ratio_transition(1.0, 2.0, 8000, 8000, 4, "cents-linear")
print(q.round(4).tolist(), rational_ratio(1.5, 4))
```

```text
[1.0, 1.1892, 1.4142, 1.6818, 2.0] (3, 2)
```

An octave in four equal steps of 300 cents. `rational_ratio(q, b)` writes the step as a fraction of two integers `D / U` (page 3 of the paper): `A = round(b q)`, then divide `A` and `b` by their greatest common divisor. Here `(3, 2)` is `D` and `U` with `D / U = 1.5`.
