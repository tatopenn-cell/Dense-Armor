# Filters

A filter takes a sound and keeps some of it. This page goes from the smallest filter, a biquad, to a filter with taps, a delay line, and a filter with poles whose coefficients can change at every sample. Every filter here is a primitive with `update(x)` for one sample and `update_many(values)` for a block, and every one can be given sound through [`AudioInput`](index.md).

## 1. A biquad

A biquad filter mixes the current sample with the two before it and the two previous outputs. `lowpass(500.0, 8000.0)` gives the coefficients `(b, a)` that keep what is below 500 Hz at 8000 samples per second, and `Biquad(b, a)` is the filter.

```python
import numpy as np
from dense_armor.core.audio import Audio, Biquad, lowpass

n = Audio(np.random.default_rng(0).standard_normal(8000), 8000)
f = Biquad(*lowpass(500.0, 8000.0))
z = f.update_many(n.x)
print(round(float(z.std() / n.x.std()), 3))
```

```text
0.373
```

White noise has the same strength at every frequency, so the filtered noise keeps a fraction of it. `highpass`, `bandpass` (peak gain 0 dB, or the constant skirt gain), `notch`, `peaking`, `lowshelf` and `highshelf` work the same way. The formulas are the ones of the Audio EQ Cookbook of R. Bristow-Johnson, a technical note and not a peer-reviewed paper. A frequency at or above half the sample rate raises `ValueError`, and so does a filter with a pole on or outside the unit circle.

## 2. A filter with taps, a delay, a convolution

`FIR` adds the last few inputs, each multiplied by a weight (the taps `h`). `convolve` does the same on a whole block, and `DelayLine` repeats a sample `d` steps later. Hayes and others (2023, Eq. 6) write the FIR filter as the convolution of the input with its taps, the same operation as a convolutional layer of a neural network.

```python
from dense_armor.core.audio import FIR, DelayLine, convolve

print(convolve([1.0, 2.0, 3.0], [1.0, 1.0]).tolist())
print(FIR([0.5, 0.5]).update_many([1.0, 1.0, 3.0]).tolist())
print(DelayLine(2).update_many([1.0, 2.0, 3.0, 4.0]).tolist())
```

```text
[1.0, 3.0, 5.0, 3.0]
[0.5, 1.0, 2.0]
[0.0, 0.0, 1.0, 2.0]
```

The filter with taps 0.5 and 0.5 averages two neighbouring samples, which smooths the sound.

## 3. A filter with poles

`AllPole(a)` is the filter `y[n] = x[n] - a_1 y[n-1] - ... - a_M y[n-M]` (Yu and others 2024, Eq. 1). It remembers its last `M` outputs, and a call can bring its own coefficients, so they can change at every sample.

```python
from dense_armor.core.audio import AllPole, reflection_to_lpc

a = reflection_to_lpc([0.5, 0.25])
f = AllPole(a)
print(a.tolist(), f.stable, f.update_many([1.0, 0, 0, 0]).round(4).tolist())
```

```text
[0.625, 0.25] True [1.0, -0.625, 0.1406, 0.0684]
```

`reflection_to_lpc(k)` builds the coefficients from reflection coefficients (Hayes and others 2023, Eq. 5). When every `|k|` is below 1 the filter is stable, so this is the safe way to build one.

The two papers do not write the sign of the filter in the same way. Hayes writes `y = e + sum a y` (his Eq. 4); with that sign, the coefficients of his Eq. 5 gave an unstable filter in 1570 of 2000 random cases. With the sign of Yu, all 2000 were stable. The module uses the sign of Yu for both, so the output of `reflection_to_lpc` goes straight into `AllPole`.
