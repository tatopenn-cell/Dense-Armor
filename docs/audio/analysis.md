# Separating sources

This page splits a sound into parts. The method is the non-negative matrix factorisation of Lee and Seung (2001), applied to the magnitude of the short-time Fourier transform.

## 1. Factorise a matrix

`nmf(V, rank, n_iter, cost, seed)` writes a non-negative matrix `V` as the product `W H` of two non-negative matrices with `rank` columns and rows, by multiplicative updates: each update multiplies the matrix by a ratio, so a value that starts positive stays positive. With `cost="euclidean"` it follows Eq. (4) of the paper and with `cost="divergence"` Eq. (5); by the theorems of the paper the cost does not increase at any update. It returns `W`, `H` and the cost after every iteration.

```python
import numpy as np
from dense_armor.core.audio import nmf

rng = np.random.default_rng(1)
V = rng.random((12, 3)) @ rng.random((3, 20))
W, H, cost = nmf(V, 3, 300, "euclidean", seed=1)
print(all(cost[i + 1] <= cost[i] + 1e-9 for i in range(len(cost) - 1)))
print(bool(np.linalg.norm(V - W @ H) / np.linalg.norm(V) < 0.05))
```

```text
True
True
```

A denominator is floored at `1e-12` so that it never divides by zero, which the paper does not discuss.

## 2. Split a sound

`separate(audio, rank, n_fft, hop, n_iter)` takes the magnitude spectrogram of a mono sound, factorises it, and rebuilds each component with the phase of the mixture. The parts add back to the mixture only up to the error of the factorisation, because the phase is the one of the mixture.

```python
import numpy as np
from dense_armor.core.audio import Audio, separate

sr = 8000
t = np.arange(sr * 2) / sr
x = np.where(t < 1, np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 1000 * t))
parts = separate(Audio(x, sr), 2, 512, 256, 200)
peaks = sorted(round(float(np.abs(np.fft.rfft(c)).argmax() * sr / len(c))) for c in parts)
print(len(parts), peaks)
```

```text
2 [440, 1000]
```

A tone of 440 Hz in the first second and a tone of 1000 Hz in the second: the two parts have their spectral peaks at 440 and 1000 Hz.
