# Robust filters (standalone detectors)

The four classic anomaly detectors of the previous page give one score per sample; on a
**recorded** series you can also run them as batch filters, and combine them into a single
verdict. `pressure_valve` does that: it merges the four into one cleaned value, one flag, one
pressure score and one threshold per sample.

This page is for off-line cleanup of a series already recorded. The causal streaming versions on
[Streaming](streaming.md) are the real-time counterparts.

## 1. Hampel filter

The batch Hampel filter uses a window centred on each point (it sees the future) and replaces
outliers with the window median.

```python
import numpy as np
from dense_armor.utility.anomaly.robust_filters import hampel_filter

rng = np.random.default_rng(42)
v = 0.5 + 0.1 * rng.standard_normal(3000)
v[1500] = 2.5
clean, idx = hampel_filter(v, radius=15, n_sigmas=3.0)
print(1500 in idx, round(float(clean[1500]), 3), len(idx))
```

```
True 0.498 35
```

A joint velocity at 100 Hz around 0.5 with noise 0.1 and one collision at sample 1500. The
collision is flagged and replaced by the local median, 0.498. The filter also flags 35 samples in
total: with a window of 31 points and 3 sigma, some ordinary noise values fall out. Every function
on this page returns `(clean, idx)`: the cleaned series (raw value where normal, local median
where flagged) and the list of flagged indices. The rule is the Hampel identifier
$|x_k - m_k| > t\,S_k$ with $S_k = 1.4826 \cdot \mathrm{MAD}$ (Pearson et al., 2016, eqs. 3 and 4;
explained step by step on [Streaming](streaming.md)).

## 2. The other three

`tukey_fences`, `chauvenet_criterion` and `sigma_clip` do the same for the other three classic
rules.

```python
import numpy as np
from dense_armor.utility.anomaly.robust_filters import (
    hampel_filter, tukey_fences, chauvenet_criterion, sigma_clip)

rng = np.random.default_rng(42)
v = 0.5 + 0.1 * rng.standard_normal(3000)
v[1500] = 2.5
for f in (hampel_filter, tukey_fences, chauvenet_criterion, sigma_clip):
    clean, idx = f(v, radius=15)
    print(f.__name__, 1500 in idx, len(idx))
```

```
hampel_filter True 35
tukey_fences True 65
chauvenet_criterion True 48
sigma_clip True 24
```

All four catch the collision; they differ in how many ordinary samples they also flag. Tukey uses
the quartiles (outside $[Q_1 - 1.5\,\mathrm{IQR},\, Q_3 + 1.5\,\mathrm{IQR}]$), Chauvenet rejects
when the expected number of samples that far out, $N \cdot P$, is below 0.5, sigma clipping
removes values beyond `n_sigmas` standard deviations and repeats on what remains.

## 3. pressure_valve: four estimates combined

Each of the four methods gives its own estimate of the local centre and of its uncertainty
(scale). `pressure_valve` combines them with the weights that make the combination as precise as
possible.

```python
import numpy as np

s = np.array([0.10, 0.12, 0.10, 0.30])
w = (1 / s**2) / np.sum(1 / s**2)
print(np.round(w, 3), round(float(1 / np.sqrt(np.sum(1 / s**2))), 4))
```

```
[0.356 0.248 0.356 0.04 ] 0.0597
```

The weights minimise the variance of $\sum_k w_k c_k$ under $\sum_k w_k = 1$ (Lagrange
multiplier):

$$w_k = \frac{1/s_k^2}{\sum_j 1/s_j^2}, \qquad s = \frac{1}{\sqrt{\sum_j 1/s_j^2}}, \qquad p = \frac{|x - \sum_k w_k c_k|}{s},$$

where $c_k$ and $s_k$ are the centre and scale of method $k$ (Chauvenet: mean and standard
deviation; sigma clipping: clipped mean and standard deviation; Hampel: median and
$1.4826 \cdot \mathrm{MAD}$; Tukey: median and $\mathrm{IQR}/1.349$), $s$ is the scale of the
combination and $p$ the pressure of the sample. In the example four methods have scales 0.10,
0.12, 0.10, 0.30: the noisy fourth one gets weight 0.04, and the combined scale 0.0597 is
narrower than any single one. This is the minimum-variance (Gauss–Markov) combination of
independent estimates; no weight is chosen by hand.

## 4. pressure_valve on the joint signal

```python
import numpy as np
from dense_armor.utility.anomaly.robust_filters import pressure_valve

rng = np.random.default_rng(42)
v = 0.5 + 0.1 * rng.standard_normal(3000)
v[1500] = 2.5
clean, idx, p, thr = pressure_valve(v, radius=15)
print(idx, round(float(p[1500]), 1), round(float(np.median(thr)), 2))
```

```
[949, 1500, 2707] 42.7 8.46
```

Three samples flagged out of 3,000: the collision (pressure 42.7) and two noise values. The
threshold is not fixed: at each sample the local window is compared with a wider reference window
(`radius * ref_mult`) by their Jensen–Shannon divergence $\mathrm{JSD} \in [0, 1]$, and

$$\text{threshold}_i = \text{soglia\_pressione} \cdot (1 + k_{\text{molla}} \cdot \mathrm{JSD}_i).$$

When the two windows look alike (steady noise) JSD is near 0 and the threshold stays at its base
value 8.0; when they differ (a real change of level is under way) the threshold widens, so the
filter does not attack a genuine transition. Here the median threshold is 8.46.

## API reference

::: dense_armor.utility.anomaly.robust_filters

---

## Details

- **Base threshold 8.0.** On pure stationary noise (N = 300, fixed seed) the pressure percentiles
  are 50 % = 1.4, 95 % = 4.6, 99 % = 6.0, 99.9 % = 7.6, so 8.0 leaves about one false positive in
  300 while a true outlier (pressure about 88) stays far above it.
- **Why Chauvenet keeps its non-robust scale.** It is the original 1863 formulation (mean and
  standard deviation); when its window already contains an outlier its scale inflates, and the
  minimum-variance weights lower its influence automatically, without discarding it.
- **Jensen–Shannon on small windows.** `_jensen_shannon` uses adaptive bins and Laplace smoothing;
  a naive version with fixed bins gave a higher divergence on stationary noise than near a real
  transition.
- **Sources.** Pearson, R. K. et al. (2016), "Generalized Hampel filters", *EURASIP J. Adv. Signal
  Process.*, eqs. 3–4. Chauvenet, W. (1863), *A Manual of Spherical and Practical Astronomy*,
  vol. 2. Tukey, J. W. (1977), *Exploratory Data Analysis*.
