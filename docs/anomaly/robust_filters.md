# Robust filters (standalone detectors)

You already have a recorded series: a joint velocity at 100 Hz, a temperature log, a
sensor reading. A few samples in it are wrong, and you want to know which ones and to see
the series with them replaced. That is what this page does.

<img width="2327" height="1498" alt="deepseek_mermaid_20261017_d1e5de" src="https://github.com/user-attachments/assets/ee3933b5-d2b3-4025-9590-adcef02ade07" />


The library has four classic batch filters for this job: **Hampel**, **Tukey fences**,
**Chauvenet** and **sigma clipping**. Each one takes the whole series, compares every sample
with its neighbours, and returns two things: the series with the flagged samples replaced by
a local estimate, and the list of positions it flagged. The four differ in how they decide
what "strange" means, so they often disagree on borderline samples.

`pressure_valve` settles the disagreement: it runs all four, weighs them by how confident each
one is, and returns one cleaned series, one list of flagged positions, and one pressure score
per sample. The sections below show each method in turn, then the combination.

These filters look at samples after the current one (the window is centred on it). For a
series that is arriving now, where looking ahead is not allowed, use the causal versions on
[Streaming](streaming.md).

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

The stream is a joint velocity at 100 Hz: a level of 0.5, Gaussian noise with standard
deviation 0.1, and one collision at sample 1500 that drives the reading to 2.5.

The Hampel filter flags the collision and replaces it with the local median, 0.498. It also
flags 34 ordinary samples: with a 31-point window and a threshold of 3 in the Hampel rule,
some noise values fall beyond it. The total is 35 flagged positions.

Every function on this page returns two values, `clean` and `idx`. `clean` is the input
series where every unflagged sample keeps its value and every flagged sample is replaced by
the local median. `idx` is the list of flagged positions.

The rule is `|x_k - m_k| > t * S_k`, where `m_k` is the median of the window centred on
sample `k`, `S_k = 1.4826 * MAD` is the robust estimate of the standard deviation of that
window, and `t` is the threshold (Pearson et al., 2016, eqs. 3 and 4; step by step on
[Streaming](streaming.md)).

## 2. The other three

`tukey_fences`, `chauvenet_criterion` and `sigma_clip` implement the other three classic
rules, with the same `(clean, idx)` return as `hampel_filter`.

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

Each of the four methods (Hampel, Tukey fences, Chauvenet, sigma clipping) returns two
numbers for every sample. The first is a **centre** `c_k`: the method's estimate of the
true value at that sample. The second is a **scale** `s_k`: the method's estimate of its own
uncertainty. Both come from the window around the sample, and the pair differs per method:

| method | centre `c_k` | scale `s_k` |
|---|---|---|
| Hampel | median | 1.4826 * MAD |
| Tukey fences | median | IQR / 1.349 |
| Chauvenet | mean | standard deviation |
| sigma clipping | clipped mean | standard deviation |

The four scales can differ a lot: a method whose window already contains an outlier reports
a large scale, a method whose window is clean reports a small one.

`pressure_valve` weights each centre by the inverse of its variance, so a method that is
sure of itself counts more than one that is not. The weight of method `k` is

$$w_k = \frac{1/s_k^2}{\sum_j 1/s_j^2},$$

the minimum-variance (inverse-variance) combination of independent estimates. No weight is
chosen by hand.

```python
import numpy as np

s = np.array([0.10, 0.12, 0.10, 0.30])
w = (1 / s**2) / np.sum(1 / s**2)
print(np.round(w, 3), round(float(1 / np.sqrt(np.sum(1 / s**2))), 4))
```

```
[0.356 0.248 0.356 0.04 ] 0.0597
```

A method with a small scale gets a large weight; a method with a large scale gets a small
one. The combined estimate is more precise than any single method, and its scale is

$$s = \frac{1}{\sqrt{\sum_j 1/s_j^2}}.$$

The pressure of a new sample `x` is how far it lies from the combined centre, measured in
units of the combined scale:

$$p = \frac{\left|x - \sum_k w_k c_k\right|}{s}.$$

In the code above the four scales are 0.10, 0.12, 0.10 and 0.30. The method with scale 0.30
gets weight 0.04; the two methods with scale 0.10 get weight 0.36 each. The combined scale is
0.0597, smaller than any of the four. The four weights are the solution of the calculation
above, not numbers chosen by hand. This is the minimum-variance combination of independent
estimates.

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

Of the 3,000 samples, the filter flags three: the collision at 1500 (pressure 42.7) and two
noise values at 949 and 2707.

The threshold is not the same at every sample. At each sample it depends on how different
the local window looks from a wider reference window of size `radius * ref_mult`. The
difference is measured by the Jensen–Shannon divergence, a number in `[0, 1]`: near `0` when
the two windows have the same shape (steady noise), near `1` when they have different shapes
(a real change of level is under way). The threshold is

$$\text{threshold}_i = \text{soglia\_pressione} \cdot (1 + k_{\text{molla}} \cdot \mathrm{JSD}_i),$$

where `soglia_pressione` (default 8.0) and `k_molla` are the two parameters of the code.

When the two windows look the same, `JSD` is near `0` and the threshold stays at its base
value, 8.0. When they differ, the threshold rises and the filter tolerates a larger
deviation, so a genuine change of level is not flagged as an anomaly. On the 3,000 samples
above the median threshold is 8.46.

## API reference

::: dense_armor.utility.anomaly.robust_filters

---

## Details

- **Why the base threshold is 8.0.** On pure stationary noise (N = 300, fixed seed) the
  pressure percentiles are 50% = 1.4, 95% = 4.6, 99% = 6.0, 99.9% = 7.6. A base
  threshold of 8.0 sits above the 99.9th percentile, so on average fewer than one sample
  in 300 crosses it. A true outlier reaches a pressure well above the base threshold: the
  collision in the example above, for instance, has pressure 42.7.

- **Why Chauvenet keeps its non-robust scale.** The Chauvenet rule is used here in its
  original 1863 formulation, with the mean and the standard deviation of the window.
  When the window contains an outlier, that scale inflates, and the minimum-variance
  weights automatically lower Chauvenet's influence. Replacing Chauvenet's mean and
  standard deviation with a robust centre and scale would change the rule; keeping the
  original and letting the weighting handle the outlier is the choice made here.

- **Jensen–Shannon on small windows.** `_jensen_shannon` uses adaptive bins and Laplace
  smoothing. A naive version with fixed bins gave a higher divergence on stationary
  noise than on a real transition, the opposite of what the threshold formula needs: the
  threshold would rise on noise and would not rise near a transition. Adaptive bins fix
  this by sizing the histogram to the range the window actually covers.

- **Sources.** Pearson, R. K. et al. (2016), "Generalized Hampel filters", *EURASIP J.
  Adv. Signal Process.*, eqs. 3–4. Chauvenet, W. (1863), *A Manual of Spherical and
  Practical Astronomy*, vol. 2. Tukey, J. W. (1977), *Exploratory Data Analysis*.
