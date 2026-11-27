# KSWIN: compare the shapes, one alarm per change

[← Drift detectors](index.md)

`KSWIN` compares the newest 30 values with 30 values drawn at random from the older part of a window of 300, using the Kolmogorov–Smirnov test. The test measures how far apart the two cumulative histograms are: a small distance means that the two groups look alike, a large distance means that they do not. After an alarm the detector keeps only the newest values, so a single change gives a single alarm.

## First example

```python
import numpy as np
from dense_armor.core.online.drift.kswin import KSWIN

rng = np.random.default_rng(14)
x = list(rng.normal(0, 1, 400)) + list(rng.normal(2.5, 1, 400))
det = KSWIN(seed=0)
print([i for i, v in enumerate(x) if det.update(v).drift_detected])
```

```
[414]
```

In this example the first 400 values are normal noise, for instance the daily cost of an order after removing its usual level, and from the value 400 the mean moves up by 2.5 standard deviations. The detector gives exactly one alarm, at 414.

## How it works

The detector keeps a window of `window_size` values. When the window is full, at every new value, the newest `stat_size` values are the new sample, and `stat_size` values drawn at random from the older ones are the reference sample. The two samples go through a two-sample Kolmogorov–Smirnov test, and the detector flags a change when the p-value of the test is below `alpha`. The paper's values are the defaults: a window of 300, a sample of 30 and $\alpha = 10^{-4}$ (the paper reduces $\alpha$ with a Bonferroni–Dunn correction because many tests are made). No test runs before the window is full, so no alarm can come before the value 299.

The reset after an alarm is not in the paper. Without it, KSWIN would keep firing on the same change for as long as the old and the new values are in the window together. With it, only the newest `stat_size` values are kept, the window has to refill before the test runs again, and the refill uses values of the new regime only. A second change is therefore seen again: the same stream with a return to the old level at the value 800:

```python
import numpy as np
from dense_armor.core.online.drift.kswin import KSWIN

for seed in (0, 1, 2):
    r = np.random.default_rng(14)
    x = list(r.normal(0, 1, 400)) + list(r.normal(2.5, 1, 400)) + list(r.normal(0, 1, 400))
    det = KSWIN(seed=seed)
    print(seed, [i for i, v in enumerate(x) if det.update(v).drift_detected])
```

```
0 [414, 813]
1 [415, 816]
2 [415, 814]
```

There is one alarm for each of the two changes, a few values after each. `seed` seeds the random draw of the reference sample: the same seed gives the same alarms, and other seeds move them by a few values.

On 2000 values of normal noise, with no change, the detector gave no alarm (`KSWIN(seed=0)`, noise from `default_rng(5)`).

## What it sees

`KSWIN` compares distributions, not only averages, so it can see a change of the spread. A change of the spread from 1 to 3 standard deviations at the value 400, with the mean unchanged, ten seeded streams:

```python
import numpy as np
from dense_armor.core.online.drift.kswin import KSWIN

res = []
for s in range(10):
    r = np.random.default_rng(s)
    x = np.concatenate([r.normal(0, 1, 400), r.normal(0, 3, 400)])
    det = KSWIN(seed=0)
    res.append(next((i for i, v in enumerate(x) if det.update(v).drift_detected), None))
print(res)
```

```
[None, None, None, 426, None, None, None, 488, 605, 442]
```

It found the change in 4 streams out of 10, and late (the first alarm 26 to 205 values after it); the change of the mean of the first example was found 14 values after it. A value that is not a number is skipped and counted in `n_missing`.

## The parameters

| Parameter | Default | What it does |
|---|---|---|
| `alpha` | 0.0001 | the significance level of the test (the paper's value) |
| `window_size` | 300 | the length of the window (the paper's value) |
| `stat_size` | 30 | the size of each of the two samples that are compared (the paper's value) |
| `seed` | `None` | the seed of the random draw of the reference sample |

If `stat_size` is not smaller than `window_size` the constructor raises a `ValueError`: `stat_size (300) must be < window_size (300)`.

## API reference

::: dense_armor.core.online.drift.kswin

## Details

- **Source.** Raab, C., Heusinger, M., Schleif, F.-M. (2020), *Reactive soft prototype computing for concept drift streams*, Neurocomputing 416, 340-351, arXiv:2007.05432. The authors were working on learning systems that keep a set of representative prototypes instead of a full model, and they needed to know when the prototypes had become stale. The Kolmogorov–Smirnov test is a classical two-sample test, free of distributional assumptions, so it fits a stream where the shape can change and not only the mean. The parameters of the paper's experiments are a sample of 30, a window of 300 and $\alpha = 0.0001$.
- **What is not in the paper.** The window after an alarm: keeping the newest `stat_size` values gives one alarm per change.
