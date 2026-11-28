# Robust filters

## The map of robust filters

Click a function to open its page, or the button at the top to go back to [Detection](../../detection/index.md).

<object type="image/svg+xml" data="../../assets/maps/robust_filters.svg" style="width:100%"></object>

You already have a recorded series: a joint velocity at 100 Hz, a temperature log, a
sensor reading. A few samples in it are wrong, and you want to know which ones and to see
the series with them replaced. That is what this page does.

<img width="2327" height="1498" alt="Diagram: a recorded series with one spike goes through four filters (Hampel, Tukey fences, Chauvenet, sigma clipping); pressure_valve, with weight 1/scale squared, returns the clean series, the flagged samples and the pressure score against the threshold" src="https://github.com/user-attachments/assets/ee3933b5-d2b3-4025-9590-adcef02ade07" />

The library has four classic batch filters for this job: **Hampel**, **Tukey fences**,
**Chauvenet** and **sigma clipping**. Each one takes the whole series, compares every sample
with its neighbours, and returns two things: the series with the flagged samples replaced by
a local estimate, and the list of positions it flagged. The four differ in how they decide
what "strange" means, so they often disagree on borderline samples.

`pressure_valve` settles the disagreement: it runs all four, weighs them by how confident each
one is, and returns one cleaned series, one list of flagged positions, and one pressure score
per sample. The sections below show each method in turn, then the combination.

Every function of this section returns two values, `clean` and `idx`. `clean` is the input
series where every unflagged sample keeps its value and every flagged sample is replaced by
the local median. `idx` is the list of flagged positions.

These filters look at samples after the current one (the window is centred on it). For a
series that is arriving now, where looking ahead is not allowed, use the causal versions on
[Streaming](../streaming/index.md).

## A first example: daily order execution cost

A desk watches the daily cost of executing an order: 500 days around 12 with a noise of 1.5, and one day, day 250, when it was 25.0. The four filters are run on the same stream.

```python
import numpy as np
from dense_armor.core.online.anomaly.robust_filters import hampel_filter, tukey_fences, chauvenet_criterion, sigma_clip

rng = np.random.default_rng(0)
cost = 12.0 + 1.5 * rng.standard_normal(500)
cost[250] = 25.0
for f in (hampel_filter, tukey_fences, chauvenet_criterion, sigma_clip):
    clean, idx = f(cost, radius=15)
    print(f.__name__, 250 in idx, round(float(clean[250]), 2), len(idx))
```

```
hampel_filter True 11.48 8
tukey_fences True 11.48 11
chauvenet_criterion True 11.48 10
sigma_clip True 11.48 8
```

All four flag day 250 and replace it with 11.48, the local median. They differ in how many ordinary days they also flag: 7 for `hampel_filter` and `sigma_clip`, 10 for `tukey_fences` and 9 for `chauvenet_criterion`.

## The pages of the section

| Page | What it shows |
|---|---|
| [hampel_filter](hampel_filter.md) | uses a window centred on each point and replaces outliers with the window median |
| [tukey_fences](tukey_fences.md) | uses the quartiles: a point is an outlier outside the fences |
| [chauvenet_criterion](chauvenet_criterion.md) | rejects when the expected number of samples that far out, N · P, is below 0.5 |
| [sigma_clip](sigma_clip.md) | removes values beyond `n_sigmas` standard deviations and repeats on what remains |
| [pressure_valve](pressure_valve.md) | runs all four, weighs them by how confident each one is, and returns one cleaned series, one list of flagged positions and one pressure score per sample |

## The same on a robot

```python
import numpy as np
from dense_armor.core.online.anomaly.robust_filters import (
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

All four catch the collision; they differ in how many ordinary samples they also flag.
