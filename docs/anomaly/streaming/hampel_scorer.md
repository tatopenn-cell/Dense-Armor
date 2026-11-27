# Hampel: distance from the median

[← Stream](index.md)

`HampelScorer` scores a new value by how far it is from the median of the recent window, measured in robust standard deviations: the robust spread is `1.4826 * MAD`. This is the classic Hampel rule, applied to a causal window of `2 * radius` samples strictly before the value. `is_outlier` is `True` when the score is above `n_sigmas`. It is an `AnomalyDetector`, and it never sees the future.

![On a seeded noise stream with one injected spike, the fraction of injected spikes flagged, for 8 spike amplitudes (2 to 10 standard deviations) and 8 values of n_sigmas (2 to 6).](../../assets/streaming/hampel_scorer_3d.png)

On a seeded noise stream with one injected spike, this surface shows how often the spike is flagged. The horizontal axis is the spike amplitude in standard deviations; the other axis is `n_sigmas`, the flagging threshold. Higher means the spike is caught more often.

## A first example: a daily cost of executing an order

A desk watches the daily cost of executing an order. Each day it tells the scorer the new cost, then asks if the day is an outlier.

```python
import numpy as np
from dense_armor.core.online.anomaly.filters import HampelScorer

rng = np.random.default_rng(0)
cost = 12.0 + 1.5 * rng.standard_normal(500)
cost[250] = 25.0
s = HampelScorer(radius=15)
for v in cost[:250]:
    s.learn_one({"x": float(v)})
print(s.is_outlier({"x": float(cost[250])}), round(s.score_one({"x": float(cost[250])}), 2))
```

```
True 7.92
```

![The same seeded daily-cost stream, flagged outlier in pink.](../../assets/streaming/hampel_scorer_finance.png)

On the same seeded daily-cost stream, the spike day is flagged; ordinary days are not.

## The same on a robot

```python
from dense_armor.core.online.anomaly.filters import HampelScorer

s = HampelScorer(radius=15, n_sigmas=3.0)
for v in [1.0, 1.2, 0.9, 1.1, 1.0, 0.8, 1.05]:
    s.learn_one({"v": v})
print(s.is_outlier({"v": 50.0}), s.is_outlier({"v": 1.1}))
```

```
True False
```

The robot case comes last. A reading of 50 is an outlier; a reading of 1.1 is ordinary.

## How it works

The score is how far the new value is from the usual value, measured in robust standard deviations. The usual value is the median of the recent window; the robust standard deviation is `1.4826 * MAD`. `learn_one` adds a value; `score_one` does not change anything.

The score is

$$\text{score} = \frac{|x - \mathrm{med}(w)|}{1.4826 \cdot \mathrm{MAD}(w)},$$

where $x$ is the new value; $w$ is the recent window; $\mathrm{med}(w)$ is its median; $\mathrm{MAD}(w)$ is the median distance from that median. The window holds `2 * radius` samples strictly before the value being scored; the value itself is never part of its own window. `HampelScorer` uses `2 * radius`; `streamDD` uses `radius * ref_mult`. `is_outlier` returns `True` when `score_one(x)` is above `n_sigmas`. `score_one` returns `0.0` for an unusable value or while fewer than four values have been learned. If the window is flat and the value differs from the median, the score is `positive infinity`, otherwise `0.0`.

This is `robust_filters.hampel_filter` applied to a causal window instead of a window centred on the value. `HampelScorer` is the streaming member; batch Hampel is on the robust-filters page.

## Comparison of the four causal scorers

Precision and recall on 5,000 seeded samples with 50 spikes of size 5-15 (1% contamination), window 30 (`radius=15`), and cost per sample:

| Scorer | stationary noise (σ=0.5): precision / recall | sine (amplitude 5, period 500) + same noise: precision / recall | µs per sample |
|---|---|---|---|
| HampelScorer | 0.459 / 1.000 | 0.318 / 1.000 | 90 |
| TukeyScorer | 0.303 / 1.000 | 0.195 / 1.000 | 75 |
| ChauvenetScorer | 0.282 / 0.980 | 0.167 / 1.000 | 28 |
| SigmaClipScorer | 0.588 / 1.000 | 0.370 / 1.000 | 72 |

Every scorer catches the spikes; precision is the tuning knob. A trend inside the window and a short noisy window lower precision. Raise `n_sigmas` or combine scores. At 100 Hz the cost is under 1% of the loop budget. This table appears once, here; the other scorer pages link to it. See [TukeyScorer](tukey_scorer.md), [ChauvenetScorer](chauvenet_scorer.md), [SigmaClipScorer](sigma_clip_scorer.md).

## The parameters

`HampelScorer` is an `AnomalyDetector`. `learn_one(x)`, `score_one(x)`, `is_outlier(x)`, `threshold`, `n_missing`. Signature:

`(self, radius: int = 10, n_sigmas: float = 3.0, feature: str | None = None, eps: float = 1e-09)`

| Parameter | Default | What it does |
|---|---|---|
| `radius` | 10 | with a `2 * radius` causal window |
| `n_sigmas` | 3.0 | `is_outlier` when the score is above `n_sigmas`; `threshold` returns it |
| `feature` | `None` | key of the feature to watch; `None` takes the feature with the smallest key, order-independent |
| `eps` | 1e-9 | below this scale the window is flat |

## API reference

::: dense_armor.core.online.anomaly.filters.HampelScorer

## Details

- **Sources.** Hampel, F. R. (1974). The influence curve and its role in robust estimation. *JASA* 69(346), 383-393. Davies, L., Gather, U. (1993). The identification of multiple outliers. *JASA* 88(423), 782-792. Pearson et al. (2016), *Generalized Hampel filters*, for the `1.4826 * MAD` rule. Batch `robust_filters.hampel_filter`.
- **Causal window.** `2 * radius` samples strictly before; `is_outlier` ↔ batch would flag; batch Hampel uses a window centred on the value (sees future); streaming is causal.
- **Scale.** robust spread is `1.4826 * MAD`; flat window, `positive infinity` if dev > `eps` `0.0` otherwise. `n_missing` counts values that are not a number or `None`.
- **Shared interface.** the four scorers share `learn_one` / `score_one`, feature selection, `is_outlier`.
