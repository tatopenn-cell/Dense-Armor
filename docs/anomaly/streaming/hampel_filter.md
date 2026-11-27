# HampelFilter

[← Streaming anomaly detection](index.md)

`HampelFilter` reads one value per call and returns the stream with outliers replaced by a clean value. When a reading is not an outlier it returns the reading unchanged; when it is an outlier it returns the median of the recent past. This is the same Hampel rule as `HampelScorer`, applied to a causal window, but the output is a cleaned series rather than a score. Other keys in the input are passed through unchanged.

![On a seeded noise stream with one injected spike, the fraction of injected spikes flagged, for 8 spike amplitudes (2 to 10 standard deviations) and 8 values of n_sigmas (2 to 6).](../../assets/streaming/hampel_filter_3d.png)

On a seeded noise stream with one injected spike, this surface shows how often the spike is flagged. The horizontal axis is the spike amplitude in standard deviations; the other axis is `n_sigmas`. Higher means the spike is caught more often.

## A first example: a daily cost of executing an order

A desk watches the daily cost of executing an order. Each day it tells the filter the new cost, then reads back the cleaned cost. The spike day is replaced by a normal value.

```python
import numpy as np
from dense_armor.core.online.anomaly.filters import HampelFilter

rng = np.random.default_rng(0)
cost = 12.0 + 1.5 * rng.standard_normal(500)
cost[250] = 25.0
f = HampelFilter(radius=15)
cleaned = []
for v in cost:
    f.learn_one({"x": float(v)})
    cleaned.append(f.transform_one({"x": float(v)})["x"])
print(round(cleaned[250], 2), round(cost[250], 2))
```

```
11.36 25.0
```

![The same seeded daily-cost stream, cleaned.](../../assets/streaming/hampel_filter_finance.png)

On the same seeded daily-cost stream, the spike day is replaced by a typical value.

## The same on a robot

A robot joint has a sensor glitch; `HampelFilter` removes it.

```python
import numpy as np
from dense_armor.core.online.anomaly.filters import HampelFilter

rng = np.random.default_rng(0)
f = HampelFilter(radius=5)
for v in rng.normal(0, 1, 20):
    f.learn_one({"v": float(v)})
out = f.transform_one({"v": 50.0})
print(round(out["v"], 2))
```

```
-0.43
```

The robot case comes last. The outlier `50.0` is replaced by a small value from the recent window.

## How it works

`learn_one` adds a value to a short causal window; `transform_one` does not learn. `transform_one` returns the value itself when it is not an outlier, or the median of the `2 * radius` samples strictly before it when it is. The same rule as `HampelScorer`. A missing value is passed through unchanged. Other keys are passed through unchanged. The `HampelFilter` is an `AnomalyDetector`-style `Transformer`; see [HampelScorer](hampel_scorer.md) for the score and `n_sigmas`.

## The parameters

`HampelFilter` is a `Transformer`. `learn_one(x)`, `transform_one(x)`, `n_missing`. Signature:

`(self, radius: int = 10, n_sigmas: float = 3.0, feature: str | None = None, eps: float = 1e-09)`

| Parameter | Default | What it does |
|---|---|---|
| `radius` | 10 | with a `2 * radius` causal window |
| `n_sigmas` | 3.0 | outlier when score above `n_sigmas` |
| `feature` | `None` | key to filter; `None` takes the feature with the smallest key, order-independent |
| `eps` | 1e-9 | below this scale the window is flat |

## API reference

::: dense_armor.core.online.anomaly.filters.HampelFilter

## Details

- **Sources.** Hampel, F. R. (1974). The influence curve and its role in robust estimation. *JASA* 69(346), 383-393. Davies, L., Gather, U. (1993). The identification of multiple outliers. *JASA* 88(423), 782-792. Batch `robust_filters.hampel_filter`. Same replacement rule as `HampelScorer`.
- **Causal.** `2 * radius` strictly before; value not part of own window. `n_missing`.
