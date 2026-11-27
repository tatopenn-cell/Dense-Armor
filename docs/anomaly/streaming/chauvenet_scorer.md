# ChauvenetScorer

[← Streaming anomaly detection](index.md)

`ChauvenetScorer` scores a new value with `N * P(|Z| >= z)`, the expected number of observations in a normal sample of size N that would lie at least this far from the mean. Unlike the other scorers, a **low** score means anomalous: `is_outlier` is `True` when the score is below `threshold` (`0.5`). `mu` and `sigma` are the ordinary mean and standard deviation of a causal window of `2 * radius` samples strictly before the value. It is an `AnomalyDetector`, causal.

It looks only at the `2 * radius` samples before the value being scored, like the other scorers; why the batch versions differ is explained on the page of [HampelScorer](hampel_scorer.md).

![The base 10 logarithm of the score for 8 values of z from 0.5 to 5 and 8 window sizes from 10 to 300; the orange plane is the threshold 0.5.](../../assets/streaming/chauvenet_scorer_3d.png)

The score is the expected number of values this far from the mean, so it grows with the window size and falls as the value moves away. The same distance is rejected in a short window and kept in a long one.

## A first example: a daily cost of executing an order

A desk watches the daily cost of executing an order, around 12 with a noise of 1.5. The same cost of 16.5, about 3 standard deviations above, is judged with a window of 30 days and with a window of 300 days.

```python
import numpy as np
from dense_armor.core.online.anomaly.filters import ChauvenetScorer

rng = np.random.default_rng(3)
for radius in (15, 150):
    c = ChauvenetScorer(radius=radius)
    for v in rng.normal(12.0, 1.5, 2 * radius):
        c.learn_one({"x": float(v)})
    print(2 * radius, round(c.score_one({"x": 16.5}), 2), c.is_outlier({"x": 16.5}))
```

```
30 0.29 True
300 0.89 False
```

With 30 days the score is 0.29, below 0.5: Chauvenet rejects the value. With 300 days it is 0.89: the value is kept. The longer the window, the more values this far from the mean are expected, so one such value is less surprising. Remember the direction: for Chauvenet a low score means an outlier.

![The expected count N times P as a curve of z for a window of 30 and one of 300 days, with the threshold 0.5 and the two scores of the example as pink points.](../../assets/streaming/chauvenet_scorer_finance.png)

The orange curve is a window of 30 and the blue one a window of 300. The grey line is the threshold 0.5. At z = 3 the orange curve is below it and the blue one above it.

## The same on a robot

```python
from dense_armor.core.online.anomaly.filters import ChauvenetScorer

c = ChauvenetScorer(radius=5)
for v in [10, 11, 9, 10, 12, 10, 11, 9, 10]:
    c.learn_one({"v": float(v)})
print(round(c.score_one({"v": 10.0}), 2), round(c.score_one({"v": 14.0}), 4), c.is_outlier({"v": 14.0}))
```

```
7.28 0.0003 True
```

A reading at the centre of the window scores high; a reading far from it scores almost zero and is rejected. A low score is an outlier.

## How it works

Chauvenet rejects when $N \cdot \mathrm{erfc}(z/\sqrt 2) < 0.5$; its `score_one` returns $N \cdot P$, so for Chauvenet a **low** score means anomalous.

The score is the expected number of points in a normal sample that would be at least this far out; too small means too unlikely. Chauvenet asks: in a normal sample, would we expect less than half a point this far out? If yes, reject. The direction is opposite to the other scorers, which is why `score_is_inverted` is set.

The score is

$$\mathrm{score} = N \cdot P(|Z| \ge z), \qquad z = \frac{|x - \mu|}{\sigma}, \qquad P = \mathrm{erfc}\!\left(\frac{z}{\sqrt{2}}\right),$$

where $x$ is the new value; $\mu$ is the window mean; $\sigma$ is the window standard deviation; $N$ is `2 * radius`, the number of samples in the causal window strictly before the value. `mu` and `sigma` are the ordinary mean and standard deviation, not robust; a window that already has outliers can distort them. `is_outlier` is `score_one(x)` below `threshold`, with `threshold = 0.5`, Chauvenet's value. The window is `2 * radius` strictly before; it is never part of its own window. `score_one` returns `positive infinity` while fewer than four values have been learned or for an unusable value. On a flat window, `score` is `N` if the value does not deviate, `0.0` if it does. `score_one` does not change anything.

`ChauvenetScorer` is `robust_filters.chauvenet_criterion` applied to a causal window. See the comparison on [HampelScorer](hampel_scorer.md).

**Not usable with the gates.** `score_is_inverted = True`, so `AnomalyGate` and `Protected` refuse it: their constructors raise `ValueError`. Use `HampelScorer`, `TukeyScorer`, `SigmaClipScorer`, or call `is_outlier` directly. (This behavior is in the source.)

## The parameters

`ChauvenetScorer` is an `AnomalyDetector`. `learn_one(x)`, `score_one(x)`, `is_outlier(x)`, `n_missing`. Signature:

`(self, radius: int = 10, threshold: float = 0.5, feature: str | None = None, eps: float = 1e-09)`

| Parameter | Default | What it does |
|---|---|---|
| `radius` | 10 | with a `2 * radius` causal window; also `N` in `N * P` |
| `threshold` | 0.5 | `is_outlier` when score `< threshold`; Chauvenet's value `0.5` |
| `feature` | `None` | key of the feature to watch; `None` takes the feature with the smallest key, order-independent |
| `eps` | 1e-9 | below this scale the window is treated as flat |

No `n_sigmas`. `score_is_inverted = True`.

## API reference

::: dense_armor.core.online.anomaly.filters.ChauvenetScorer

## Details

- **Sources.** Chauvenet, W. (1863). *A Manual of Spherical and Practical Astronomy*, Vol. II, Appendix on the Method of Least Squares. Batch `robust_filters.chauvenet_criterion`.
- **Direction.** reject when `N * P < threshold`; opposite of "high score = anomalous"; not usable with `AnomalyGate` / `Protected`.
- **Scale.** non-robust mean + std; known limitation of original criterion.
- **Comparison.** table on [HampelScorer](hampel_scorer.md).
