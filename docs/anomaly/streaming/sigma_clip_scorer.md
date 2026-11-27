# SigmaClipScorer

[← Streaming anomaly detection](index.md)

`SigmaClipScorer` first removes points beyond `n_sigmas`, recomputes the centre and spread of what remains, repeats until stable or `max_iters`, and then scores the new value by how far it is from that cleaned centre in cleaned-standard-deviation units. The window is `2 * radius` samples strictly before the value. `is_outlier` is `True` when the score is above `n_sigmas`. It is an `AnomalyDetector`, causal.

It looks only at the `2 * radius` samples before the value being scored, like the other scorers; why the batch versions differ is explained on the page of [HampelScorer](hampel_scorer.md).

![The score of a cost of 18 for 0 to 14 bad ticks in a window of 40 days and 1 to 8 passes of clipping.](../../assets/streaming/sigma_clip_scorer_3d.png)

With four bad ticks one pass is not enough (score 0.52) and two passes clean the window (score 4.32). With six or more bad ticks the passes do not help (score 0.02 with six): the bad ticks stretch the standard deviation so much that none of them is beyond 3 deviations.

## A first example: a daily cost of executing an order

The window of 40 days holds 5 bad ticks, recorded between 45 and 55. A cost of 18 arrives, about 4 standard deviations above the usual 12.

```python
import numpy as np
from dense_armor.core.online.anomaly.filters import SigmaClipScorer

rng = np.random.default_rng(4)
cost = 12.0 + 1.5 * rng.standard_normal(40)
cost[[3, 9, 15, 22, 31]] = [45.0, 50.0, 48.0, 55.0, 47.0]
for passes in (1, 5):
    s = SigmaClipScorer(radius=20, n_sigmas=3.0, max_iters=passes)
    for v in cost:
        s.learn_one({"x": float(v)})
    print(passes, round(s.score_one({"x": 18.0}), 2), s.is_outlier({"x": 18.0}))
```

```
1 0.24 False
5 4.13 True
```

With one pass the bad ticks are not clipped enough and 18 scores 0.24: it is not an outlier. With five passes the window is cleaned and 18 scores 4.13, above the threshold of 3. The spread of the window, measured at the start of each pass, is 12.5, 10.99 and 9.57, and 1.54 once the three passes that remove days are done.

![The 40 days of the window; the bad ticks are coloured by the pass that removes them, and today's cost is a diamond.](../../assets/streaming/sigma_clip_scorer_finance.png)

Cyan days stay. Orange days leave in pass 1, pink days in pass 2 and white days in pass 3.

## The same on a robot

```python
from dense_armor.core.online.anomaly.filters import SigmaClipScorer

s = SigmaClipScorer(radius=5, n_sigmas=3.0)
for v in [1.0, 1.2, 0.9, 1.1, 30.0, 0.8, 1.05, 1.0, 0.95, 1.1]:
    s.learn_one({"v": v})
print(round(s.score_one({"v": 1.3}), 2), s.is_outlier({"v": 1.3}), s.is_outlier({"v": 50.0}))
```

```
0.3 False True
```

The bad reading of 30 in the window is clipped, so a reading of 1.3 is ordinary (score 0.3) and a reading of 50 is an outlier.

## How it works

`SigmaClipScorer` repeatedly removes values beyond `n_sigmas` standard deviations and scores against what remains.

The score is how far the new value is from a cleaned centre, measured in cleaned-standard-deviation units. First it removes obvious outliers from the recent window, then it measures. `learn_one` adds a value; `score_one` does not change anything.

It takes the `2 * radius` samples strictly before the value. The mean and standard deviation are recomputed after removing points beyond `n_sigmas`, repeated until stable or `max_iters`, leaving `mu_clean` and `sigma_clean`. The score is

$$\mathrm{score} = \frac{|x - \mu_{\mathrm{clean}}|}{\sigma_{\mathrm{clean}}},$$

where $x$ is the new value; $\mu_{\mathrm{clean}}$ is the mean of the cleaned window; $\sigma_{\mathrm{clean}}$ is its standard deviation. `n_sigmas` is both the clipping threshold and the outlier threshold. `is_outlier` is `score_one(x)` above `n_sigmas`. `score_one` returns `0.0` for an unusable value or while fewer than four values have been learned. On a window whose cleaned scale is degenerate, the score is `positive infinity` if the value deviates from `mu_clean`, `0.0` if it does not. `score_one` does not change anything.

`SigmaClipScorer` is `robust_filters.sigma_clip` applied to a causal window. See the comparison on [HampelScorer](hampel_scorer.md).

## The parameters

`SigmaClipScorer` is an `AnomalyDetector`. `learn_one(x)`, `score_one(x)`, `is_outlier(x)`, `n_missing`. Signature:

`(self, radius: int = 10, n_sigmas: float = 3.0, max_iters: int = 5, feature: str | None = None, eps: float = 1e-09)`

| Parameter | Default | What it does |
|---|---|---|
| `radius` | 10 | with a `2 * radius` causal window |
| `n_sigmas` | 3.0 | clipping threshold and outlier threshold |
| `max_iters` | 5 | maximum number of clipping iterations |
| `feature` | `None` | key of the feature to watch; `None` takes the feature with the smallest key, order-independent |
| `eps` | 1e-9 | below this scale the cleaned window is treated as flat |

## API reference

::: dense_armor.core.online.anomaly.filters.SigmaClipScorer

## Details

- **Sources.** Standard in astronomy; see Akritas, M. G., Bershady, M. A. (1996). Linear regression for astronomical data with measurement errors and intrinsic scatter. *ApJ* 470, 706. Batch `robust_filters.sigma_clip`.
- **Causal window.** `2 * radius` strictly before; not part of own window.
- **Comparison.** table on [HampelScorer](hampel_scorer.md).
