# SigmaClipScorer

[← Streaming anomaly detection](index.md)

SigmaClipScorer first removes samples exceeding n_sigmas standard deviations, recalculates the center and dispersion on the remaining data, and repeats the process until convergence or until max_iters iterations are reached; finally, it evaluates the new sample by measuring its distance from the cleaned center, expressed in units of cleaned standard deviation. The causal window comprises the 2 * radius samples strictly preceding the current value. The is_outlier method returns True when the score exceeds n_sigmas. It is a purely causal AnomalyDetector.

Like the other scorers in the library, it examines exclusively the 2 * radius samples preceding the value to be evaluated; the reasons for the differences compared to the batch counterparts are illustrated on the [HampelScorer](hampel_scorer.md) page.

![Trend of the score assigned to a cost equal to 18 as the number of anomalous values in the window varies (from 0 to 14 out of 40 days) and as the number of clipping passes varies (from 1 to 8).](../../assets/streaming/sigma_clip_scorer_3d.png)

In the presence of four anomalous values in the window, a single pass is not sufficient (score 0.52), whereas two passes completely clean the window (score 4.32). With six or more anomalous values, on the other hand, increasing the passes is not enough (score 0.02 with six outliers): the anomalous elements stretch the initial standard deviation to such an extent that none of them exceeds 3 standard deviations on the first pass anymore.

## A first example: daily order execution cost

A 40-day window contains 5 anomalous spikes, recorded with values between 45 and 55. This is followed by a daily cost equal to 18, roughly 4 standard deviations above the habitual level of 12.

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

With a single pass (max_iters=1), the spurious values are not sufficiently removed and 18 obtains a score of 0.24, turning out normal (False). With five passes (max_iters=5), the window is effectively cleaned and 18 reaches a score of 4.13, well above the threshold of 3 (True). The dispersion of the window, measured at the beginning of each iteration, drops progressively from 12.5 to 10.99, then to 9.57, until stabilizing at 1.54 once the three passes required to remove all anomalous days are completed.

![The 40 days of the window; anomalous spikes are colored according to the pass that eliminates them, while today's cost is indicated by a diamond.](../../assets/streaming/sigma_clip_scorer_finance.png)

Normal days kept are in cyan; days eliminated on the first pass are in orange, those on the second pass in pink, and those on the third pass in white.

## The same applies to a robot

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

The anomalous value equal to 30.0 present in the window is discarded during clipping: consequently, a reading of 1.3 turns out normal (score 0.3, False), while 50.0 is flagged as an outlier (True).

## How it works

SigmaClipScorer iteratively removes samples exceeding n_sigmas standard deviations and calculates the score against the surviving data only.

The score expresses how much the new sample deviates from the cleaned center of the window, in units of cleaned standard deviation: the algorithm first purges the recent window of obvious outliers and then performs the measurement. The learn_one method adds a data point to the window; score_one calculates the score without altering the internal state.

The algorithm operates on the 2 * radius samples strictly preceding the value to be evaluated. At each iteration, mean and standard deviation are recalculated by discarding points beyond n_sigmas, until the set stabilizes or max_iters is reached, thus obtaining $\mu_{\mathrm{clean}}$ and $\sigma_{\mathrm{clean}}$. The score z is defined as:

$$z = \frac{|x - \mu_{\mathrm{clean}}|}{\sigma_{\mathrm{clean}}}$$

Where:

- x is the new value to be evaluated;
- μclean is the mean of the cleaned window;
- σclean is the standard deviation of the cleaned window.

The n_sigmas parameter acts both as an iterative clipping threshold and as a final classification threshold (is_outlier returns True when score_one(x) > n_sigmas). score_one returns 0.0 for invalid values (NaN, inf) or until at least four samples have been learned. If the cleaned window exhibits a degenerate scale (sigma_clean < eps), the score is +inf (float('inf')) whenever the value differs from μclean, otherwise 0.0. score_one never modifies the internal state.

SigmaClipScorer corresponds to applying robust_filters.sigma_clip on a causal window. For the comparative table among the scorers, refer to the [HampelScorer](hampel_scorer.md) page.

## The parameters

SigmaClipScorer is an AnomalyDetector equipped with the learn_one(x), score_one(x), is_outlier(x) methods and the n_missing attribute.

Signature:

`SigmaClipScorer(radius: int = 10, n_sigmas: float = 3.0, max_iters: int = 5, feature: str | None = None, eps: float = 1e-09)`

| Parameter | Default | Description |
|---|---|---|
| radius | 10 | Half-width unit; defines a causal window of 2 * radius samples. |
| n_sigmas | 3.0 | Iterative clipping threshold and outlier detection threshold. |
| max_iters | 5 | Maximum number of clipping iterations. |
| feature | None | Key of the feature to monitor; if None, selects the lexicographically smallest key, regardless of order. |
| eps | 1e-9 | Scale threshold: below this value the cleaned window is considered flat. |

## API reference

::: dense_armor.core.online.anomaly.filters.SigmaClipScorer

## Details

- Sources: standard method in astronomy; Akritas, M. G., Bershady, M. A. (1996), Linear Regression for Astronomical Data with Measurement Errors and Intrinsic Scatter, ApJ 470, 706; batch counterpart in robust_filters.sigma_clip.
- Causal window: uses exclusively the 2 * radius strictly preceding samples; the current value never becomes part of its own window.
- Comparison: the comparative table with the other scorers is available on the [HampelScorer](hampel_scorer.md) page.
