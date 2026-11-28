# TukeyScorer

[← Streaming anomaly detection](index.md)

TukeyScorer evaluates each new sample by measuring how far it departs from Tukey's fences, expressed in interquartile-range (IQR) units. The fences are defined by the interval [Q1 - k * IQR, Q3 + k * IQR], computed on the quartiles of a causal window made of the 2 * radius samples strictly preceding the current data point; the multiplier k is 1.5 for ordinary outliers and 3.0 for extreme ones. Inside the fences the score is 0.0, while is_outlier returns True as soon as the score exceeds 0.0. It is a purely causal AnomalyDetector, with no look-ahead.

Like the other scorers of the library, it examines only the 2 * radius samples preceding the value to be evaluated; the reasons for the differences from the batch counterparts are explained on the [HampelScorer](hampel_scorer.md) page.

![Trend of the score evaluated on 9 values between -6 and 6 standard deviations and 8 different values of k from 0.5 to 3.](../../assets/streaming/tukey_scorer_3d.png)

The score is zero inside Tukey's fences and grows linearly outside them. With k = 1.5, a value at 3 standard deviations gets a score of 0.71 and one at 5 standard deviations reaches 2.37; with k = 3, the fences widen and a value at 5 standard deviations gets a score of 1.0.

## A first example: the size of an order

A trading desk monitors the size of its orders, expressed in number of shares. The distribution shows a strong positive skew (right tail): very many small orders and a few large ones. A new order of 200 shares arrives: is it anomalous?

```python
import numpy as np
from dense_armor.core.online.anomaly.filters import TukeyScorer

size = np.random.default_rng(2).lognormal(mean=4.0, sigma=0.6, size=60)
t = TukeyScorer(radius=30, k=1.5)
for v in size:
    t.learn_one({"x": float(v)})
q1, q3 = np.percentile(size, [25, 75])
print(round(q3 + 1.5 * (q3 - q1), 1), round(size.mean() + 3 * size.std(), 1), round(t.score_one({"x": 200.0}), 2))
```

```
167.5 193.4 0.63
```

Tukey's upper fence sits at 167.5 shares: the new order of 200 shares exceeds it and gets a score of 0.63 IQR units. The classic rule based on the mean plus 3 standard deviations sets the threshold at 193.4 instead: in a skewed stream, the largest orders stretch the standard deviation and move the limit upward, away from the bulk of the data. The quartiles, on the contrary, remain stable.

![Histogram of the 60 order sizes: Tukey's upper fence in orange, the mean plus 3 standard deviations in grey and the new order in pink.](../../assets/streaming/tukey_scorer_finance.png)

The orange line marks Tukey's upper threshold, while the grey one represents the mean plus 3 standard deviations. On a skewed distribution, the grey line sits markedly further to the right.

## The same applies to a robot

```python
from dense_armor.core.online.anomaly.filters import TukeyScorer

t = TukeyScorer(radius=5, k=1.5)
for v in [10, 11, 9, 10, 12, 10, 11, 9, 10]:
    t.learn_one({"v": float(v)})
print(t.is_outlier({"v": 13.0}), t.is_outlier({"v": 12.0}))
```

```
True False
```

For this reference window the normality interval is [8.5, 12.5]: the value 13.0 falls outside (True), while 12.0 is within the fences (False).

## How it works

The normality interval is defined as $[\mathrm{low}, \mathrm{high}]$, where $Q_1$ and $Q_3$ are the first and third quartiles of the window and $\mathrm{IQR} = Q_3 - Q_1$ (in the example of the window [10, 11, 9, 10, 12, 10, 11, 9, 10], the interval is [8.5, 12.5]).

The score measures how far the new sample lies outside this inner interval, expressed in interquartile-range units. First the 25th and 75th percentiles of the recent window are found. The learn_one method adds the value to the window; score_one computes the score without modifying the internal state.

Tukey's fences are given by:

$$\mathrm{low} = Q_1 - k \cdot \mathrm{IQR}, \qquad \mathrm{high} = Q_3 + k \cdot \mathrm{IQR}$$

Where $Q_1$ is the first quartile and $Q_3$ is the third quartile. The score is assigned according to three cases:

- if $x \in [\mathrm{low}, \mathrm{high}]$, the score is 0.0;
- if $x < \mathrm{low}$, the score is $\frac{\mathrm{low} - x}{\mathrm{IQR}}$;
- if $x > \mathrm{high}$, the score is $\frac{x - \mathrm{high}}{\mathrm{IQR}}$.

The parameter k is typically 1.5 to detect ordinary outliers and 3.0 for extreme outliers. The window contains exactly 2 * radius samples strictly preceding the current value (the evaluated sample is never part of its own window). score_one returns 0.0 for invalid values (NaN, inf) or until at least four samples have been learned. On a flat window (IQR < eps), the score is +inf (float('inf')) if the value differs from the fence, otherwise 0.0. The is_outlier method returns True when score_one(x) > 0.0.

TukeyScorer corresponds to applying robust_filters.tukey_fences on a causal window. For the comparison table among the scorers, see the [HampelScorer](hampel_scorer.md) page.

## The parameters

TukeyScorer is an AnomalyDetector equipped with the learn_one(x), score_one(x), is_outlier(x) methods and the n_missing attribute.

Signature:

`TukeyScorer(radius: int = 10, k: float = 1.5, feature: str | None = None, eps: float = 1e-09)`

| Parameter | Default | Description |
|---|---|---|
| radius | 10 | Half-width unit; defines a causal window of 2 * radius samples. |
| k | 1.5 | Multiplier of Tukey's fences (1.5 for ordinary outliers, 3.0 for extreme outliers). |
| feature | None | Key of the feature to monitor; if None, selects the lexicographically smallest key, regardless of order. |
| eps | 1e-9 | Scale threshold: below this IQR value the window is considered flat. |

TukeyScorer does not use the n_sigmas parameter: is_outlier is triggered directly when score_one(x) > 0.0.

## API reference

::: dense_armor.core.online.anomaly.filters.TukeyScorer

## Details

- Sources: Tukey, J. W. (1977), Exploratory Data Analysis, Addison-Wesley; batch counterpart in robust_filters.tukey_fences.
- Causal window: uses only the 2 * radius strictly preceding samples; is_outlier coincides exactly with the batch flagging condition (score > 0), with the only difference that the batch version of Tukey uses a centred window.
- Scale and missing values: based on quartiles and IQR; it keeps track of non-finite values through n_missing.
- Comparison: the table comparing it with the other scorers is on the [HampelScorer](hampel_scorer.md) page.
