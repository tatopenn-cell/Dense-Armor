# OnlineGaussianNB: a classifier that keeps a mean and a variance

`OnlineGaussianNB` is the simplest of the three online classifiers. For each class and each column it keeps a running mean and variance, updated one sample at a time, and it answers with the class that makes the sample most likely. It never stores the samples. With a forgetting factor (`alpha`) it forgets the old ones exponentially, so it can follow a world that changes. The other two classifiers are [OnlineSoftmaxRegression](softmax.md) and [DriftAdaptiveClassifier](drift_adaptive.md); the overview is on [Online classifiers](online_classifiers.md).

## Gamestory

London, deep night. The young detective is in front of the bank ledger and has to check thousands of payments one after another. So as not to waste time flipping through endless archives, he pulls OnlineGaussianNB out of his pocket, a tiny notebook of averages.

- **1. Writing down only two little numbers on the fly:** Every time a payment goes through, the detective doesn't jot down the whole endless list of the evening (otherwise the notebook would burst!). He updates only two numbers with his pen: how much honest people and thieves spend on average, and how much those numbers bounce around the average.
- **2. Betting on the most likely:** When a new payment arrives, the detective takes a lightning glance at those two little numbers in his notebook and places his bet: "Does this withdrawal look more like the habits of honest citizens or those of the thief?".
- **3. The trick to forget (alpha):** If halfway through the night the thief changes habits and switches from big withdrawals to small ones, a normal notebook stays stuck in the past. But if the detective uses the alpha rule, he lets the old notes slowly fade away and gives more weight to the latest payments, getting right back on the right track!

In the end, what is OnlineGaussianNB? It is a super-light and super-fast algorithm that learns one piece of data at a time by calculating constantly updated "averages" without having to save all past data, and thanks to the alpha trick it knows how to forget old rules when things change!

![Left: the average amount of a fraud as the classifier sees it over 6000 payments, with and without forgetting: at payment 3000 the true value falls from 400 to 60; with alpha 0.1 the belief follows within a few dozen payments, with alpha 0 it slowly drifts toward the average of the two regimes. Right: the probability of fraud given the amount at payment 6000: with alpha 0.1 small payments are frauds, with alpha 0 it is flat at 0.49.](../assets/learn/nb_learning.png)

Left: what the classifier believes the average amount of a fraud is. At payment 3000 the fraudsters change habit and a fraud becomes a small payment (60 instead of 400). With `alpha = 0.1` the belief follows within a few dozen payments; with `alpha = 0` it is the average of everything it has ever seen and drifts slowly toward 230 (226 at payment 6000). Right: the probability of fraud given the amount, at payment 6000. With `alpha = 0.1` it is a sharp border at about 150 (small payments are the frauds now); with `alpha = 0` it is a flat 0.49 whatever the amount: the classifier no longer knows.

## A first example: payments that change meaning

Half of the payments are fraud. At first a fraud is a large payment (around 400) and a normal one is small (around 60); after 3000 payments they swap. Two classifiers see the same payments, one that never forgets (`alpha=0`) and one that forgets a little all the time (`alpha=0.1`). At payment 3000 and at payment 6000 each is asked for the probability of fraud of a payment of 400 and of one of 60.

```python
import numpy as np
from dense_armor.core.online.learn.online_classifiers import OnlineGaussianNB

rng = np.random.default_rng(1)
models = {
    "alpha=0": OnlineGaussianNB(),
    "alpha=0.1": OnlineGaussianNB(alpha=0.1),
}
for i in range(6000):
    fraud = int(rng.random() < 0.5)
    big = fraud if i < 3000 else 1 - fraud
    x = {
        "amount": rng.normal(400 if big else 60, 60 if big else 20),
        "hour": rng.normal(14, 3),
    }
    for m in models.values():
        m.learn_one(x, fraud)
    if i + 1 in (3000, 6000):
        for name, m in models.items():
            p = [
                m.predict_proba_one({"amount": a, "hour": 14.0}).get(1, 0.0)
                for a in (400.0, 60.0)
            ]
            print(i + 1, name, [round(v, 2) for v in p])
```

```
3000 alpha=0 [1.0, 0.0]
3000 alpha=0.1 [1.0, 0.0]
6000 alpha=0 [0.49, 0.49]
6000 alpha=0.1 [0.0, 1.0]
```

At payment 3000 both are right: 400 is a fraud (1.0) and 60 is not (0.0). At payment 6000 the classifier with `alpha=0.1` has followed the change (a payment of 400 is honest, 0.0, and one of 60 is a fraud, 1.0). The one with `alpha=0` is undecided: 0.49 for both, because its statistics are the mixture of the two regimes.

## How it learns

For each class and each feature, the classifier keeps a running mean and variance,
updated by Welford's algorithm:

```
n ← n + 1
δ = v − m
m ← m + δ / n
M2 ← M2 + δ (v − m)
variance = M2 / n
```

**Hand case.** Three values `2, 4, 6`:

```
v = 2: n = 1, m = 2, M2 = 0
v = 4: n = 2, m = 3, M2 = 2
v = 6: n = 3, m = 4, M2 = 8
```

Mean 4, variance `8 / 3 = 2.667`.

The classifier picks the class with the highest

```
log prior + Σ log N(x_i; m_i, var_i)
```

With `alpha = 0`, the statistics equal the batch MLE of a Gaussian NB on the observed
samples; the wrapper matches a batch Gaussian NB to numerical precision (max difference
1.8e-15 on 500 samples). With `alpha > 0`, the statistics **forget** exponentially:
old counts are multiplied by `(1 − α)` before the new sample is added, so the classifier
adapts when the class-conditional distributions change.

## The parameters

| Parameter | Meaning | Default |
|---|---|---|
| `alpha` | forgetting factor: old counts are multiplied by `(1 − alpha)` before each sample; `0` means no forgetting | 0.0 |
| `var_smoothing` | a small number added to the variances, to keep a feature with no spread from dominating | 0.0 |
| `eps` | the smallest variance used in the likelihood | 1e-12 |
| `dtype` | `None` for the default precision, `numpy.float32` for the float32 mode below | `None` |

Cost, on one CPU core: about 10.5 µs per sample.

## The float32 mode

`OnlineGaussianNB(dtype=numpy.float32)` keeps per-class and per-feature
state in float32 and applies the shifted-data algorithm with
``K = x[0]`` (Healy 2022, Algorithm 2) online: two TwoSum sums per
feature and the two-term variance. Relative error of the variance
against the float64 reference on N(1e3, 0.01), 100,000 samples, seed 0:
6.2e-8. On a constant stream at 1e10 and 1e38 the variance is exactly
0.0. With this mode ``alpha`` must be 0 (the exponential decay is not
compatible with a fixed shift).

Cost, median of 7 runs of one update over 20,000 samples, one CPU thread: 0.0108 ms for the default mode (IQR 0.0006) and 0.0201 ms for the float32 mode (IQR 0.0080). The float32 mode is about 1.9 times slower here: on a scalar loop the numpy float32 arithmetic is slower than the Python float64 arithmetic, so the goal that it be at most as slow as the default is not met. The speed of the float32 mode on a batch is not measured: these classes have no `learn_many`.

## API reference

::: dense_armor.core.online.learn.online_classifiers.OnlineGaussianNB
    options:
      docstring_style: numpy

---

## Details

- `learn_one(x, y)` takes a dict and a label; `predict_proba_one(x)` returns a dict with the probability of every class seen so far, and `predict_one(x)` the most probable one. A class appears the first time its label arrives.
- The robot version of the same experiment is at the end of [Online classifiers](online_classifiers.md).
