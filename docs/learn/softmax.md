# OnlineSoftmaxRegression: a border that moves

`OnlineSoftmaxRegression` is a multinomial logistic regression trained one sample at a time. It does not keep a mean and a variance per class like [OnlineGaussianNB](gaussian_nb.md): it keeps a weight for each column and each class, turns them into probabilities with a softmax, and moves the weights a little every time it is wrong. It is often better than the naive Bayes when the border between the classes is not made of Gaussians, and it has no forgetting factor to choose: its step shrinks by itself on the columns that have already moved a lot (AdaGrad). The other two classifiers are [OnlineGaussianNB](gaussian_nb.md) and [DriftAdaptiveClassifier](drift_adaptive.md); the overview is on [Online classifiers](online_classifiers.md).

## Gamestory

London, deep night. The thief has gotten smart: he no longer makes just big or small withdrawals, but mixes times and amounts in complicated ways. The old notebook of averages is no longer enough, so the detective launches OnlineSoftmaxRegression, a program that draws moving borders on a map.

- **1. Drawing a border line:** The program puts all payments on a map and draws an invisible line: on one side are the Honest, on the other the Scammers. The further a payment is from the line, the more certain the program is of its answer.
- **2. Moving the line on every mistake:** Every time the program misjudges a payment, it immediately shifts the border line so as not to repeat the same mistake.
- **3. Braking on clues already seen (AdaGrad):** The program has a smart brake called AdaGrad. If a clue (like the time of day) has already caused many adjustments, the program starts moving it with smaller, more cautious steps, while staying ready to take big steps on new clues! Also, if the thief invents a third type of scam, the program immediately adds a new territory on the map on the fly.

In the end, what is OnlineSoftmaxRegression? It is a program that learns to separate situations by drawing and shifting "borders" one clue at a time, adjusting on its own the speed at which it changes its mind!

![The probability of fraud given the amount of a payment, at five moments of 6000 payments: after 1000 and 3000 payments large payments are frauds; after the change at payment 3000 the curve turns, at 3200 it is still half way, at 3600 and 6000 small payments are the frauds.](../assets/learn/softmax_boundary.png)

Each curve is the probability of fraud that the classifier gives to a payment of a given amount, at one moment of the stream. The first two are the old rule: the larger the amount, the more likely a fraud, with the border (the amount where the probability is 0.5) at about 185 after 1000 payments and about 260 after 3000. At payment 3000 the fraudsters change habit and the classifier moves the border, one mistake at a time: at 3200 the curve has turned and flattened (it is unsure), at 3600 it has almost finished turning, and at 6000 it is the opposite rule, with the border at about 180.

## A first example: payments that change meaning

Half of the payments are fraud. At first a fraud is a large payment (around 400) and a normal one is small (around 60); after 3000 payments they swap. Each payment is first predicted and then learned, and the accuracy is counted before the change, over the 200 payments after it, and late in the stream.

```python
import numpy as np
from dense_armor.core.online.learn.online_classifiers import (
    OnlineSoftmaxRegression,
)

rng = np.random.default_rng(1)
sm = OnlineSoftmaxRegression()
hits = []
for i in range(6000):
    fraud = int(rng.random() < 0.5)
    big = fraud if i < 3000 else 1 - fraud
    x = {
        "amount": rng.normal(400 if big else 60, 60 if big else 20),
        "hour": rng.normal(14, 3),
    }
    hits.append(sm.predict_one(x) == fraud)
    sm.learn_one(x, fraud)
h = np.array(hits, float)
print(
    h[500:3000].mean().round(3),
    h[3000:3200].mean().round(3),
    h[5000:5900].mean().round(3),
)
```

```
0.996 0.765 1.0
```

Before the change it is right 99.6 % of the time. In the 200 payments after the change it is right 76.5 % of the time: it has already moved most of the way. Late in the stream it is right every time (1.0). There is no forgetting factor to choose.

## How it learns

Probabilities are `softmax(W z)` where `z = [features..., 1]`. Probabilities are `softmax(W z)` where `z = [features..., 1]`. The gradient for class
`c` on sample `(z, y)` is

```
(p_c − [c = y]) · z
```

with `p_c` the softmax output for class `c` and `[c = y]` equal to 1 when `c` is the
true label and 0 otherwise. The AdaGrad step is `η / √G` where `G` is the running sum of
squared gradients, updated as `G ← G + g²` before the parameter step. `η` is the base
learning rate (default 0.1); `l2` adds a small penalty.

Classes are added on the fly when a new label arrives.

## The parameters

| Parameter | Meaning | Default |
|---|---|---|
| `eta` | the base learning rate | 0.1 |
| `l2` | a small penalty on the weights; `0` means none | 0.0 |
| `adaptive` | whether the step is shrunk by AdaGrad (`η / √G`) | `True` |
| `eps` | a small number that keeps the AdaGrad step finite | 1e-8 |

Cost, on one CPU core: about 36.8 µs per sample.

## API reference

::: dense_armor.core.online.learn.online_classifiers.OnlineSoftmaxRegression
    options:
      docstring_style: numpy

---

## Details

- `learn_one(x, y)` takes a dict and a label; `predict_proba_one(x)` returns a dict with the probability of every class seen so far.
- The robot version of the same experiment is at the end of [Online classifiers](online_classifiers.md).
