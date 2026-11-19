# DriftAdaptiveClassifier: a substitute on the bench

`DriftAdaptiveClassifier` is not a classifier of its own: it wraps any classifier (here [OnlineGaussianNB](gaussian_nb.md)) with an alarm that watches its mistakes. When the alarm rings, the wrapper starts a fresh copy of the classifier that learns only the new samples, and when that copy proves better than the old one it takes over. So you do not have to know in advance how fast the world changes, which is what the forgetting factor of the naive Bayes asks. The other two classifiers are [OnlineGaussianNB](gaussian_nb.md) and [OnlineSoftmaxRegression](softmax.md); the overview is on [Online classifiers](online_classifiers.md).

## Gamestory

London, deep night. Halfway through the chase, the thief completely flips the rules of the game: everything that used to be a sign of a scam is now normal, and vice versa. To avoid being caught off guard, the detective activates DriftAdaptiveClassifier, a team made up of a Starter, an Alarm, and a Substitute.

- **1. The Starter and the Alarm:** On the field is the Starter program checking the payments. Next to it is an alarm detector that measures how many mistakes the Starter makes over the last 50 payments (not on a single one, otherwise it would ring for every tiny slip!).
- **2. The siren goes off!:** When the thief suddenly changes his trick, the Starter begins making mistakes left and right. The error average shoots up and the Alarm sets off the siren!
- **3. In comes the fresh Substitute:** As soon as the alarm rings, the detective sends a brand-new copy of the program (the Substitute) to warm up, studying only the new payments without having its head full of the old wrong rules. As soon as the Substitute proves over the last 50 cases that it is better than the Starter, it takes over command!

In the end, what is DriftAdaptiveClassifier? It is a protective shell that wraps around a normal program: it keeps an eye on its mistakes and, if it realizes the world has changed, trains a fresh substitute on the bench ready to take command!

![Accuracy over the last 200 payments of three classifiers: the one that never forgets, the one with alpha 0.1 and the drift-adaptive wrapper. After the change at payment 3000 the wrapper drops, its backup starts at payment 3013 and takes over at payment 3062, after which it is back at 1.0; alpha 0.1 dips for about 200 payments.](../assets/learn/drift_swap.png)

The figure follows the wrapper through the change at payment 3000. The alarm rings 13 payments after the change and starts the backup; until the backup takes over, at payment 3062, the wrapper answers with the old classifier, which is wrong; from there it is right again. The classifier with `alpha=0.1` takes about 200 payments to recover, because its forgetting rate is fixed; the one that never forgets does not recover within the 3000 payments after the change.

## A first example: payments that change meaning

Half of the payments are fraud. At first a fraud is a large payment (around 400) and a normal one is small (around 60); after 3000 payments they swap. Each payment is first predicted and then learned, and the accuracy is counted before the change, over the 200 payments after it, and late in the stream (payments 5000 to 5900).

```python
import numpy as np
from dense_armor.core.online.drift.detector import CUSUMDriftDetector
from dense_armor.core.online.learn.online_classifiers import (
    DriftAdaptiveClassifier,
    OnlineGaussianNB,
)

rng = np.random.default_rng(1)
detector = CUSUMDriftDetector(
    reference="fixed", radius=20, ref_mult=5, two_sided=False
)
clf = DriftAdaptiveClassifier(
    OnlineGaussianNB(), detector, window=50, smooth_window=50
)
hits = []
for i in range(6000):
    fraud = int(rng.random() < 0.5)
    big = fraud if i < 3000 else 1 - fraud
    x = {
        "amount": rng.normal(400 if big else 60, 60 if big else 20),
        "hour": rng.normal(14, 3),
    }
    hits.append(clf.predict_one(x) == fraud)
    clf.learn_one(x, fraud)
h = np.array(hits, float)
print(
    h[500:3000].mean().round(3),
    h[3000:3200].mean().round(3),
    h[5000:5900].mean().round(3),
)
```

```
1.0 0.685 1.0
```

The wrapper is perfect before the change, right on 68.5 % of the 200 payments after it (the old classifier is in command for the first 62, and the backup after that) and perfect again late in the stream.

## How it works

`DriftAdaptiveClassifier` wraps any classifier with a drift detector on the classifier's
own prediction loss. Per sample:

1. Predict, then compute the log-loss `−log p(y)`.
2. Feed the **50-sample rolling mean** of the log-loss to the detector.
3. Learn.
4. If the detector fires, start a **background copy** trained only on new samples.
5. When the background's accuracy over the last 50 samples beats the main classifier's
   over the same window, **promote** the background to main.

The 50-sample rolling mean is not optional in practice: the raw log-loss is near zero
on most samples, so its robust scale (MAD) is tiny and a CUSUM fed with the raw value
fires on almost every error — hundreds of alarms in 6,000 samples. The rolling mean
gives the detector a stable scale.

### Repeated alarms

After the change the detector rings more than once: in this example it rang 34 times between payment 3014 and payment 3099. A ring while a backup is already running does not restart it: the backup keeps the payments it has already seen, and it takes over as soon as it has `window` (50) of them and beats the main classifier, here at payment 3062. A first version of the wrapper restarted the backup at every ring; in this example it rang 170 times between payment 3013 and payment 4279, the backup never reached 50 payments without being replaced, and the takeover came only at payment 4276, 1276 payments after the change. That was a defect, and it is fixed.

## The parameters

| Parameter | Meaning | Default |
|---|---|---|
| `classifier` | the classifier to wrap; a fresh copy is made for the main and for every backup | (required) |
| `detector` | a drift detector with `update(x)` and `drift_detected`, for example `CUSUMDriftDetector` | (required) |
| `window` | the number of recent samples over which the backup and the main classifier are compared | 50 |
| `smooth_window` | the number of recent log-losses averaged before they reach the detector | 50 |

Cost, on one CPU core, for a prediction and a learning step of the 6000 payments of the example (best of 3 runs): about 178 µs per payment for the wrapper and 60 µs for the bare naive Bayes, 3 times, because it runs a backup in parallel after an alarm.

## API reference

::: dense_armor.core.online.learn.online_classifiers.DriftAdaptiveClassifier
    options:
      docstring_style: numpy

---

## Details

- `learn_one(x, y)` takes a dict and a label; every other method is delegated to the main classifier.
- The robot version of the same experiment is at the end of [Online classifiers](online_classifiers.md).
