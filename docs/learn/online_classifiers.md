# Online classifiers (robot states)

A robot has to tell, sample by sample, which state it is in: free motion, contact,
collision. It has to do this **from its own joint signals**, and it has to keep doing it
right when the robot's world changes — a different payload, a worn gripper, a slightly
different surface.

`OnlineGaussianNB` and `OnlineSoftmaxRegression` learn one sample at a time.
`DriftAdaptiveClassifier` wraps either of them with a drift detector on its own
prediction loss and swaps in a fresh copy when the old one goes stale.

## The data

6,000 samples, 3 states from a Markov chain (stay probability 0.95, mean dwell 20
samples), 4 joint features with unit noise:

- **free**: `(0, 0, 0, 0)`
- **contact**: `(2, 2, 0, 0)`
- **collision**: `(4, 4, 4, 4)`

From sample 3000 the contact load **moves to the other two joints**: contact becomes
`(0, 0, 2, 2)`. The classifier was trained on the old signature.

```python
import numpy as np
rng = np.random.default_rng(42)

state_means = {
    "free":      np.array([0, 0, 0, 0], dtype=float),
    "contact":   np.array([2, 2, 0, 0], dtype=float),
    "collision": np.array([4, 4, 4, 4], dtype=float),
}
states = ["free", "contact", "collision"]
p_stay = 0.95
n = 6000
y = np.empty(n, dtype=object)
s = "free"
for i in range(n):
    if rng.random() > p_stay:
        s = rng.choice(states)
    y[i] = s
    if i >= 3000 and s == "contact":
        y[i] = "contact"
X = np.zeros((n, 4))
for i in range(n):
    if y[i] == "contact" and i >= 3000:
        mean = np.array([0.0, 0.0, 2.0, 2.0])
    else:
        mean = state_means[y[i]]
    X[i] = mean + rng.standard_normal(4)
```

## 1. Gaussian naive Bayes, one sample at a time

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

```python
from dense_armor.learn.online_classifiers import OnlineGaussianNB

clf = OnlineGaussianNB(alpha=0.0)
for i in range(n):
    features = {f"j{k}": float(X[i, k]) for k in range(4)}
    clf.learn_one(features, y[i])
```

## 2. What happens when the signature changes

The contact state moves from joints 0–1 to joints 2–3 at sample 3000. A classifier
without forgetting keeps the old contact mean and gets every post-change contact sample
wrong. Prequential accuracy over `[500, 3000)` (pre), `[3000, 3200)` (post), and
`[5000, 5900)` (late):

| Classifier | pre | post | late | recovery |
|---|---|---|---|---|
| `OnlineGaussianNB(alpha=0)` | 0.952 | 0.480 | 0.879 | 617 |
| `OnlineGaussianNB(alpha=0.01)` | 0.952 | 0.790 | 0.941 | 59 |
| `OnlineGaussianNB(alpha=0.1)` | 0.950 | 0.925 | 0.913 | 0 |
| `OnlineSoftmaxRegression(eta=0.1, l2=1e-4)` | 0.934 | 0.565 | 0.919 | 213 |
| `DriftAdaptiveClassifier(OnlineGaussianNB(alpha=0))` | 0.952 | 0.480 | 0.931 | 145 |
| `DriftAdaptiveClassifier(OnlineSoftmaxRegression)` | 0.912 | 0.775 | 0.904 | 38 |

"Recovery" is the number of samples after 3000 until the 200-sample accuracy is back
within 0.05 of pre.

Without forgetting, Gaussian NB goes from 0.952 to 0.480 at the change and needs **617
samples** to recover. A fixed forgetting factor reacts faster if you already know how
fast the world changes — `alpha = 0.1` never loses accuracy, at the cost of a noisier
late phase (0.913 instead of 0.941).

The drift-adaptive wrapper is the third option: it does not need a forgetting factor
chosen in advance.

## 3. The drift-adaptive wrapper

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

```python
from dense_armor.learn.online_classifiers import OnlineGaussianNB, DriftAdaptiveClassifier
from dense_armor.drift.detector import CUSUMDriftDetector

det = CUSUMDriftDetector(reference="fixed", radius=20, ref_mult=5, two_sided=False)
clf = DriftAdaptiveClassifier(OnlineGaussianNB(), det, window=50, smooth_window=50)
for i in range(n):
    features = {f"j{k}": float(X[i, k]) for k in range(4)}
    pred = clf.predict_one(features)
    clf.learn_one(features, y[i])
```

On the running data the wrapper leaves its pre-drift accuracy untouched (0.952), fires
about 55 samples after the change, and recovers in 145 samples. It reaches 0.931 in the
late phase — better than the no-forgetting baseline (0.879) and close to the tuned
`alpha = 0.1` (0.913).

## 4. The softmax regression alternative

`OnlineSoftmaxRegression` is a multinomial logistic regression with AdaGrad. It uses
the same interface and is often better when the class boundaries are not Gaussian.

### The formula, symbol by symbol

Probabilities are `softmax(W z)` where `z = [features..., 1]`. The gradient for class
`c` on sample `(z, y)` is

```
(p_c − [c = y]) · z
```

with `p_c` the softmax output for class `c` and `[c = y]` equal to 1 when `c` is the
true label and 0 otherwise. The AdaGrad step is `η / √G` where `G` is the running sum of
squared gradients, updated as `G ← G + g²` before the parameter step. `η` is the base
learning rate (default 0.1); `l2` adds a small penalty.

Classes are added on the fly when a new label arrives.

## 5. Cost

Per sample, on one CPU core:

| classifier | µs per sample |
|---|---|
| `OnlineGaussianNB` | 10.5 |
| `OnlineSoftmaxRegression` | 36.8 |
| `DriftAdaptiveClassifier(OnlineGaussianNB)` | 75.8 |

At 100 Hz the loop has 10 ms per cycle: all three fit with three orders of magnitude
to spare. The wrapper is 7× the bare NB because it runs a background copy in parallel
after each drift alarm.

## API reference

::: dense_armor.learn.online_classifiers

---

## Details

The Gaussian NB variance uses the exponentially weighted Welford update
(Welford 1962); `OnlineSoftmaxRegression` uses AdaGrad steps (Duchi, Hazan and Singer,
JMLR 2011); the background-learner swap follows Gama et al., *Learning with drift
detection* (SBIA 2004).

All three classifiers return probabilities, so [calibration](calibration.md)
(`OnlinePlattScaling`) can wrap them in the binary case. The two together — adaptive
learning and honest probabilities — are what a control policy needs to be able to
trust the output.
