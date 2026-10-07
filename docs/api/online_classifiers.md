# Online classifiers (robot states with drift adaptation)

A robot has to tell, sample by sample, which state it is in (free motion, contact, collision)
from its joint signals, while the robot and its environment change. `OnlineGaussianNB` and
`OnlineSoftmaxRegression` learn one sample at a time; `DriftAdaptiveClassifier` wraps either of
them with a drift detector on its own prediction loss and swaps in a fresh copy when the old one
goes stale.

```bash
pip install dense-armor[river]
```

```python
from dense_armor.utility.online_classifiers import OnlineGaussianNB, DriftAdaptiveClassifier
from dense_armor.utility.river_drift import CUSUMDriftDetector

det = CUSUMDriftDetector(reference="fixed", radius=20, ref_mult=5, two_sided=False)
clf = DriftAdaptiveClassifier(OnlineGaussianNB(), det, window=50, smooth_window=50)
for x, y in [({"j0": 0.1, "j1": 2.1}, "contact"), ({"j0": 0.0, "j1": 0.2}, "free")]:
    pred = clf.predict_one(x)
    clf.learn_one(x, y)
```

Each sample is predicted first and learned after (prequential evaluation). The wrapper feeds the
detector a 50-sample rolling mean of the log-loss `-log p(y)`. When the detector fires it starts
a background copy trained only on the new samples, and promotes it once its accuracy over the
last 50 samples beats the current classifier's. All three classifiers return probabilities, so
`OnlinePlattScaling` (see [calibration](calibration.md)) can wrap them in the binary case.

## A contact signature that changes

6000 samples, 3 states from a Markov chain (stay probability 0.95, mean dwell 20 samples),
4 joint features with unit noise: free `(0, 0, 0, 0)`, contact `(2, 2, 0, 0)`,
collision `(4, 4, 4, 4)`. From sample 3000 the contact load moves to the other two joints,
`(0, 0, 2, 2)`. Prequential accuracy over `[500, 3000)` (pre), `[3000, 3200)` (post) and
`[5000, 5900)` (late); recovery is the number of samples after 3000 until the 200-sample accuracy
is back within 0.05 of pre.

| Classifier | pre | post | late | recovery |
|---|---|---|---|---|
| `OnlineGaussianNB(alpha=0)` | 0.952 | 0.480 | 0.879 | 617 |
| `OnlineGaussianNB(alpha=0.01)` | 0.952 | 0.790 | 0.941 | 59 |
| `OnlineGaussianNB(alpha=0.1)` | 0.950 | 0.925 | 0.913 | 0 |
| `OnlineSoftmaxRegression(eta=0.1, l2=1e-4)` | 0.934 | 0.565 | 0.919 | 213 |
| `DriftAdaptiveClassifier(OnlineGaussianNB(alpha=0))` | 0.952 | 0.480 | 0.931 | 145 |
| `DriftAdaptiveClassifier(OnlineSoftmaxRegression)` | 0.912 | 0.775 | 0.904 | 38 |

Without forgetting, Gaussian NB keeps the old contact mean and needs 617 samples to recover. The
wrapper leaves its pre-drift accuracy untouched (0.952), fires 55 samples after the change and
recovers in 145. A fixed forgetting factor reacts faster when the drift rate is known in advance:
`alpha = 0.1` never loses accuracy here but pays for it with a noisier late phase (0.913).

With `alpha = 0`, `OnlineGaussianNB` gives the same probabilities as a batch Gaussian naive Bayes
fitted on the same samples (maximum difference 1.8e-15 on 500 samples). Per sample, on one CPU
core: 10.5 µs (`OnlineGaussianNB`), 36.8 µs (`OnlineSoftmaxRegression`), 75.8 µs (wrapped NB).

::: dense_armor.utility.online_classifiers

---

**Details**: the raw log-loss is close to zero on most samples, so its robust scale (MAD) is tiny
and a CUSUM fed with it fires on almost every error (hundreds of alarms in 6000 samples). The
50-sample rolling mean gives it a stable scale, and the one-sided test ignores loss drops. The
Gaussian NB variance uses the exponentially weighted Welford update (Welford 1962) and
`OnlineSoftmaxRegression` uses AdaGrad steps (Duchi, Hazan and Singer, JMLR 2011); the
background-learner swap follows Gama et al., *Learning with drift detection* (SBIA 2004).
