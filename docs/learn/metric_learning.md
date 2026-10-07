# Metric learning (online)

A k-nearest-neighbours classifier is only as good as its distance. `metric_learning` learns
that distance online, one pair or triplet at a time, with three algorithms ported from their
papers: **OASIS** (Chechik et al. 2009, bilinear similarity from triplets), **LEGO** (Jain et
al. 2008, LogDet-regularized Mahalanobis matrix from pairs with a target distance) and
**POLA** (Shalev-Shwartz et al. 2004, pseudo-metric from similar/dissimilar pairs, kept
positive semi-definite).

A metric learner does not fit river's `Classifier`, `Regressor` or `Transformer` types: it
learns from two or three samples at a time and outputs a distance. `MetricLearner` is the
base class; `MetricKNNClassifier` is a river k-NN classifier that uses the learner as its
distance and, with `learn_metric=True` (the default), trains it online inside `learn_one`:
each new sample is paired with random samples from the window (a triplet for OASIS, a
similar/dissimilar pair for POLA, a pair with a target distance for LEGO).

```bash
pip install dense-armor[river]
```

```python
from dense_armor.learn.metric_learning import MetricKNNClassifier, OASIS

learner = OASIS(C=0.1)
learner.learn_triplet({"x": 1.0}, {"x": 0.9}, {"x": 0.1})

knn = MetricKNNClassifier(learner, n_neighbors=1)
knn.learn_one({"x": 1.0}, "A")
knn.learn_one({"x": 0.0}, "B")
knn.predict_one({"x": 0.95})
```

```
'A'
```

OASIS's learned `W` is not symmetric; its distance uses `M = WᵀW`, which is always positive
semi-definite. LEGO needs a target squared distance per pair: `LEGO.percentile_targets`
returns the 5th/95th percentiles of same-class/different-class distances, as in Section 4 of
the paper. Inside `MetricKNNClassifier` these targets are estimated once, from 50 random
pairs of the window after the first 30 samples, and then kept fixed.

## Online learning on river streams

`progressive_val_score` (predict, then learn), `StandardScaler` in front, k = 5,
`window_size=1000`, metric learned online with the default parameters:

| Stream | river `KNNClassifier` (Euclidean) | OASIS | LEGO | POLA |
|---|---|---|---|---|
| Phishing (1,250 samples) | 0.8823 | **0.8911 (+0.88)** | 0.8815 (−0.08) | 0.8871 (+0.48) |
| ImageSegments (2,310 samples) | 0.8705 | 0.9082 (+3.77) | 0.9112 (+4.07) | **0.9138 (+4.33)** |

On ImageSegments (7 classes, 18 features) all three learned metrics beat the Euclidean k-NN by
about 4 points; on Phishing the gains are under one point and LEGO is level with the baseline.

::: dense_armor.learn.metric_learning

---

**Details**: the three algorithms compared as kNN metrics on Iris, Wine, Digits and
20 newsgroups (including `WᵀW` against the symmetrized `W`) are in Dense-Evolution-Discovery,
[Online Metric Learning: OASIS, LEGO, POLA](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/river_metric_learning/).
