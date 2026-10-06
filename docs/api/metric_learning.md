# Metric learning (online)

A k-nearest-neighbours classifier is only as good as its distance. `metric_learning` learns
that distance online, one pair or triplet at a time, with three algorithms ported from their
papers: **OASIS** (Chechik et al. 2009, bilinear similarity from triplets), **LEGO** (Jain et
al. 2008, LogDet-regularized Mahalanobis matrix from pairs with a target distance) and
**POLA** (Shalev-Shwartz et al. 2004, pseudo-metric from similar/dissimilar pairs, kept
positive semi-definite).

A metric learner does not fit river's `Classifier`, `Regressor` or `Transformer` types: it
learns from two or three samples at a time and outputs a distance. `MetricLearner` is the
base class; `MetricKNNClassifier` plugs a trained learner into a river k-NN classifier.

```bash
pip install dense-armor[river]
```

```python
from dense_armor.utility.metric_learning import MetricKNNClassifier, OASIS

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
the paper.

::: dense_armor.utility.metric_learning

---

**Details**: the three algorithms compared as kNN metrics on Iris, Wine, Digits and
20 newsgroups (including `WᵀW` against the symmetrized `W`) are in Dense-Evolution-Discovery,
[Online Metric Learning: OASIS, LEGO, POLA](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/river_metric_learning/).
