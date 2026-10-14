"""Exact nearest-neighbour classifiers and regressors on a sliding window.

Measured cost (p99 of one learn_one + predict_one, 3000 samples,
window_size=200, n_neighbors=5, 4 features, 1 CPU thread; state size in
bytes):

    KNNClassifier   p99 0.164 ms   13 224 B
    KNNRegressor    p99 0.142 ms   14 600 B

Both classes are O(window_size) in the state and O(window_size * d) per
query. budget_s is 0.001 s.


Both estimators keep the ``window_size`` most recent samples and answer
a query by scanning the window (LazySearch). The distance is the Minkowski
distance with parameter ``p`` (``p=2`` Euclidean, ``p=1`` Manhattan), or
an optional learned metric from :class:`MetricLearner`.
"""

from dense_armor.utility.neighbors.knn import (
    KNNClassifier,
    KNNRegressor,
    LazySearch,
)

__all__ = ["KNNClassifier", "KNNRegressor", "LazySearch"]
