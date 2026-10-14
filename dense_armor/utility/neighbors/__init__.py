"""Exact nearest-neighbour classifiers and regressors on a sliding window.

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
