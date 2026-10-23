"""Batch decision tree (CART) for offline learning.

A batch CART tree, fitted on the whole dataset in one call. It is the
offline counterpart of the online Hoeffding tree: same classification
task, same gini impurity, but the tree is built once on a full
training set instead of being updated one sample at a time.

Reference: Breiman, Friedman, Olshen, Stone (1984). *Classification and
Regression Trees.* Wadsworth. Chapter 2: gini impurity and the
recursive binary splitting.

Example:
    >>> from dense_armor.utility.tree.batch import BatchDecisionTreeClassifier
    >>> X = [{"x": 0.0}, {"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
    >>> y = [0, 0, 1, 1]
    >>> clf = BatchDecisionTreeClassifier(max_depth=2)
    >>> _ = clf.fit(X, y)
    >>> clf.predict_one({"x": 0.5})
    0
    >>> clf.predict_one({"x": 2.5})
    1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dense_armor.roles import Classifier


@dataclass
class _Node:
    """One node of the CART tree."""

    feature: str | None = None
    threshold: float | None = None
    left: _Node | None = None
    right: _Node | None = None
    counts: dict = field(default_factory=dict)

    def is_leaf(self) -> bool:
        return self.feature is None


def _gini(counts: dict) -> float:
    """Gini impurity of a class-count dict."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    p = np.array(list(counts.values()), dtype=float) / total
    return float(1.0 - np.sum(p * p))


def _majority(counts: dict) -> Any:
    """Most frequent class, ties broken by the smallest class."""
    if not counts:
        return None
    best = max(counts.values())
    return min(c for c, v in counts.items() if v == best)


def _counts(y: np.ndarray) -> dict:
    """Class counts of a numpy array."""
    out: dict = {}
    for v in y:
        key = v.item() if hasattr(v, "item") else v
        out[key] = out.get(key, 0) + 1
    return out


def _best_split(
    X: np.ndarray, y: np.ndarray, feature_names: list[str],
) -> tuple[str | None, float | None, float]:
    """Return (best_feature, best_threshold, best_gini).

    Vectorised scan of every feature and every midpoint between
    consecutive sorted values. For each feature the samples are sorted
    once (O(n log n)), the class counts to the left and to the right of
    each midpoint are accumulated with a cumsum on a one-hot matrix
    (O(n K)), and the weighted gini impurity of the two children is
    computed for all midpoints at once. The split that minimises the
    weighted gini wins; ties go to the lower feature index. If no
    split improves on the parent gini, returns ``(None, None,
    parent_gini)``.
    """
    n, d = X.shape
    parent_counts = _counts(y)
    parent = _gini(parent_counts)
    classes = sorted(parent_counts.keys())
    cidx = {c: i for i, c in enumerate(classes)}
    K = len(classes)
    Y = np.zeros((n, K))
    for i, v in enumerate(y):
        Y[i, cidx[v.item() if hasattr(v, "item") else v]] = 1.0

    best_f = None
    best_t = None
    best_g = parent

    for j in range(d):
        col = X[:, j]
        order = np.argsort(col, kind="stable")
        col_sorted = col[order]
        if col_sorted[0] == col_sorted[-1]:
            continue
        valid = col_sorted[:-1] < col_sorted[1:]
        Y_sorted = Y[order]
        cum = np.cumsum(Y_sorted, axis=0)  # (n, K)
        total = cum[-1]  # (K,)
        left_counts = cum[:-1]  # (n-1, K)
        right_counts = total[None, :] - left_counts  # (n-1, K)
        n_left: np.ndarray = np.arange(1, n, dtype=float)
        n_right = n - n_left
        g_left = 1.0 - np.sum((left_counts / n_left[:, None]) ** 2, axis=1)
        g_right = 1.0 - np.sum(
            (right_counts / n_right[:, None]) ** 2, axis=1)
        weighted = (n_left * g_left + n_right * g_right) / n
        wv = np.where(valid, weighted, np.inf)
        k = int(np.argmin(wv))
        if wv[k] < best_g - 1e-12:
            best_g = float(wv[k])
            best_f = feature_names[j]
            best_t = 0.5 * (col_sorted[k] + col_sorted[k + 1])

    return best_f, best_t, best_g


class BatchDecisionTreeClassifier(Classifier):
    """Batch CART classifier.

    A binary decision tree trained on the whole dataset in one call to
    :meth:`fit`, with the gini impurity of Breiman, Friedman, Olshen
    and Stone (1984), chapter 2. The tree is built recursively: at
    each node, :func:`_best_split` scans every feature and every
    midpoint between consecutive sorted values, and the split that
    minimises the weighted gini impurity of the two children wins.

    A node is a leaf when it is pure (one class), when ``max_depth`` is
    reached, when ``min_samples_split`` is not met, or when no split
    reduces the gini impurity. The prediction of a leaf is the majority
    class, ties broken by the smallest class.

    This tree is the offline counterpart of
    :class:`~dense_armor.utility.tree.hoeffding.HoeffdingTreeClassifier`.
    It is fitted on the whole training set once, and it never sees a
    sample after the fit. :meth:`learn_one` and :meth:`predict_one`
    after the fit are still available (the base class contract), but
    they do not change the tree: the tree is frozen at the end of
    :meth:`fit`. The two learners answer the same question with two
    protocols: one pass on the full training set (this class), one
    sample at a time (the online Hoeffding tree).

    When a feature of the tree is missing from the dict given to
    :meth:`predict_one` or :meth:`predict_proba_one`, or its value is not
    a number, the answer is the majority class (the distribution) of the
    node where the path stops, and not of a leaf. ``budget_s`` is the
    budget of the per-sample calls (``predict_one`` and ``learn_one``),
    not of :meth:`fit`. ``memory_class`` describes the fitted tree; the
    memory check of ``check_estimator`` reads only ``"O(1)"`` and
    ``"O(window)"``, so it is skipped for this class.

    Args:
        max_depth: maximum depth of the tree. ``None`` means no limit.
        min_samples_split: minimum number of samples at a node to
            consider a split. Default 2 (a node with 2 samples can be
            split into two leaves of 1).

    Examples:
        >>> from dense_armor.utility.tree.batch import BatchDecisionTreeClassifier
        >>> X = [{"x": 0.0}, {"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
        >>> y = [0, 0, 1, 1]
        >>> clf = BatchDecisionTreeClassifier(max_depth=2)
        >>> _ = clf.fit(X, y)
        >>> clf.predict_one({"x": 0.5})
        0
        >>> clf.predict_one({"x": 2.5})
        1
    """

    budget_s: float = 1e-3
    memory_class: str = "O(n_nodes * n_classes)"

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
    ) -> None:
        if max_depth is not None and max_depth < 0:
            raise ValueError(f"max_depth must be >= 0, got {max_depth}")
        if min_samples_split < 2:
            raise ValueError(
                f"min_samples_split must be >= 2, got {min_samples_split}"
            )
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self._root: _Node | None = None
        self._feature_names: list[str] = []

    def _unit_test_skips(self) -> set:
        return {"check_step_is_pure", "check_step_is_jittable"}

    def _build(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        depth: int,
    ) -> _Node:
        counts = _counts(y)
        leaf = _Node(counts=counts)
        if len(counts) <= 1:
            return leaf
        if self.max_depth is not None and depth >= self.max_depth:
            return leaf
        if len(y) < self.min_samples_split:
            return leaf
        feature, threshold, _g = _best_split(X, y, feature_names)
        if feature is None:
            return leaf
        j = feature_names.index(feature)
        mask = X[:, j] <= threshold
        leaf.feature = feature
        leaf.threshold = threshold
        leaf.counts = counts
        leaf.left = self._build(X[mask], y[mask], feature_names, depth + 1)
        leaf.right = self._build(
            X[~mask], y[~mask], feature_names, depth + 1
        )
        return leaf

    def fit(
        self,
        X: Any,
        y: Any = None,
        t: Any = None,
        epochs: int = 1,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> BatchDecisionTreeClassifier:
        """Fit the tree on the whole dataset.

        Args:
            X: list of feature dicts.
            y: list of labels, same length as ``X``.
            t: accepted for the interface of ``Root.fit``; unused.
            epochs: accepted for the interface of ``Root.fit``; unused.
            shuffle: accepted for the interface of ``Root.fit``; unused.
            seed: accepted for the interface of ``Root.fit``; unused.

        Returns:
            ``self``.

        Raises:
            ValueError: on empty ``X``, on ``y`` equal to ``None``, on a
                length mismatch, on a row without one of the features of
                the first row, on a value that is not a number, or on a
                NaN or infinite value. The features are the keys of the
                first row; the other keys of the later rows are ignored.
        """
        if y is None:
            raise ValueError("fit needs the labels y")
        if len(X) == 0:
            raise ValueError("cannot fit on an empty dataset")
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows, y has {len(y)}")
        feature_names = sorted(X[0].keys())
        rows = []
        for i, row in enumerate(X):
            try:
                rows.append([float(row[k]) for k in feature_names])
            except KeyError as err:
                raise ValueError(
                    f"row {i} has no feature {err.args[0]!r}"
                ) from None
            except (TypeError, ValueError):
                raise ValueError(f"row {i} holds a value that is not a number") from None
        X_arr = np.asarray(rows, dtype=float)
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("X holds a NaN or an infinite value")
        self._feature_names = feature_names
        y_arr = np.asarray(y)
        self._root = self._build(X_arr, y_arr, feature_names, 0)
        return self

    def _predict_leaf(self, x: dict) -> _Node | None:
        if self._root is None:
            return None
        node: _Node | None = self._root
        while node is not None and not node.is_leaf():
            v = x.get(node.feature)
            if v is None:
                return node
            try:
                v = float(v)
            except (TypeError, ValueError):
                return node
            thr = node.threshold
            assert thr is not None  # invariant: feature and threshold set together
            node = node.left if v <= thr else node.right
        return node

    def predict_one(self, x: dict, t: float | None = None) -> Any:
        """Majority class of the leaf that ``x`` reaches."""
        leaf = self._predict_leaf(x)
        if leaf is None:
            return None
        return _majority(leaf.counts)

    def predict_proba_one(self, x: dict, t: float | None = None) -> dict:
        """Class distribution of the leaf that ``x`` reaches."""
        leaf = self._predict_leaf(x)
        if leaf is None:
            return {}
        total = sum(leaf.counts.values())
        if total == 0:
            return {}
        return {c: v / total for c, v in leaf.counts.items()}

    def learn_one(self, x: dict, y: Any, t: float | None = None) -> BatchDecisionTreeClassifier:
        """No-op after the fit; the tree is frozen."""
        return self
