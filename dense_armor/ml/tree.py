"""Decision trees and Mondrian forests grown on the whole table.

References:
    Breiman, L., Friedman, J., Olshen, R., Stone, C. (1984). Classification and
        Regression Trees, Wadsworth, chapter 2: the gini impurity and the
        recursive binary splitting (not in the folder, bibliographic only).
    Lakshminarayanan, B., Roy, D. M., Teh, Y. W. (2014). Mondrian forests:
        efficient online random forests, NIPS, section 2, Algorithms 1 and 2
        (``SampleMondrianTree`` and ``SampleMondrianBlock``, page 4).
    Lakshminarayanan, B., Roy, D. M., Teh, Y. W. (2016). Mondrian forests for
        large-scale regression when uncertainty matters, AISTATS, Algorithm 5 and
        eq. 3: the prediction as a mixture over the nodes of the path, as the
        docstring of the online class writes it (not in the folder,
        bibliographic only).
"""

from __future__ import annotations

import math
from abc import abstractmethod

import numpy as np

from dense_armor.ml.base import MLBase

__all__ = [
    "BatchDecisionTreeClassifier",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "MondrianForestClassifier",
    "MondrianForestRegressor",
]

_STREAM = "the reference defines the algorithm on a stream (a bound or a test on statistics that arrive one example at a time): the algorithm of the reference is written for one example or one window at a time"
STOPPED = {
    "HoeffdingTreeClassifier": "the batch tree that the Hoeffding tree approaches is BatchDecisionTreeClassifier",
    "HoeffdingAnytimeTreeClassifier": "the batch tree that the Hoeffding tree approaches is BatchDecisionTreeClassifier",
    "HoeffdingAdaptiveTreeClassifier": "the batch tree that the Hoeffding tree approaches is BatchDecisionTreeClassifier; the replacement of subtrees on drift is defined on a stream",
    "FIMTDDRegressor": "the reference chapter (Ikonomovska 2012, chapter 6) lies outside the folder and the formulas of the chapter are outside the folder; the model tree with drift detection is defined on a stream",
    "SGTRegressor": _STREAM,
    "SGTClassifier": _STREAM,
}


def _gini(counts: np.ndarray) -> float:
    n = counts.sum()
    return 0.0 if n == 0 else float(1.0 - ((counts / n) ** 2).sum())


class BatchDecisionTreeClassifier(MLBase):
    """Binary decision tree with the gini impurity, grown recursively on the whole table.

    At each node every feature and every midpoint between consecutive sorted
    distinct values is tried, and the split with the smallest weighted gini
    impurity of the two children wins (the first one on a tie). A node is a leaf
    when it is pure, when ``max_depth`` is reached, when it has fewer than
    ``min_samples_split`` rows, or when no split lowers the impurity; a leaf
    predicts its majority class, a tie going to the smallest class.

    Args:
        max_depth: depth limit, or ``None``.
        min_samples_split: smallest node that is split.

    Example:
        >>> m = BatchDecisionTreeClassifier().fit([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1])
        >>> m.predict([[0.5], [2.5]]).tolist(), m.depth_
        ([0, 1], 1)
    """

    def __init__(
        self, max_depth: int | None = None, min_samples_split: int = 2
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> dict:
        counts = np.array([(y == c).sum() for c in self.classes_], dtype=float)
        leaf = {"counts": counts}
        n = len(y)
        if (
            (counts > 0).sum() <= 1
            or n < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return leaf
        parent = _gini(counts)
        best: tuple[float, int | None, float | None] = (parent, None, None)
        for f in range(X.shape[1]):
            order = np.argsort(X[:, f], kind="stable")
            xs, ys = X[order, f], y[order]
            onehot = (ys[:, None] == self.classes_[None, :]).astype(float)
            left = np.cumsum(onehot, axis=0)[:-1]
            for i in np.flatnonzero(xs[:-1] < xs[1:]):
                nl = i + 1
                g = (nl * _gini(left[i]) + (n - nl) * _gini(counts - left[i])) / n
                if g < best[0] - 1e-15:
                    best = (g, f, float((xs[i] + xs[i + 1]) / 2.0))
        if best[1] is None:
            return leaf
        _, bf, thr = best
        m = X[:, bf] <= thr
        return {
            "feature": bf,
            "threshold": thr,
            "left": self._grow(X[m], y[m], depth + 1),
            "right": self._grow(X[~m], y[~m], depth + 1),
        }

    def fit(self, X, y=None):  # type: ignore[override]
        A, t = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(t)
        self.tree_ = self._grow(A, t, 0)
        return self

    def _leaf_counts(self, row: np.ndarray) -> np.ndarray:
        node = self.tree_
        while "feature" in node:
            node = (
                node["left"]
                if row[node["feature"]] <= node["threshold"]
                else node["right"]
            )
        return node["counts"]

    def predict_proba(self, X) -> np.ndarray:
        """Class frequencies of the leaf each row falls in."""
        c = np.array([self._leaf_counts(r) for r in np.asarray(X, dtype=float)])
        return c / c.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        """The majority class of the leaf (the smallest class on a tie)."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    @property
    def depth_(self) -> int:
        """Depth of the fitted tree (0 for a single leaf)."""

        def d(n):
            return 0 if "feature" not in n else 1 + max(d(n["left"]), d(n["right"]))

        return d(self.tree_)


def _score(
    kind: str,
    left: np.ndarray,
    right: np.ndarray,
    nl: np.ndarray,
    nr: np.ndarray,
    lam: float,
) -> np.ndarray:
    if kind == "cls":
        return (left**2).sum(axis=-1) / nl + (right**2).sum(axis=-1) / nr
    if kind == "reg":
        return left[..., 0] ** 2 / nl + right[..., 0] ** 2 / nr
    return left[..., 0] ** 2 / (left[..., 1] + lam) + right[..., 0] ** 2 / (
        right[..., 1] + lam
    )


def _split(X, S, kind, feats, min_leaf, rng, random_thr, lam):
    """Best split ``(score, feature, threshold)`` over ``feats``, or ``None``."""
    n = X.shape[0]
    tot = S.sum(axis=0)
    if kind == "grad":
        base = tot[0] ** 2 / (tot[1] + lam)
    elif kind == "cls":
        base = (tot**2).sum() / n
    else:
        base = tot[0] ** 2 / n
    best = None
    for f in feats:
        col = X[:, f]
        if random_thr:
            lo, hi = col.min(), col.max()
            if lo == hi:
                continue
            thr = rng.uniform(lo, hi)
            mask = col <= thr
            nl = mask.sum()
            if nl < min_leaf or n - nl < min_leaf:
                continue
            left = S[mask].sum(axis=0)
            sc = (
                _score(
                    kind,
                    left,
                    tot - left,
                    np.array(float(nl)),
                    np.array(float(n - nl)),
                    lam,
                )
                - base
            )
            if best is None or sc > best[0]:
                best = (float(sc), int(f), float(thr))
            continue
        order = np.argsort(col, kind="stable")
        cs = np.cumsum(S[order], axis=0)[:-1]
        sv = col[order]
        valid = sv[:-1] < sv[1:]
        nl = np.arange(1, n, dtype=float)
        valid &= (nl >= min_leaf) & (n - nl >= min_leaf)
        if not valid.any():
            continue
        sc = _score(kind, cs, tot - cs, nl, n - nl, lam) - base
        sc = np.where(valid, sc, -np.inf)
        i = int(np.argmax(sc))
        if best is None or sc[i] > best[0]:
            best = (float(sc[i]), int(f), float((sv[i] + sv[i + 1]) / 2.0))
    return best


class _Tree(MLBase):
    """Greedy tree: ``kind`` is ``"cls"``, ``"reg"`` or ``"grad"``."""

    def _grow(self, X, S, depth, rng, kind):
        n, d = X.shape
        leaf = {"stats": S.sum(axis=0), "n": n}
        if (
            self.max_depth is not None and depth >= self.max_depth
        ) or n < self.min_samples_split:
            return leaf
        if kind == "cls" and (S.sum(axis=0) == n).any():
            return leaf
        k = d if self.max_features is None else min(self.max_features, d)
        feats = rng.permutation(d)[:k] if k < d else np.arange(d)
        sp = _split(
            X,
            S,
            kind,
            feats,
            self.min_samples_leaf,
            rng,
            getattr(self, "splitter", "best") == "random",
            getattr(self, "lambda_", 0.0),
        )
        gamma = getattr(self, "gamma", 0.0)
        if sp is None or sp[0] <= (2.0 * gamma if kind == "grad" else 1e-12):
            return leaf
        _, f, thr = sp
        m = X[:, f] <= thr
        return {
            "feature": f,
            "threshold": thr,
            "left": self._grow(X[m], S[m], depth + 1, rng, kind),
            "right": self._grow(X[~m], S[~m], depth + 1, rng, kind),
            "n": n,
        }

    def _leaves(self, X) -> list[dict]:
        out = []
        for row in np.asarray(X, dtype=float):
            node = self.tree_
            while "feature" in node:
                node = (
                    node["left"]
                    if row[node["feature"]] <= node["threshold"]
                    else node["right"]
                )
            out.append(node)
        return out

    def _depth(self, node=None) -> int:
        node = self.tree_ if node is None else node
        return (
            0
            if "feature" not in node
            else 1 + max(self._depth(node["left"]), self._depth(node["right"]))
        )

    @property
    def depth_(self) -> int:
        """Depth of the fitted tree (0 for a single leaf)."""
        return self._depth()


class DecisionTreeClassifier(_Tree):
    """CART classification tree: splits that maximise the Gini decrease.

    Args:
        max_depth: depth limit, or ``None``.
        min_samples_split: smallest node that is split.
        min_samples_leaf: smallest number of rows in a leaf.
        max_features: number of features tried at each split, or ``None`` for all.
        splitter: ``"best"`` or ``"random"`` (a random threshold, as in extremely randomised trees).
        seed: seed of the random choices.

    Example:
        >>> X = [[0.0], [1.0], [2.0], [3.0]]
        >>> m = DecisionTreeClassifier().fit(X, [0, 0, 1, 1])
        >>> m.predict([[0.5], [2.5]]).tolist(), m.depth_
        ([0, 1], 1)
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | None = None,
        splitter: str = "best",
        seed: int = 0,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.splitter = splitter
        self.seed = seed

    def fit(self, X, y=None, sample_weight=None):  # type: ignore[override]
        A = np.asarray(X, dtype=float)
        t = np.asarray(y)
        self.classes_ = np.unique(t)
        S = (t[:, None] == self.classes_[None, :]).astype(float)
        if sample_weight is not None:
            S = S * np.asarray(sample_weight, dtype=float)[:, None]
        self.tree_ = self._grow(A, S, 0, np.random.default_rng(self.seed), "cls")
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Class frequencies of the leaf each row falls in."""
        st = np.array([lf["stats"] for lf in self._leaves(X)])
        return st / st.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        """The most frequent class of the leaf."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class DecisionTreeRegressor(_Tree):
    """CART regression tree: splits that maximise the variance reduction; leaves hold the mean.

    Args:
        max_depth: depth limit, or ``None``.
        min_samples_split: smallest node that is split.
        min_samples_leaf: smallest number of rows in a leaf.
        max_features: number of features tried at each split, or ``None`` for all.
        splitter: ``"best"`` or ``"random"``.
        seed: seed of the random choices.

    Example:
        >>> m = DecisionTreeRegressor().fit([[0.0], [1.0], [2.0], [3.0]], [1.0, 1.0, 5.0, 5.0])
        >>> m.predict([[0.2], [2.8]]).tolist()
        [1.0, 5.0]
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | None = None,
        splitter: str = "best",
        seed: int = 0,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.splitter = splitter
        self.seed = seed

    def fit(self, X, y=None):  # type: ignore[override]
        A = np.asarray(X, dtype=float)
        t = np.asarray(y, dtype=float)
        self.tree_ = self._grow(
            A, t[:, None], 0, np.random.default_rng(self.seed), "reg"
        )
        return self

    def predict(self, X) -> np.ndarray:
        """The mean target of the leaf."""
        return np.array([lf["stats"][0] / lf["n"] for lf in self._leaves(X)])


class _Mondrian(MLBase):
    """Mondrian trees of Algorithms 1 and 2 (page 4) and the mixture prediction of Algorithm 5 (2016)."""

    @abstractmethod
    def _stats(self, y: np.ndarray):
        """Statistics of the targets of a block."""

    def _block(self, X, y, tau_parent, rng):
        lx, ux = X.min(axis=0), X.max(axis=0)
        ext = ux - lx
        node = {"lx": lx, "ux": ux, "n": len(y), "stat": self._stats(y)}
        total = float(ext.sum())
        e = rng.exponential(1.0 / total) if total > 0 else math.inf
        if tau_parent + e < self.lifetime:
            node["tau"] = tau_parent + e
            f = int(rng.choice(X.shape[1], p=ext / total))
            thr = float(rng.uniform(lx[f], ux[f]))
            m = X[:, f] <= thr
            if m.all() or not m.any():
                node["tau"] = self.lifetime
                return node
            node["feature"], node["threshold"] = f, thr
            node["left"] = self._block(X[m], y[m], node["tau"], rng)
            node["right"] = self._block(X[~m], y[~m], node["tau"], rng)
        else:
            node["tau"] = self.lifetime
        return node

    def _fit_trees(self, A, y):
        rng = np.random.default_rng(self.seed)
        self.trees_ = [self._block(A, y, 0.0, rng) for _ in range(self.n_trees)]

    @staticmethod
    def _branch(node, x, parent_tau) -> float:
        delta = node["tau"] - parent_tau
        if math.isinf(delta):
            delta = 1e300
        eta = float(
            np.maximum(x - node["ux"], 0.0).sum()
            + np.maximum(node["lx"] - x, 0.0).sum()
        )
        if eta <= 0.0 or delta <= 0.0:
            return 0.0
        return 1.0 - math.exp(-delta * eta)

    def _mixture(self, tree, x):
        """Weighted statistics of the components along the path of ``x``."""
        path, node = [], tree
        while True:
            path.append(node)
            if "feature" not in node:
                break
            node = (
                node["left"]
                if x[node["feature"]] <= node["threshold"]
                else node["right"]
            )
        comps, not_sep, parent_tau, block = [], 1.0, 0.0, path[0]
        for node in path:
            p = self._branch(node, x, parent_tau)
            if p > 0.0:
                comps.append((not_sep * p, block["stat"]))
            not_sep *= 1.0 - p
            parent_tau, block = node["tau"], node
        comps.append((not_sep, path[-1]["stat"]))
        comps = [(w, s) for w, s in comps if w > 0.0]
        tot = sum(w for w, _ in comps)
        return [(w / tot, s) for w, s in comps]


class MondrianForestClassifier(_Mondrian):
    """Forest of Mondrian trees; the probability is the mixture of the class histograms along the path.

    Each tree is sampled by Algorithms 1 and 2 of Lakshminarayanan et al. (2014):
    a block is split after an exponential time of rate equal to the sum of its
    side lengths, along a side chosen in proportion to its length, at a uniform
    location, while the time stays below ``lifetime``. A query ``x`` branches off
    just before a node with probability ``1 - exp(-Delta eta(x))``; each such
    component uses the class histogram of the block that contains the node, and
    the forest averages the trees.

    Args:
        n_trees: number of trees.
        lifetime: lifetime of the Mondrian process (``inf`` splits until the points are alone).
        seed: seed of the random choices.

    Example:
        >>> X = [[0.0], [0.1], [1.0], [1.1]]
        >>> m = MondrianForestClassifier(n_trees=8, seed=1).fit(X, [0, 0, 1, 1])
        >>> m.predict([[0.05], [1.05]]).tolist()
        [0, 1]
    """

    def __init__(
        self, n_trees: int = 10, lifetime: float = math.inf, seed: int = 0
    ) -> None:
        self.n_trees = n_trees
        self.lifetime = lifetime
        self.seed = seed

    def _stats(self, y):
        return np.array([(y == c).sum() for c in self.classes_], dtype=float)

    def fit(self, X, y=None):  # type: ignore[override]
        A, t = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(t)
        self._fit_trees(A, t)
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Mean over the trees of the mixture of the normalised class histograms."""
        out = []
        for x in np.asarray(X, dtype=float):
            p = np.zeros(len(self.classes_))
            for tree in self.trees_:
                for w, c in self._mixture(tree, x):
                    p += w * c / c.sum() / len(self.trees_)
            out.append(p)
        return np.array(out)

    def predict(self, X) -> np.ndarray:
        """The class with the highest probability."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class MondrianForestRegressor(_Mondrian):
    """Forest of Mondrian trees; the prediction is the mean of the mixture over the path.

    The components are the Gaussians of the blocks along the path, with mean and
    variance of the targets of the block (eq. 3 of the 2016 paper); the forest
    averages the trees.

    Args:
        n_trees: number of trees.
        lifetime: lifetime of the Mondrian process.
        seed: seed of the random choices.

    Example:
        >>> m = MondrianForestRegressor(n_trees=8, seed=1).fit([[0.0], [0.1], [1.0], [1.1]], [0.0, 0.0, 4.0, 4.0])
        >>> m.predict([[0.05], [1.05]]).round().tolist()
        [0.0, 4.0]
    """

    def __init__(
        self, n_trees: int = 10, lifetime: float = math.inf, seed: int = 0
    ) -> None:
        self.n_trees = n_trees
        self.lifetime = lifetime
        self.seed = seed

    def _stats(self, y):
        return (float(y.mean()), float(y.var(ddof=1)) if len(y) > 1 else 0.0)

    def fit(self, X, y=None):  # type: ignore[override]
        self._fit_trees(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, X, return_std: bool = False):
        """The mean of the mixture, averaged over the trees; the standard deviation if asked."""
        mu, var = [], []
        for x in np.asarray(X, dtype=float):
            m = v = 0.0
            for tree in self.trees_:
                comps = self._mixture(tree, x)
                mt = sum(w * s[0] for w, s in comps)
                m += mt / len(self.trees_)
                v += sum(w * (s[1] + (s[0] - mt) ** 2) for w, s in comps) / len(
                    self.trees_
                )
            mu.append(m)
            var.append(v)
        return (np.array(mu), np.sqrt(var)) if return_std else np.array(mu)
