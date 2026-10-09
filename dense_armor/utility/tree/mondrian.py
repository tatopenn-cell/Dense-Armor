"""Online Mondrian forests for regression and classification.

A Mondrian process (Roy and Teh 2009) is a random hierarchical
partition of the input space whose splits occur at exponentially
distributed times, with rate equal to the sum of the extents of the
block being split. A Mondrian tree is the restriction of a Mondrian
process to a finite set of points: the tree is finite, the split
dimension is chosen with probability proportional to the extent of
the block along that dimension, and the split location is uniform in
that extent. This is the tree distribution of Lakshminarayanan et al.
2014 (classification) and 2016 (regression).

The online extension follows Algorithms 3 and 4 of the 2014 paper:
when a new point falls outside a node's data extent, an additional
split above that node is sampled with rate equal to the extent
extension; the dimension is chosen with probability proportional to
its extension and the cut is uniform in it.

A forest is a collection of independent Mondrian trees. The
prediction of the forest averages the tree predictions. For
regression, each tree returns a Gaussian mixture over the labels
(2016, eq. 3): along the path from the root to the leaf that contains
the query point, the probability of branching off just before each
node contributes the weight of that node's posterior. The predictive
variance is the mixture variance, so it grows as the query moves away
from the training data.

Args:
    n_trees: number of trees in the forest.
    lifetime: lifetime parameter of the Mondrian process. ``inf``
        (default) means the tree can grow as deep as the data allows.
    max_nodes: maximum number of nodes per tree. Default ``10000``.
    seed: base random seed; tree ``i`` uses ``seed + i``.
    min_samples_split: minimum samples at a leaf before the tree
        attempts to grow a new split.
    task: ``"regression"`` or ``"classification"``.

Raises:
    ValueError: on a non-positive parameter or a bad ``task``.

Examples:
    >>> from dense_armor.utility.tree.mondrian import MondrianForestRegressor
    >>> f = MondrianForestRegressor(n_trees=5, seed=0)
    >>> for i in range(500):
    ...     x = (i % 50) / 49.0
    ...     _ = f.learn_one({"x": x}, y=x)
    >>> e = f.predict_one({"x": 0.5}, return_estimate=True)
    >>> e.var >= 0.0
    True

References:
    Lakshminarayanan, B., Roy, D. M., Teh, Y. W. (2014). Mondrian
        forests: efficient online random forests. In NIPS, section 2,
        section 3.1, Algorithms 2, 3 and 4.
    Lakshminarayanan, B., Roy, D. M., Teh, Y. W. (2016). Mondrian
        forests for large-scale regression when uncertainty matters.
        In AISTATS, section 3, eq. 3.
    Roy, D. M., Teh, Y. W. (2009). The Mondrian process. In NIPS.
"""

import math
import random
from typing import Any

from dense_armor.roles import Classifier, Estimate, Regressor


def _as_dict(x: Any) -> dict:
    if hasattr(x, "to_dict"):
        return dict(x.to_dict())
    return dict(x)


class _LeafState:
    """Leaf statistics: Gaussian for regression, counts for classification."""

    __slots__ = ("counts_", "m2_", "mean_", "n_")

    def __init__(self) -> None:
        self.n_ = 0
        self.mean_ = 0.0
        self.m2_ = 0.0
        self.counts_: dict = {}

    def update_reg(self, y: float) -> None:
        self.n_ += 1
        d = y - self.mean_
        self.mean_ += d / self.n_
        self.m2_ += d * (y - self.mean_)

    def update_cls(self, y: Any) -> None:
        self.n_ += 1
        self.counts_[y] = self.counts_.get(y, 0) + 1

    def copy(self) -> "_LeafState":
        c = _LeafState()
        c.n_ = self.n_
        c.mean_ = self.mean_
        c.m2_ = self.m2_
        c.counts_ = dict(self.counts_)
        return c

    @property
    def var(self) -> float:
        if self.n_ < 2:
            return 0.0
        return self.m2_ / (self.n_ - 1)


class _MNode:
    """Node of a Mondrian tree with split time and block bounds."""

    __slots__ = (
        "depth",
        "feature",
        "lb_",
        "leaf",
        "left",
        "lx_",
        "right",
        "tau",
        "threshold",
        "ub_",
        "ux_",
    )

    def __init__(self, depth: int, tau: float) -> None:
        self.depth = depth
        self.leaf = _LeafState()
        self.feature: str | None = None
        self.threshold: float | None = None
        self.tau = tau
        self.lb_: dict[str, float] = {}
        self.ub_: dict[str, float] = {}
        self.lx_: dict[str, float] = {}
        self.ux_: dict[str, float] = {}
        self.left: _MNode | None = None
        self.right: _MNode | None = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


class _MondrianTree:
    """One online Mondrian tree with a task-specific leaf."""

    def __init__(
        self,
        rng: random.Random,
        lifetime: float,
        max_nodes: int | None,
        min_samples_split: int,
        task: str,
    ) -> None:
        self.rng = rng
        self.lifetime = lifetime
        self.max_nodes = max_nodes
        self.min_samples_split = min_samples_split
        self.task = task
        self.root = _MNode(0, lifetime)
        self.n_nodes = 1

    def _extend_bounds(self, node: _MNode, d: dict) -> bool:
        """Update block and data bounds; True if the data extent grew."""
        grew = False
        for f, v in d.items():
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(fv):
                continue
            if f not in node.lb_:
                node.lb_[f] = fv
                node.ub_[f] = fv
                node.lx_[f] = fv
                node.ux_[f] = fv
                continue
            node.lb_[f] = min(node.lb_[f], fv)
            node.ub_[f] = max(node.ub_[f], fv)
            if fv < node.lx_[f]:
                node.lx_[f] = fv
                grew = True
            if fv > node.ux_[f]:
                node.ux_[f] = fv
                grew = True
        return grew

    def _sample_new_split(self, node: _MNode, d: dict, parent_tau: float):
        """Return ``(feature, cut, tau)`` for a new split above ``node``."""
        rates: dict[str, float] = {}
        for f, v in d.items():
            if v is None or f not in node.lx_:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(fv):
                continue
            e_l = max(node.lx_[f] - fv, 0.0)
            e_u = max(fv - node.ux_[f], 0.0)
            rate = e_l + e_u
            if rate > 0.0:
                rates[f] = rate
        total = sum(rates.values())
        if total <= 0.0:
            return None
        e_time = self.rng.expovariate(total)
        tau_new = parent_tau + e_time
        if tau_new >= node.tau:
            return None
        if tau_new >= self.lifetime:
            return None
        feats = list(rates)
        weights = [rates[f] for f in feats]
        chosen = self.rng.choices(feats, weights=weights, k=1)[0]
        raw = d.get(chosen)
        if raw is None:
            return None
        fv = float(raw)
        if fv > node.ux_[chosen]:
            lo, hi = node.ux_[chosen], fv
        else:
            lo, hi = fv, node.lx_[chosen]
        if hi - lo <= 0.0:
            return None
        cut = self.rng.uniform(lo, hi)
        return chosen, cut, tau_new

    def _descend(self, node: _MNode, d: dict) -> _MNode | None:
        f = node.feature
        s = node.threshold
        if f is None or s is None:
            return None
        v = d.get(f)
        if v is None:
            return None
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(fv):
            return None
        return node.left if fv <= s else node.right

    def _can_grow(self) -> bool:
        return self.max_nodes is None or self.n_nodes + 2 <= self.max_nodes

    def _update(self, node: _MNode, y: Any) -> None:
        if self.task == "regression":
            node.leaf.update_reg(float(y))
        else:
            node.leaf.update_cls(y)

    def learn_one(self, d: dict, y: Any) -> None:
        """ExtendMondrianBlock (Algorithm 4 of the 2014 paper) along the path.

        At each node the new split above it is sampled from the extension of
        the node's box by ``x`` (the box before ``x`` is added); if one is
        introduced, ``x`` goes to a new sibling leaf and the walk stops,
        otherwise the box grows and the walk continues to the child. Every
        node keeps the statistics of the points that reached its block.
        """
        cur = self.root
        parent: _MNode | None = None
        parent_tau = 0.0
        while True:
            candidate = None
            if cur.lx_ and self._can_grow():
                candidate = self._sample_new_split(cur, d, parent_tau)
            if candidate is not None:
                f, cut, tau_new = candidate
                new_parent = _MNode(cur.depth, tau_new)
                new_parent.feature = f
                new_parent.threshold = cut
                new_parent.lb_ = dict(cur.lb_)
                new_parent.ub_ = dict(cur.ub_)
                new_parent.lx_ = dict(cur.lx_)
                new_parent.ux_ = dict(cur.ux_)
                new_parent.leaf = cur.leaf.copy()
                self._extend_bounds(new_parent, d)
                self._update(new_parent, y)
                sibling = _MNode(cur.depth + 1, self.lifetime)
                self._extend_bounds(sibling, d)
                self._update(sibling, y)
                if float(d[f]) <= cut:
                    new_parent.left, new_parent.right = sibling, cur
                else:
                    new_parent.left, new_parent.right = cur, sibling
                stack = [cur]
                while stack:
                    n = stack.pop()
                    n.depth += 1
                    stack.extend(c for c in (n.left, n.right) if c is not None)
                if parent is None:
                    self.root = new_parent
                elif parent.left is cur:
                    parent.left = new_parent
                else:
                    parent.right = new_parent
                self.n_nodes += 2
                return
            self._extend_bounds(cur, d)
            self._update(cur, y)
            if cur.is_leaf:
                if cur.leaf.n_ >= self.min_samples_split and self._can_grow():
                    self._try_local_split(cur, parent_tau)
                return
            parent, parent_tau = cur, cur.tau
            nxt = self._descend(cur, d)
            if nxt is None:
                return
            cur = nxt

    def _try_local_split(self, node: _MNode, parent_tau: float) -> None:
        """Sample a split at the node using the current bounds."""
        if node.feature is not None:
            return
        extents: dict[str, float] = {}
        for f in node.lb_:
            extents[f] = node.ub_[f] - node.lb_[f]
        total = sum(e for e in extents.values() if e > 0.0)
        if total <= 0.0:
            return
        e_time = self.rng.expovariate(total)
        tau_new = parent_tau + e_time
        if tau_new >= self.lifetime:
            return
        feats = [f for f, e in extents.items() if e > 0.0]
        weights = [extents[f] for f in feats]
        chosen = self.rng.choices(feats, weights=weights, k=1)[0]
        lo, hi = node.lb_[chosen], node.ub_[chosen]
        if hi - lo <= 0.0:
            return
        cut = self.rng.uniform(lo, hi)
        node.feature = chosen
        node.threshold = cut
        node.tau = tau_new
        node.left = _MNode(node.depth + 1, self.lifetime)
        node.right = _MNode(node.depth + 1, self.lifetime)
        self.n_nodes += 2

    def _branch_prob(self, node: _MNode, x: dict, parent_tau: float) -> float:
        """Probability of branching off just before ``node``.

        ``1 - exp(-Delta * eta)`` with ``Delta = tau_j - tau_parent(j)`` and
        ``eta`` the distance of ``x`` outside the box of ``node``
        (Algorithm 5 of the 2016 paper, lines 5 and 6).
        """
        delta = node.tau - parent_tau
        if math.isinf(delta):
            delta = 1e300
        eta = 0.0
        for f, v in x.items():
            if v is None or f not in node.lx_:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(fv):
                continue
            eta += max(fv - node.ux_[f], 0.0) + max(node.lx_[f] - fv, 0.0)
        if eta <= 0.0 or delta <= 0.0:
            return 0.0
        return 1.0 - math.exp(-delta * eta)

    def _path(self, d: dict) -> list:
        path: list = []
        cur: _MNode | None = self.root
        while cur is not None:
            path.append(cur)
            if cur.is_leaf:
                return path
            cur = self._descend(cur, d)
        return path

    def predict(self, d: dict):
        """Return ``(mean, var, proba)`` for the query point.

        Algorithm 5 of the 2016 paper: walking from the root, ``x`` branches
        off just before node ``j`` with probability ``1 - exp(-Delta_j
        eta_j(x))``; that component uses the statistics of the block that
        contains ``j`` (its parent), the last one the statistics of the
        leaf. The mixture of eq. 3 gives the mean and the variance; for
        classification each component is a normalised class histogram.
        """
        path = self._path(d)
        if not path:
            return 0.0, 0.0, {}
        comps: list[tuple[float, _LeafState]] = []
        not_sep = 1.0
        parent_tau = 0.0
        block = path[0]
        for node in path:
            p = self._branch_prob(node, d, parent_tau)
            if p > 0.0:
                comps.append((not_sep * p, block.leaf))
            not_sep *= 1.0 - p
            parent_tau = node.tau
            block = node
        comps.append((not_sep, path[-1].leaf))
        comps = [(w, st) for w, st in comps if w > 0.0 and st.n_ > 0]
        tot_w = sum(w for w, _ in comps)
        if tot_w <= 0.0:
            return 0.0, 0.0, {}
        if self.task == "classification":
            proba: dict = {}
            for w, st in comps:
                tot = sum(st.counts_.values())
                for k, c in st.counts_.items():
                    proba[k] = proba.get(k, 0.0) + w / tot_w * c / tot
            return 0.0, 0.0, proba
        mu = sum(w * st.mean_ for w, st in comps) / tot_w
        var = sum(w * (st.var + (st.mean_ - mu) ** 2) for w, st in comps) / tot_w
        return mu, var, {}


class _BatchMondrianTree:
    """Batch sampler of a Mondrian tree on a finite data set."""

    def __init__(
        self,
        rng: random.Random,
        lifetime: float,
        min_samples_split: int,
    ) -> None:
        self.rng = rng
        self.lifetime = lifetime
        self.min_samples_split = min_samples_split

    def sample(self, d_list: list) -> _MNode:
        root = _MNode(0, 0.0)
        self._recurse(root, d_list, parent_tau=0.0)
        return root

    def _recurse(self, node: _MNode, d_list: list, parent_tau: float) -> None:
        if len(d_list) < self.min_samples_split:
            return
        bounds: dict[str, tuple[float, float]] = {}
        for d in d_list:
            for f, v in d.items():
                if v is None:
                    continue
                fv = float(v)
                if f not in bounds:
                    bounds[f] = (fv, fv)
                else:
                    lo, hi = bounds[f]
                    bounds[f] = (min(lo, fv), max(hi, fv))
        extents = {f: hi - lo for f, (lo, hi) in bounds.items()}
        total = sum(e for e in extents.values() if e > 0.0)
        if total <= 0.0:
            return
        e_time = self.rng.expovariate(total)
        tau = parent_tau + e_time
        if tau >= self.lifetime:
            return
        feats = [f for f, e in extents.items() if e > 0.0]
        weights = [extents[f] for f in feats]
        chosen = self.rng.choices(feats, weights=weights, k=1)[0]
        lo, hi = bounds[chosen]
        cut = self.rng.uniform(lo, hi)
        node.feature = chosen
        node.threshold = cut
        node.tau = tau
        left_list = [d for d in d_list if float(d[chosen]) <= cut]
        right_list = [d for d in d_list if float(d[chosen]) > cut]
        node.left = _MNode(node.depth + 1, tau)
        node.right = _MNode(node.depth + 1, tau)
        self._recurse(node.left, left_list, tau)
        self._recurse(node.right, right_list, tau)


def batch_mondrian_root_split(
    d_list: list,
    seed: int,
    lifetime: float = float("inf"),
    min_samples_split: int = 5,
):
    """Return ``(feature, cut, tau)`` of the root of a batch tree."""
    rng = random.Random(seed)
    tree = _BatchMondrianTree(rng, lifetime, min_samples_split)
    root = tree.sample(d_list)
    if root.feature is None or root.threshold is None:
        return "", 0.0, 0.0
    return root.feature, float(root.threshold), float(root.tau)


class _MondrianBase:
    """Shared machinery for the two Mondrian forest roles."""

    def __init__(
        self,
        n_trees: int,
        lifetime: float,
        max_nodes: int | None,
        seed: int,
        min_samples_split: int,
        task: str,
    ) -> None:
        if n_trees < 1:
            raise ValueError(f"n_trees must be >= 1, got {n_trees}")
        if lifetime <= 0:
            raise ValueError(f"lifetime must be > 0, got {lifetime}")
        if max_nodes is not None and max_nodes < 1:
            raise ValueError(f"max_nodes must be >= 1, got {max_nodes}")
        if min_samples_split < 2:
            raise ValueError(f"min_samples_split must be >= 2, got {min_samples_split}")
        if task not in ("regression", "classification"):
            raise ValueError(
                f"task must be 'regression' or 'classification', got {task!r}"
            )
        self.n_trees = n_trees
        self.lifetime = lifetime
        self.max_nodes = max_nodes
        self.seed = seed
        self.min_samples_split = min_samples_split
        self.task = task
        self.n_seen_ = 0
        self.n_missing_ = 0
        self.trees_: list = []

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    def _ensure_trees(self) -> None:
        if self.trees_:
            return
        for i in range(self.n_trees):
            rng = random.Random(self.seed + i)
            self.trees_.append(
                _MondrianTree(
                    rng=rng,
                    lifetime=self.lifetime,
                    max_nodes=self.max_nodes,
                    min_samples_split=self.min_samples_split,
                    task=self.task,
                )
            )

    def _valid(self, d: dict) -> bool:
        if not d:
            return False
        for v in d.values():
            if v is None:
                return False
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return False
            if not math.isfinite(fv):
                return False
        return True

    def _forest_predict(self, d: dict):
        if not self.trees_:
            return 0.0, 0.0, {}
        if self.task == "classification":
            acc: dict = {}
            for t in self.trees_:
                _, _, c = t.predict(d)
                for k, v in c.items():
                    acc[k] = acc.get(k, 0.0) + v
            if acc:
                tot = sum(acc.values())
                acc = {k: v / tot for k, v in acc.items()}
            return 0.0, 0.0, acc
        means = []
        vars_ = []
        for t in self.trees_:
            m, v, _ = t.predict(d)
            means.append(m)
            vars_.append(v)
        mu = sum(means) / len(means)
        var = sum(v + (m - mu) ** 2 for m, v in zip(means, vars_)) / len(means)
        return mu, var, {}


class MondrianForestRegressor(Regressor, _MondrianBase):
    """Online Mondrian forest for regression.

    Prediction is the mean of the per-tree Gaussian posteriors; the
    predictive variance is the mixture variance of eq. 3 of
    Lakshminarayanan et al. 2016, so it grows as the query point moves
    away from the training data. Memory grows with the number of nodes
    in the forest; ``max_nodes`` (default ``10000``) bounds each tree.

    ``budget_s`` = 2e-2 s: p99 of ``learn_one`` measured at 1.3e-3 s on a
    2000-sample, two-feature stream on a desktop CPU (default
    parameters); the budget leaves room for slower machines.
    """

    budget_s = 2e-2
    memory_class = "O(window)"

    def __init__(
        self,
        n_trees: int = 10,
        lifetime: float = math.inf,
        max_nodes: int | None = 10000,
        seed: int = 0,
        min_samples_split: int = 5,
    ) -> None:
        Regressor.__init__(self)
        _MondrianBase.__init__(
            self,
            n_trees=n_trees,
            lifetime=lifetime,
            max_nodes=max_nodes,
            seed=seed,
            min_samples_split=min_samples_split,
            task="regression",
        )

    def learn_one(
        self, x: Any, y: Any, t: float | None = None
    ) -> "MondrianForestRegressor":
        """Update every tree with one sample."""
        self._time_step(t)
        d = _as_dict(x)
        if not self._valid(d) or y is None:
            self.n_missing_ += 1
            return self
        try:
            yf = float(y)
        except (TypeError, ValueError):
            self.n_missing_ += 1
            return self
        if not math.isfinite(yf):
            self.n_missing_ += 1
            return self
        self._ensure_trees()
        for tree in self.trees_:
            tree.learn_one(d, yf)
        self.n_seen_ += 1
        return self

    def predict_one(
        self,
        x: Any,
        t: float | None = None,
        return_std: bool = False,
        return_estimate: bool = False,
    ) -> Any:
        """Return the predictive mean, and the variance on request."""
        d = _as_dict(x)
        if not self._valid(d):
            self.n_missing_ += 1
            if return_estimate:
                return Estimate(mean=0.0, var=0.0)
            if return_std:
                return 0.0, 0.0
            return 0.0
        mu, var, _ = self._forest_predict(d)
        if return_estimate:
            return Estimate(mean=mu, var=var)
        if return_std:
            return mu, math.sqrt(var)
        return mu


class MondrianForestClassifier(Classifier, _MondrianBase):
    """Online Mondrian forest for classification.

    Each tree predicts a smoothed distribution at the leaf that
    contains the query point: counts along the path are weighted by
    the probability of branching off just before each node. The forest
    averages the per-tree distributions. Memory grows with the number
    of nodes; ``max_nodes`` (default ``10000``) bounds each tree.

    ``budget_s`` = 2e-2 s: p99 of ``learn_one`` measured at 1.6e-3 s on a
    2000-sample, two-feature stream on a desktop CPU (default
    parameters); the budget leaves room for slower machines.
    """

    budget_s = 2e-2
    memory_class = "O(window)"

    def __init__(
        self,
        n_trees: int = 10,
        lifetime: float = math.inf,
        max_nodes: int | None = 10000,
        seed: int = 0,
        min_samples_split: int = 5,
    ) -> None:
        Classifier.__init__(self)
        _MondrianBase.__init__(
            self,
            n_trees=n_trees,
            lifetime=lifetime,
            max_nodes=max_nodes,
            seed=seed,
            min_samples_split=min_samples_split,
            task="classification",
        )

    def learn_one(
        self, x: Any, y: Any, t: float | None = None
    ) -> "MondrianForestClassifier":
        """Update every tree with one sample."""
        self._time_step(t)
        d = _as_dict(x)
        if not self._valid(d) or y is None:
            self.n_missing_ += 1
            return self
        self._ensure_trees()
        for tree in self.trees_:
            tree.learn_one(d, y)
        self.n_seen_ += 1
        return self

    def predict_proba_one(self, x: Any, t: float | None = None) -> dict:
        """Return the class probabilities from the forest."""
        d = _as_dict(x)
        if not self._valid(d):
            self.n_missing_ += 1
            return {}
        _, _, counts = self._forest_predict(d)
        if not counts:
            return {}
        tot = sum(counts.values())
        if tot <= 0.0:
            return {}
        return {k: v / tot for k, v in counts.items()}

    def predict_one(self, x: Any, t: float | None = None) -> Any:
        """Return the most likely class."""
        proba = self.predict_proba_one(x, t=t)
        if not proba:
            return None
        return max(proba.items(), key=lambda kv: kv[1])[0]
