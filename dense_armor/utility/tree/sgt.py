"""Stochastic gradient trees for regression and classification.

An incremental decision tree built from stochastic gradient
information. Each node keeps the sums of the first and second
derivatives of the loss with respect to the current prediction; a
candidate split is scored with the reduction of the quadratic Taylor
expansion of the loss (equation 7 of Gouk et al. 2019), the optimal
value of a new leaf is the Newton step of equation 13, and a t-test
in place of the Hoeffding bound decides whether to split (equation
16).

For regression the loss is the squared error, whose first and second
derivatives with respect to the prediction are ``g = y_hat - y`` and
``h = 1`` (equations 24-25). For binary classification the loss is the
logistic log loss with a single logistic output; the first derivative
is ``g = p - y`` and the second is ``h = p (1 - p)`` where
``p = sigmoid(f(x))``.

Args:
    delta: one minus the confidence of the t-test that decides a
        split.
    min_samples_split: minimum number of samples at a leaf before a
        split is attempted.
    grace_period: interval in samples between two split attempts on
        the same leaf.
    n_bins: number of equal-width bins per numeric feature.
    bin_samples: number of initial samples used to estimate the range
        of each feature.
    lambda_: L2 penalty on the leaf values.
    gamma: cost per new node.
    max_depth: maximum depth, or ``None``.
    max_nodes: maximum number of nodes. Default ``10000``.

Raises:
    ValueError: on a non-positive parameter.

Examples:
    >>> from dense_armor.utility.tree.sgt import SGTRegressor
    >>> s = SGTRegressor(min_samples_split=20, n_bins=16, bin_samples=50)
    >>> for i in range(2000):
    ...     x = (i % 50) / 49.0
    ...     _ = s.learn_one({"x": x}, y=x)
    >>> s.n_seen_ > 0 and s.n_nodes_ > 1
    True

References:
    Gouk, H., Pfahringer, B., Frank, E. (2019). Stochastic gradient
        trees. In ACML, sections 3.1-3.3.
"""

import math
from typing import Any

from dense_armor.roles import Classifier, Regressor


def _as_dict(x: Any) -> dict:
    if hasattr(x, "to_dict"):
        return dict(x.to_dict())
    return dict(x)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function."""
    tiny = 1e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function ``I_x(a, b)``.

    Uses the continued fraction for the incomplete beta of the
    Numerical Recipes, with the symmetry ``I_x(a, b) = 1 - I_{1-x}(b, a)``
    applied once, non-recursively.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _student_t_sf(t: float, df: int) -> float:
    """One-sided survival function of Student's t with ``df`` degrees."""
    if df <= 0:
        return 0.5
    x = df / (df + t * t)
    ib = _betainc(0.5 * df, 0.5, x)
    if t > 0:
        return 0.5 * ib
    return 1.0 - 0.5 * ib


class _Welford:
    """Running mean and variance for one value stream."""

    __slots__ = ("m2_", "mean_", "n_")

    def __init__(self) -> None:
        self.n_ = 0
        self.mean_ = 0.0
        self.m2_ = 0.0

    def update(self, v: float) -> None:
        self.n_ += 1
        d = v - self.mean_
        self.mean_ += d / self.n_
        self.m2_ += d * (v - self.mean_)

    @property
    def var(self) -> float:
        return self.m2_ / (self.n_ - 1) if self.n_ >= 2 else 0.0


class _Cov:
    """Running covariance between two streams."""

    __slots__ = ("c_", "mx_", "my_", "n_")

    def __init__(self) -> None:
        self.n_ = 0
        self.mx_ = 0.0
        self.my_ = 0.0
        self.c_ = 0.0

    def update(self, x: float, y: float) -> None:
        self.n_ += 1
        dx = x - self.mx_
        self.mx_ += dx / self.n_
        dy = y - self.my_
        self.my_ += dy / self.n_
        self.c_ += dx * (y - self.my_)

    @property
    def cov(self) -> float:
        return self.c_ / (self.n_ - 1) if self.n_ >= 2 else 0.0


class _BinStats:
    """Per-bin running moments of gradients and Hessians."""

    __slots__ = ("g_", "g_sum_", "gh_", "h_", "h_sum_", "n_")

    def __init__(self) -> None:
        self.g_ = _Welford()
        self.h_ = _Welford()
        self.gh_ = _Cov()
        self.n_ = 0
        self.g_sum_ = 0.0
        self.h_sum_ = 0.0

    def update(self, g: float, h: float) -> None:
        self.n_ += 1
        self.g_sum_ += g
        self.h_sum_ += h
        self.g_.update(g)
        self.h_.update(h)
        self.gh_.update(g, h)


class _Node:
    """One node of a stochastic gradient tree."""

    __slots__ = (
        "depth",
        "g_sum_",
        "h_sum_",
        "left_",
        "n_",
        "right_",
        "since_split_",
        "split_feature_",
        "split_threshold_",
        "stats_",
        "v_",
    )

    def __init__(self, depth: int) -> None:
        self.depth = depth
        self.g_sum_ = 0.0
        self.h_sum_ = 0.0
        self.v_ = 0.0
        self.n_ = 0
        self.since_split_ = 0
        self.split_feature_: str | None = None
        self.split_threshold_: float | None = None
        self.left_: _Node | None = None
        self.right_: _Node | None = None
        self.stats_: dict[str, dict[int, _BinStats]] = {}

    @property
    def is_leaf(self) -> bool:
        return self.split_feature_ is None


def _merge(stats: list[_BinStats]) -> tuple[int, float, float, float, float, float]:
    """Merge the moments of several bins into the moments of their union.

    Pairwise update of Chan, Golub and LeVeque, the merge that section
    3.3 takes from Bennett et al. (2009): for two groups of sizes ``a``
    and ``b`` with mean difference ``d``, the second moment of the union
    is ``M2_a + M2_b + d**2 * a * b / (a + b)``, and the co-moment of two
    streams adds ``d_g * d_h * a * b / (a + b)``.

    Returns:
        ``(n, mean_g, m2_g, mean_h, m2_h, c_gh)`` of the union.
    """
    n = 0
    mg = m2g = mh = m2h = c = 0.0
    for bs in stats:
        nb = bs.n_
        if nb == 0:
            continue
        tot = n + nb
        dg = bs.g_.mean_ - mg
        dh = bs.h_.mean_ - mh
        w = n * nb / tot
        m2g += bs.g_.m2_ + dg * dg * w
        m2h += bs.h_.m2_ + dh * dh * w
        c += bs.gh_.c_ + dg * dh * w
        mg += dg * nb / tot
        mh += dh * nb / tot
        n = tot
    return n, mg, m2g, mh, m2h, c


def _side_moments(stats: list[_BinStats], lambda_: float) -> tuple[int, float, float]:
    """Mean and variance of the loss change ``L_i`` on one side of a split.

    ``v = -G / (H + lambda)`` is the value of the new leaf on that side,
    computed from all its samples; ``L_i = v g_i + v**2 h_i / 2`` (eq. 17)
    has variance ``v**2 Var(G) + v**4 Var(H) / 4 + v**3 Cov(G, H)``
    (eq. 18), with the moments of the merged bins.

    Returns:
        ``(n, mean, variance)`` of ``L_i`` on that side.
    """
    n, mg, m2g, mh, m2h, c = _merge(stats)
    if n == 0:
        return 0, 0.0, 0.0
    g_sum = sum(bs.g_sum_ for bs in stats)
    h_sum = sum(bs.h_sum_ for bs in stats)
    denom = lambda_ + h_sum
    if denom <= 0.0:
        return n, 0.0, 0.0
    v = -g_sum / denom
    mean = v * mg + 0.5 * v * v * mh
    if n < 2:
        return n, mean, 0.0
    var = (v * v * m2g + 0.25 * v**4 * m2h + v**3 * c) / (n - 1)
    return n, mean, max(0.0, var)


class _SGTBase:
    """Shared machinery for :class:`SGTRegressor` and :class:`SGTClassifier`."""

    def __init__(
        self,
        delta: float,
        min_samples_split: int,
        grace_period: int,
        n_bins: int,
        bin_samples: int,
        lambda_: float,
        gamma: float,
        max_depth: int | None,
        max_nodes: int | None,
    ) -> None:
        if not 0.0 < delta < 1.0:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        if min_samples_split < 2:
            raise ValueError(f"min_samples_split must be >= 2, got {min_samples_split}")
        if grace_period < 1:
            raise ValueError(f"grace_period must be >= 1, got {grace_period}")
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")
        if bin_samples < 2:
            raise ValueError(f"bin_samples must be >= 2, got {bin_samples}")
        if lambda_ < 0.0:
            raise ValueError(f"lambda_ must be >= 0, got {lambda_}")
        if gamma < 0.0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if max_depth is not None and max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        if max_nodes is not None and max_nodes < 1:
            raise ValueError(f"max_nodes must be >= 1, got {max_nodes}")
        self.delta = delta
        self.min_samples_split = min_samples_split
        self.grace_period = grace_period
        self.n_bins = n_bins
        self.bin_samples = bin_samples
        self.lambda_ = lambda_
        self.gamma = gamma
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.root_: _Node | None = None
        self.n_nodes_ = 0
        self.n_seen_ = 0
        self.n_missing_ = 0
        self._ranges: dict[str, tuple[float, float]] = {}
        self._buf: list[dict] = []

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    def _new_leaf(self, depth: int) -> _Node:
        node = _Node(depth)
        self.n_nodes_ += 1
        return node

    def _update_range(self, d: dict) -> bool:
        self._buf.append(dict(d))
        if len(self._buf) < self.bin_samples:
            return False
        for f in self._buf[0]:
            vals = []
            for x in self._buf:
                if f not in x or x[f] is None:
                    continue
                try:
                    vals.append(float(x[f]))
                except (TypeError, ValueError):
                    continue
            if vals:
                self._ranges[f] = (min(vals), max(vals))
        self._buf.clear()
        return True

    def _bin(self, f: str, v: float) -> int:
        lo, hi = self._ranges.get(f, (0.0, 1.0))
        if hi - lo <= 0.0:
            return 0
        b = int(self.n_bins * (v - lo) / (hi - lo))
        b = max(b, 0)
        if b >= self.n_bins:
            b = self.n_bins - 1
        return b

    def _descend(self, node: _Node, d: dict) -> _Node | None:
        f = node.split_feature_
        s = node.split_threshold_
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
        return node.left_ if fv <= s else node.right_

    def _path(self, d: dict) -> list:
        if self.root_ is None:
            self.root_ = self._new_leaf(0)
        path: list = []
        cur: _Node | None = self.root_
        while cur is not None:
            path.append(cur)
            if cur.is_leaf:
                return path
            cur = self._descend(cur, d)
        return path

    def _can_grow(self, node: _Node) -> bool:
        depth_ok = self.max_depth is None or node.depth < self.max_depth
        nodes_ok = self.max_nodes is None or self.n_nodes_ + 2 <= self.max_nodes
        return depth_ok and nodes_ok

    def _grad_and_hess(self, f: float, y: float) -> tuple[float, float]:
        raise NotImplementedError

    def _leaf_value(self, node: _Node) -> float:
        denom = self.lambda_ + node.h_sum_
        if denom <= 0.0:
            return 0.0
        return -node.g_sum_ / denom

    def _predict_raw(self, d: dict) -> float:
        if self.root_ is None:
            return 0.0
        path = self._path(d)
        return path[-1].v_

    def _update_leaf(self, node: _Node, d: dict, y: float) -> None:
        pred = self._predict_raw(d)
        g, h = self._grad_and_hess(pred, y)
        node.n_ += 1
        node.since_split_ += 1
        node.g_sum_ += g
        node.h_sum_ += h
        for f, v in d.items():
            if f not in self._ranges or v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(fv):
                continue
            b = self._bin(f, fv)
            per_bin = node.stats_.setdefault(f, {})
            bs = per_bin.get(b)
            if bs is None:
                bs = _BinStats()
                per_bin[b] = bs
            bs.update(g, h)
        node.v_ = self._leaf_value(node)

    def _reduction(self, node: _Node, f: str, split_bin: int):
        bins = node.stats_.get(f)
        if not bins:
            return None
        left = [bs for b, bs in bins.items() if b < split_bin]
        right = [bs for b, bs in bins.items() if b >= split_bin]
        g_l = sum(bs.g_sum_ for bs in left)
        h_l = sum(bs.h_sum_ for bs in left)
        g_r = sum(bs.g_sum_ for bs in right)
        h_r = sum(bs.h_sum_ for bs in right)
        if h_l + self.lambda_ <= 0.0 or h_r + self.lambda_ <= 0.0:
            return None
        gain = (
            0.5 * g_l * g_l / (h_l + self.lambda_)
            + 0.5 * g_r * g_r / (h_r + self.lambda_)
            - 0.5 * (g_l + g_r) ** 2 / (h_l + h_r + self.lambda_)
            - self.gamma
        )
        n = max(1, node.n_)
        mean = gain / n
        n_l, m_l, v_l = _side_moments(left, self.lambda_)
        n_r, m_r, v_r = _side_moments(right, self.lambda_)
        n_lr = n_l + n_r
        if n_lr < 2:
            return mean, 0.0
        var = (
            max(n_l - 1, 0) * v_l
            + max(n_r - 1, 0) * v_r
            + n_l * n_r / n_lr * (m_l - m_r) ** 2
        ) / (n_lr - 1)
        return mean, var

    def _try_split(self, node: _Node) -> bool:
        if node.n_ < self.min_samples_split:
            return False
        if node.since_split_ < self.grace_period:
            return False
        if not self._can_grow(node):
            return False
        best_mean = 0.0
        best_var = 0.0
        best_f: str | None = None
        best_s: float | None = None
        for f, bins in node.stats_.items():
            if len(bins) < 2:
                continue
            lo, hi = self._ranges.get(f, (0.0, 1.0))
            if hi - lo <= 0.0:
                continue
            for k in range(1, self.n_bins):
                red = self._reduction(node, f, k)
                if red is None:
                    continue
                m, v = red
                if m > best_mean:
                    best_mean = m
                    best_var = v
                    best_f = f
                    best_s = lo + (hi - lo) * k / self.n_bins
        node.since_split_ = 0
        if best_f is None or best_s is None or best_mean <= 0.0:
            return False
        n = max(2, node.n_)
        s = math.sqrt(best_var) if best_var > 0.0 else 0.0
        if s <= 0.0:
            return False
        t = best_mean / (s / math.sqrt(n))
        p = _student_t_sf(t, n - 1)
        if p < self.delta:
            node.split_feature_ = best_f
            node.split_threshold_ = best_s
            node.left_ = self._new_leaf(node.depth + 1)
            node.right_ = self._new_leaf(node.depth + 1)
            return True
        return False

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

    def _explain_path(self, x: Any) -> list:
        if self.root_ is None:
            return []
        d = _as_dict(x)
        cur: _Node | None = self.root_
        out: list = []
        while cur is not None and not cur.is_leaf:
            f = cur.split_feature_
            s = cur.split_threshold_
            if f is None or s is None:
                break
            v = d.get(f)
            if v is None:
                break
            try:
                fv = float(v)
            except (TypeError, ValueError):
                break
            side = "left" if fv <= s else "right"
            out.append({"feature": f, "threshold": s, "side": side})
            cur = cur.left_ if side == "left" else cur.right_
        return out


class SGTRegressor(Regressor, _SGTBase):
    """Stochastic gradient tree for regression.

    Minimises the squared loss by taking one Newton step per update.
    The split is accepted when the one-sided p-value of the t-test of
    section 3.3 is below ``delta``.

    Memory grows with the number of nodes; ``max_nodes`` (default
    ``10000``) bounds the total, so ``memory_class`` is ``"O(window)"``.

    Args:
        delta: one minus the confidence of the t-test.
        min_samples_split: minimum samples at a leaf before a split.
        grace_period: interval in samples between two split attempts.
        n_bins: number of equal-width bins per numeric feature.
        bin_samples: initial samples used to estimate feature ranges.
        lambda_: L2 penalty on the leaf values.
        gamma: cost per new node.
        max_depth: maximum depth, or ``None``.
        max_nodes: maximum number of nodes. Default ``10000``.

    Raises:
        ValueError: on a non-positive parameter.

    ``budget_s`` = 1e-3 s: p99 of ``learn_one`` measured at 1.2e-5 s on a
    2000-sample, two-feature stream on a desktop CPU (default
    parameters); the budget leaves room for slower machines.
    """

    budget_s = 1e-3
    memory_class = "O(window)"

    def __init__(
        self,
        delta: float = 0.01,
        min_samples_split: int = 50,
        grace_period: int = 200,
        n_bins: int = 64,
        bin_samples: int = 1000,
        lambda_: float = 0.1,
        gamma: float = 1.0,
        max_depth: int | None = None,
        max_nodes: int | None = 10000,
    ) -> None:
        Regressor.__init__(self)
        _SGTBase.__init__(
            self,
            delta=delta,
            min_samples_split=min_samples_split,
            grace_period=grace_period,
            n_bins=n_bins,
            bin_samples=bin_samples,
            lambda_=lambda_,
            gamma=gamma,
            max_depth=max_depth,
            max_nodes=max_nodes,
        )

    def _grad_and_hess(self, f: float, y: float) -> tuple[float, float]:
        return f - y, 1.0

    def learn_one(self, x: Any, y: Any, t: float | None = None) -> "SGTRegressor":
        """Update the tree with one sample."""
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
        self.n_seen_ += 1
        if self.root_ is None:
            self.root_ = self._new_leaf(0)
        if len(self._ranges) < len(d) and not self._update_range(d):
            return self
        path = self._path(d)
        leaf = path[-1]
        self._update_leaf(leaf, d, yf)
        if leaf.is_leaf:
            self._try_split(leaf)
        return self

    def predict_one(
        self,
        x: Any,
        t: float | None = None,
        return_std: bool = False,
    ) -> Any:
        """Return the predicted value at the reached leaf."""
        d = _as_dict(x)
        v = self._predict_raw(d)
        if return_std:
            return v, 0.0
        return v

    def explain_one(self, x: Any, t: float | None = None) -> list:
        """Return the path from root to leaf."""
        return self._explain_path(x)


class SGTClassifier(Classifier, _SGTBase):
    """Stochastic gradient tree for binary classification.

    A single logistic output, ``p = sigmoid(f(x))``, is trained with
    the logistic log loss. The split is accepted when the one-sided
    p-value of the t-test of section 3.3 is below ``delta``.

    Memory grows with the number of nodes; ``max_nodes`` (default
    ``10000``) bounds the total, so ``memory_class`` is ``"O(window)"``.

    Args:
        delta: one minus the confidence of the t-test.
        min_samples_split: minimum samples at a leaf before a split.
        grace_period: interval in samples between two split attempts.
        n_bins: number of equal-width bins per numeric feature.
        bin_samples: initial samples used to estimate feature ranges.
        lambda_: L2 penalty on the leaf values.
        gamma: cost per new node.
        max_depth: maximum depth, or ``None``.
        max_nodes: maximum number of nodes. Default ``10000``.

    Raises:
        ValueError: on a non-positive parameter.

    ``budget_s`` = 1e-3 s: p99 of ``learn_one`` measured at 2.4e-5 s on a
    2000-sample, two-feature stream on a desktop CPU (default
    parameters); the budget leaves room for slower machines.
    """

    budget_s = 1e-3
    memory_class = "O(window)"

    def __init__(
        self,
        delta: float = 0.01,
        min_samples_split: int = 50,
        grace_period: int = 200,
        n_bins: int = 64,
        bin_samples: int = 1000,
        lambda_: float = 0.1,
        gamma: float = 1.0,
        max_depth: int | None = None,
        max_nodes: int | None = 10000,
    ) -> None:
        Classifier.__init__(self)
        _SGTBase.__init__(
            self,
            delta=delta,
            min_samples_split=min_samples_split,
            grace_period=grace_period,
            n_bins=n_bins,
            bin_samples=bin_samples,
            lambda_=lambda_,
            gamma=gamma,
            max_depth=max_depth,
            max_nodes=max_nodes,
        )
        self.classes_: set = set()

    def _sigmoid(self, z: float) -> float:
        if z >= 0.0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)

    def _grad_and_hess(self, f: float, y: float) -> tuple[float, float]:
        p = self._sigmoid(f)
        return p - y, p * (1.0 - p)

    def learn_one(self, x: Any, y: Any, t: float | None = None) -> "SGTClassifier":
        """Update the tree with one sample."""
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
        self.classes_.add(y)
        self.n_seen_ += 1
        if self.root_ is None:
            self.root_ = self._new_leaf(0)
        if len(self._ranges) < len(d) and not self._update_range(d):
            return self
        path = self._path(d)
        leaf = path[-1]
        self._update_leaf(leaf, d, yf)
        if leaf.is_leaf:
            self._try_split(leaf)
        return self

    def predict_proba_one(self, x: Any, t: float | None = None) -> dict:
        """Return the class-probability dict."""
        if not self.classes_:
            return {}
        d = _as_dict(x)
        p = self._sigmoid(self._predict_raw(d))
        classes = list(self.classes_)
        if len(classes) == 1:
            return {classes[0]: 1.0}
        pos = classes[-1]
        neg = classes[0]
        return {neg: 1.0 - p, pos: p}

    def predict_one(self, x: Any, t: float | None = None) -> Any:
        """Return the most likely class."""
        proba = self.predict_proba_one(x, t=t)
        if not proba:
            return None
        return max(proba.items(), key=lambda kv: kv[1])[0]

    def explain_one(self, x: Any, t: float | None = None) -> list:
        """Return the path from root to leaf."""
        return self._explain_path(x)
