"""Attribute observers, split criteria and split candidates of the Hoeffding trees.

An observer sees the values of one feature at one leaf, together with the
class (or the target), and answers one question: what is the best split of
this feature, and how good is it? The trees ask it when a leaf has seen
``grace_period`` more samples.

Classification (``HoeffdingTreeClassifier`` and the trees built on it):

- ``NumericObserver`` keeps a running Gaussian per class (it is a ``dict``
  ``{class: _Gauss}``, so the naive-Bayes leaf reads it as before) and, by
  ``splitter``, the candidates of one of four methods:

  ``"class_means"`` (default, the historical one)
      the midpoints between the means of the classes; the share of each
      class on each side is read from the normal CDF.
  ``"gaussian"``
      ``n_splits`` thresholds evenly spaced between the smallest and the
      largest value seen, with the class shares from the normal CDF, as the
      Gaussian numeric observer of MOA and of River.
  ``"ebst"``
      the exhaustive search: every distinct value seen is kept, with its class
      counts, and every boundary between two consecutive values is a
      candidate, evaluated on exact counts (the extended binary search tree
      of Ikonomovska, Gama and Dzeroski 2011, here with class counts instead
      of target sums). Memory grows with the number of distinct values.
  ``"quantile"``
      a quantization observer: the values fall in slots of a fixed width, a
      slot keeps its mean and its class counts, and the boundaries between
      occupied slots are the candidates. The width is a third of the
      standard deviation of the first 30 values, or ``quantile_radius`` (the
      quantizer of Mastelini and de Carvalho 2021, with the class counts of
      the classification case). Memory is bounded by the number of slots.

- ``NominalObserver`` keeps the class counts of every category; a candidate
  is "the value is this category" against all the others.

The criteria are the information gain on the entropy (``"entropy"``) and on
the Gini impurity (``"gini"``). The range of the merit, used by the
Hoeffding bound, is ``log2(K)`` for the entropy and ``1 - 1/K`` for the Gini
impurity, with ``K`` the number of classes.

Regression (``HoeffdingTreeRegressor``): ``TargetNumericObserver`` and
``TargetNominalObserver`` keep the count, the sum and the sum of squares of the
target, and the criterion is the reduction of the variance.
"""

import bisect
import math

from dense_armor.core.online.tree._stats import _Gauss, _phi

CRITERIA = ("entropy", "gini")
SPLITTERS = ("class_means", "gaussian", "ebst", "quantile")
TARGET_SPLITTERS = ("ebst", "quantile")
_WARMUP = 30


def entropy(counts: dict) -> float:
    total = sum(counts.values())
    if total <= 0.0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c > 0.0:
            p = c / total
            h -= p * math.log2(p)
    return h


def gini(counts: dict) -> float:
    total = sum(counts.values())
    if total <= 0.0:
        return 0.0
    return 1.0 - sum((c / total) ** 2 for c in counts.values())


def impurity(criterion: str, counts: dict) -> float:
    return entropy(counts) if criterion == "entropy" else gini(counts)


def merit_range(criterion: str, k: int) -> float:
    """Range of the merit of a split with ``k`` classes (for the Hoeffding bound)."""
    return math.log2(k) if criterion == "entropy" else 1.0 - 1.0 / k


def split_gain(criterion: str, parent: dict, n: float, left: dict, right: dict):
    """Impurity of the parent minus the weighted impurity of the two sides, or ``None``."""
    n_l = sum(left.values())
    n_r = sum(right.values())
    if n_l <= 0.0 or n_r <= 0.0:
        return None
    return (
        impurity(criterion, parent)
        - (n_l / n) * impurity(criterion, left)
        - (n_r / n) * impurity(criterion, right)
    )


def _totals(points: list) -> dict:
    total: dict = {}
    for _, counts in points:
        for k, c in counts.items():
            total[k] = total.get(k, 0) + c
    return total


def _scan(criterion: str, points: list):
    """Best boundary between consecutive points, on exact class counts.

    ``points`` is a list of ``(position, {class: count})`` sorted by position.
    Returns ``(gain, threshold)``, or ``(-1.0, None)``.
    """
    if len(points) < 2:
        return -1.0, None
    total = _totals(points)
    n_obs = float(sum(total.values()))
    parent_imp = impurity(criterion, total)
    left: dict = {}
    best_gain, best_thr = -1.0, None
    for i in range(len(points) - 1):
        pos, counts = points[i]
        for k, c in counts.items():
            left[k] = left.get(k, 0) + c
        right = {
            k: total[k] - left.get(k, 0) for k in total if total[k] - left.get(k, 0) > 0
        }
        n_l = sum(left.values())
        n_r = n_obs - n_l
        if n_l <= 0 or n_r <= 0:
            continue
        gain = (
            parent_imp
            - (n_l / n_obs) * impurity(criterion, left)
            - (n_r / n_obs) * impurity(criterion, right)
        )
        if gain > best_gain:
            best_gain = gain
            best_thr = 0.5 * (pos + points[i + 1][0])
    return best_gain, best_thr


def _exact_gain(criterion: str, points: list, threshold: float):
    """Gain of ``x <= threshold`` on exact class counts, or ``None``."""
    left: dict = {}
    right: dict = {}
    for pos, counts in points:
        side = left if pos <= threshold else right
        for k, c in counts.items():
            side[k] = side.get(k, 0) + c
    n_obs = float(sum(left.values()) + sum(right.values()))
    if n_obs <= 0.0:
        return None
    return split_gain(criterion, _totals(points), n_obs, left, right)


class NumericObserver(dict):
    """``{class: _Gauss}`` of one numeric feature at one leaf, and the state of its splitter."""

    kind = "num"

    def __init__(self, splitter="class_means", dtype=None, n_splits=10, radius=None):
        super().__init__()
        self.splitter = splitter
        self.dtype = dtype
        self.n_splits = n_splits
        self.lo = math.inf
        self.hi = -math.inf
        self._vals: list = []
        self._cnts: list = []
        self._radius = radius
        self._slots: dict = {}
        self._warm: list = []

    def update(self, v: float, y) -> None:
        g = self.get(y)
        if g is None:
            g = _Gauss(dtype=self.dtype)
            self[y] = g
        g.update(v)
        if v < self.lo:
            self.lo = v
        if v > self.hi:
            self.hi = v
        if self.splitter == "ebst":
            i = bisect.bisect_left(self._vals, v)
            if i < len(self._vals) and self._vals[i] == v:
                c = self._cnts[i]
                c[y] = c.get(y, 0) + 1
            else:
                self._vals.insert(i, v)
                self._cnts.insert(i, {y: 1})
        elif self.splitter == "quantile":
            self._quantile_update(v, y)

    def _quantile_update(self, v: float, y) -> None:
        if self._radius is None:
            self._warm.append((v, y))
            if len(self._warm) >= _WARMUP:
                xs = [a for a, _ in self._warm]
                mean = sum(xs) / len(xs)
                var = sum((a - mean) ** 2 for a in xs) / len(xs)
                self._radius = math.sqrt(var) / 3.0 if var > 0.0 else 1.0
                for a, b in self._warm:
                    self._slot_add(a, b)
                self._warm = []
            return
        self._slot_add(v, y)

    def _slot_add(self, v: float, y) -> None:
        key = math.floor(v / self._radius)
        slot = self._slots.get(key)
        if slot is None:
            slot = [0.0, 0, {}]
            self._slots[key] = slot
        slot[0] += v
        slot[1] += 1
        slot[2][y] = slot[2].get(y, 0) + 1

    def _points(self) -> list:
        if self.splitter == "ebst":
            return list(zip(self._vals, self._cnts))
        if self._radius is None:
            by_value: dict = {}
            for v, y in self._warm:
                c = by_value.setdefault(v, {})
                c[y] = c.get(y, 0) + 1
            return sorted(by_value.items())
        return [
            (self._slots[k][0] / self._slots[k][1], self._slots[k][2])
            for k in sorted(self._slots)
        ]

    def _cdf_counts(self, s: float, class_counts: dict):
        left: dict = {}
        right: dict = {}
        for k, g in self.items():
            if g.n_ <= 0:
                continue
            total_k = class_counts.get(k, 0)
            if g.var <= 0.0:
                if g.mean_ <= s:
                    left[k] = total_k
                else:
                    right[k] = total_k
                continue
            z = (s - g.mean_) / math.sqrt(g.var)
            p_left = _phi(z)
            left[k] = p_left * total_k
            right[k] = (1.0 - p_left) * total_k
        return left, right

    def _thresholds(self) -> list:
        if self.splitter == "gaussian":
            if not self.lo < self.hi:
                return []
            step = (self.hi - self.lo) / (self.n_splits + 1)
            return [self.lo + step * i for i in range(1, self.n_splits + 1)]
        means = sorted(g.mean_ for g in self.values() if g.n_ > 0)
        if len(means) < 2:
            return []
        return [0.5 * (means[i] + means[i + 1]) for i in range(len(means) - 1)]

    def best_candidate(self, criterion: str, class_counts: dict, n: float):
        """``(gain, threshold)`` of the best split of this feature, or ``(-1.0, None)``."""
        if self.splitter in ("ebst", "quantile"):
            return _scan(criterion, self._points())
        best_gain, best_thr = -1.0, None
        for s in self._thresholds():
            left, right = self._cdf_counts(s, class_counts)
            g = split_gain(criterion, class_counts, n, left, right)
            if g is not None and g > best_gain:
                best_gain, best_thr = g, s
        return best_gain, best_thr

    def test_gain(self, criterion: str, class_counts: dict, n: float, threshold: float):
        """Gain of the split ``x <= threshold`` on the data of this observer, or ``None``."""
        if self.splitter in ("ebst", "quantile"):
            return _exact_gain(criterion, self._points(), threshold)
        left, right = self._cdf_counts(threshold, class_counts)
        return split_gain(criterion, class_counts, n, left, right)


class NominalObserver(dict):
    """``{category: {class: count}}`` of one nominal feature at one leaf."""

    kind = "nom"

    def update(self, v, y) -> None:
        c = self.get(v)
        if c is None:
            c = {}
            self[v] = c
        c[y] = c.get(y, 0) + 1

    def _points(self) -> list:
        return list(self.items())

    def best_candidate(self, criterion: str, class_counts: dict, n: float):
        """``(gain, category)``: the best "value == category" test, or ``(-1.0, None)``."""
        if len(self) < 2:
            return -1.0, None
        pts = self._points()
        total = _totals(pts)
        n_obs = float(sum(total.values()))
        best_gain, best_cat = -1.0, None
        for cat, counts in pts:
            right = {
                k: total[k] - counts.get(k, 0)
                for k in total
                if total[k] - counts.get(k, 0) > 0
            }
            g = split_gain(criterion, total, n_obs, dict(counts), right)
            if g is not None and g > best_gain:
                best_gain, best_cat = g, cat
        return best_gain, best_cat

    def test_gain(self, criterion: str, class_counts: dict, n: float, category):
        counts = self.get(category)
        if not counts:
            return None
        total = _totals(self._points())
        n_obs = float(sum(total.values()))
        right = {
            k: total[k] - counts.get(k, 0)
            for k in total
            if total[k] - counts.get(k, 0) > 0
        }
        return split_gain(criterion, total, n_obs, dict(counts), right)

    def cond_proba(self, value, cls, class_n: float) -> float:
        """Laplace estimate of ``P(value | class)``, with one extra slot for a new category."""
        seen = self.get(value)
        n_vc = seen.get(cls, 0) if seen else 0
        return (n_vc + 1.0) / (class_n + len(self) + 1.0)


def _variance(n: float, s: float, s2: float) -> float:
    if n <= 0:
        return 0.0
    m = s / n
    return max(0.0, s2 / n - m * m)


def _target_scan(points: list):
    """Best variance reduction between consecutive points of ``[(pos, [n, sum, sum_sq])]``."""
    if len(points) < 2:
        return -1.0, None
    n = sum(p[1][0] for p in points)
    s = sum(p[1][1] for p in points)
    s2 = sum(p[1][2] for p in points)
    parent = _variance(n, s, s2)
    ln = ls = ls2 = 0.0
    best_gain, best_thr = -1.0, None
    for i in range(len(points) - 1):
        pos, (cn, cs, cs2) = points[i]
        ln += cn
        ls += cs
        ls2 += cs2
        rn = n - ln
        if ln <= 0 or rn <= 0:
            continue
        gain = (
            parent
            - (ln / n) * _variance(ln, ls, ls2)
            - (rn / n) * _variance(rn, s - ls, s2 - ls2)
        )
        if gain > best_gain:
            best_gain = gain
            best_thr = 0.5 * (pos + points[i + 1][0])
    return best_gain, best_thr


class TargetNumericObserver:
    """Target statistics of one numeric feature at one leaf, for the variance reduction.

    ``splitter`` is ``"ebst"`` (every distinct value, exhaustive, the E-BST of
    Ikonomovska et al. 2011) or ``"quantile"`` (slots of a fixed width, the
    quantizer of Mastelini and de Carvalho 2021).
    """

    kind = "num"

    def __init__(self, splitter="ebst", radius=None):
        self.splitter = splitter
        self._vals: list = []
        self._stats: list = []
        self._radius = radius
        self._slots: dict = {}
        self._warm: list = []

    def update(self, v: float, y: float) -> None:
        if self.splitter == "ebst":
            i = bisect.bisect_left(self._vals, v)
            if i < len(self._vals) and self._vals[i] == v:
                st = self._stats[i]
            else:
                st = [0, 0.0, 0.0]
                self._vals.insert(i, v)
                self._stats.insert(i, st)
            st[0] += 1
            st[1] += y
            st[2] += y * y
            return
        if self._radius is None:
            self._warm.append((v, y))
            if len(self._warm) >= _WARMUP:
                xs = [a for a, _ in self._warm]
                mean = sum(xs) / len(xs)
                var = sum((a - mean) ** 2 for a in xs) / len(xs)
                self._radius = math.sqrt(var) / 3.0 if var > 0.0 else 1.0
                for a, b in self._warm:
                    self._slot_add(a, b)
                self._warm = []
            return
        self._slot_add(v, y)

    def _slot_add(self, v: float, y: float) -> None:
        key = math.floor(v / self._radius)
        slot = self._slots.get(key)
        if slot is None:
            slot = [0.0, 0, 0.0, 0.0]
            self._slots[key] = slot
        slot[0] += v
        slot[1] += 1
        slot[2] += y
        slot[3] += y * y

    def _points(self) -> list:
        if self.splitter == "ebst":
            return list(zip(self._vals, self._stats))
        if self._radius is None:
            by_value: dict = {}
            for v, y in self._warm:
                st = by_value.setdefault(v, [0, 0.0, 0.0])
                st[0] += 1
                st[1] += y
                st[2] += y * y
            return sorted(by_value.items())
        return [
            (self._slots[k][0] / self._slots[k][1], self._slots[k][1:])
            for k in sorted(self._slots)
        ]

    def best_candidate(self):
        """``(variance reduction, threshold)`` of the best split, or ``(-1.0, None)``."""
        return _target_scan(self._points())

    def test_gain(self, threshold: float):
        pts = self._points()
        n = sum(p[1][0] for p in pts)
        if n <= 0:
            return None
        s = sum(p[1][1] for p in pts)
        s2 = sum(p[1][2] for p in pts)
        ln = ls = ls2 = 0.0
        for pos, (cn, cs, cs2) in pts:
            if pos <= threshold:
                ln += cn
                ls += cs
                ls2 += cs2
        rn = n - ln
        if ln <= 0 or rn <= 0:
            return None
        return (
            _variance(n, s, s2)
            - (ln / n) * _variance(ln, ls, ls2)
            - (rn / n) * _variance(rn, s - ls, s2 - ls2)
        )


class TargetNominalObserver(dict):
    """``{category: [n, sum, sum_sq]}`` of the target, for one nominal feature at one leaf."""

    kind = "nom"

    def update(self, v, y: float) -> None:
        st = self.get(v)
        if st is None:
            st = [0, 0.0, 0.0]
            self[v] = st
        st[0] += 1
        st[1] += y
        st[2] += y * y

    def best_candidate(self):
        """``(variance reduction, category)`` of the best "value == category" test."""
        if len(self) < 2:
            return -1.0, None
        n = sum(st[0] for st in self.values())
        s = sum(st[1] for st in self.values())
        s2 = sum(st[2] for st in self.values())
        parent = _variance(n, s, s2)
        best_gain, best_cat = -1.0, None
        for cat, (cn, cs, cs2) in self.items():
            rn = n - cn
            if cn <= 0 or rn <= 0:
                continue
            gain = (
                parent
                - (cn / n) * _variance(cn, cs, cs2)
                - (rn / n) * _variance(rn, s - cs, s2 - cs2)
            )
            if gain > best_gain:
                best_gain, best_cat = gain, cat
        return best_gain, best_cat

    def test_gain(self, category):
        st = self.get(category)
        if not st:
            return None
        n = sum(x[0] for x in self.values())
        s = sum(x[1] for x in self.values())
        s2 = sum(x[2] for x in self.values())
        cn, cs, cs2 = st
        rn = n - cn
        if cn <= 0 or rn <= 0:
            return None
        return (
            _variance(n, s, s2)
            - (cn / n) * _variance(cn, cs, cs2)
            - (rn / n) * _variance(rn, s - cs, s2 - cs2)
        )
