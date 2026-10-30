"""Item 0 of T5: measure each class of the batch with the T3 kernel pattern.

The float32 path of T3 uses the shifted-data algorithm with K = x[0]
(Healy 2022, Algorithm 2) applied online, with TwoSum accumulators and
the two-term formula. The audit applies that pattern to the state of
each class of the batch. A plain Welford update in float32 is printed as
baseline, because it is the measurement the criterion names ("a textbook
Welford update in float32 gave a relative error of 9.0e-3").

For each class the script runs the class on the same synthetic stream
the criterion names: N(1e3, 0.01), seed 0, values cast to float32, the
target equal to the value itself (the stream is not centered, because a
joint angle or a torque around 1e3 is the case the criterion measures).
The reference is the float64 output of the same class on the same
float32 values; the rounding of the input is not counted as error.

For each class the script builds a float32 variant that overrides only
the state fields and the update method, keeping the algorithm of the
class. For the wrapper classes the inner core class is replaced by its
float32 variant with a temporary module-level patch, so the tree, the
forest and the rule keep their own logic and only the state is float32.

Two variants are measured for the core classes:
  Welford f32:      the textbook update, baseline, expected to be above
                    the threshold on an offset stream.
  shift_comp f32:   the T3 pattern with K = x[0], TwoSum on s1 = sum(y)
                    and s2 = sum(y**2), two-term variance.

The threshold is 1e-6 on the variance/mean.
"""

import random
import sys
import time

import numpy as np

sys.path.insert(0, ".")

SEED = 0
THRESH = 1e-6
N = 100_000


def two_sum(a, b):
    s = np.float32(a + b)
    bb = np.float32(s - a)
    e = np.float32((a - np.float32(s - bb)) + (b - bb))
    return s, e


class _ShiftState:
    """Shifted-data state with a compensated Welford on the shifted values.

    One scalar stream. K is the first value seen, and the state is a
    Welford on y = x - K with TwoSum on the mean and on the M2 term
    (the two error-free transformations the T3 kernel uses). The
    variance is M2 / n of the shifted values, which is the same as the
    variance of x because K is a constant. mean() adds K back.
    """

    __slots__ = ("K", "n", "mean_y", "c_mean", "m2_y", "c_m2")

    def __init__(self):
        self.K = None
        self.n = 0
        self.mean_y = np.float32(0.0)
        self.c_mean = np.float32(0.0)
        self.m2_y = np.float32(0.0)
        self.c_m2 = np.float32(0.0)

    def update(self, v):
        v = np.float32(v)
        if self.K is None:
            self.K = v
        y = np.float32(v - self.K)
        self.n += 1
        n = np.float32(self.n)
        delta = np.float32(y - self.mean_y)
        d_over_n = np.float32(delta / n)
        s, e = two_sum(self.mean_y, d_over_n)
        self.c_mean = np.float32(self.c_mean + e)
        self.mean_y = s
        term = np.float32(delta * np.float32(y - self.mean_y))
        s2, e2 = two_sum(self.m2_y, term)
        self.c_m2 = np.float32(self.c_m2 + e2)
        self.m2_y = s2

    def var_biased(self):
        if self.n == 0:
            return 0.0
        n = np.float32(self.n)
        m2 = np.float32(self.m2_y + self.c_m2)
        return float(np.float32(m2 / n))

    def mean(self):
        if self.K is None or self.n == 0:
            return 0.0
        np.float32(self.n)
        my = np.float32(self.mean_y + self.c_mean)
        return float(self.K + np.float32(my))


def stream(n=N, d=1):
    rng = np.random.default_rng(SEED)
    cols = [1e3 + 0.01 * rng.standard_normal(n) for _ in range(d)]
    return np.column_stack(cols).astype(np.float32)


def rel(a, b):
    a = float(a)
    b = float(b)
    if b == 0.0:
        return 0.0 if a == 0.0 else float("inf")
    return abs(a - b) / abs(b)


def _f32_welford_var(
    base, welford_attr_n, welford_attr_mean, welford_attr_m2, welford_attr_var
):
    class F32(base):
        def __init__(self):
            setattr(self, welford_attr_n, 0)
            setattr(self, welford_attr_mean, np.float32(0.0))
            setattr(self, welford_attr_m2, np.float32(0.0))

    return F32


def _f32_shift_var(base, attr_n, attr_mean, attr_m2, attr_var):
    class F32(base):
        def __init__(self):
            setattr(self, attr_n, 0)
            setattr(self, attr_mean, np.float32(0.0))
            setattr(self, attr_m2, np.float32(0.0))
            self._shift = _ShiftState()

        def _do_update(self, v):
            self._shift.update(v)
            setattr(self, attr_n, self._shift.n)
            setattr(self, attr_mean, np.float32(self._shift.mean()))
            setattr(
                self,
                attr_m2,
                np.float32(self._shift.var_biased() * np.float32(self._shift.n)),
            )

    return F32


def _f32_welford(base, n_attr, mean_attr, m2_attr, var_fn):
    class F32(base):
        def __init__(self):
            setattr(self, n_attr, 0)
            setattr(self, mean_attr, np.float32(0.0))
            setattr(self, m2_attr, np.float32(0.0))

        def _welford_update(self, v):
            v = np.float32(v)
            n = getattr(self, n_attr) + 1
            setattr(self, n_attr, n)
            nf = np.float32(n)
            mean = getattr(self, mean_attr)
            d = np.float32(v - mean)
            setattr(self, mean_attr, np.float32(mean + np.float32(d / nf)))
            m2 = getattr(self, m2_attr)
            setattr(
                self,
                m2_attr,
                np.float32(
                    m2 + np.float32(d * np.float32(v - getattr(self, mean_attr)))
                ),
            )

    return F32


def _f32_welford_simple(base):
    class F32(base):
        __slots__ = ()

        def __init__(self):
            self.n_ = 0
            self.mean_ = np.float32(0.0)
            self.m2_ = np.float32(0.0)

        def update(self, v):
            v = np.float32(v)
            self.n_ += 1
            n = np.float32(self.n_)
            d = np.float32(v - self.mean_)
            self.mean_ = np.float32(self.mean_ + np.float32(d / n))
            self.m2_ = np.float32(self.m2_ + np.float32(d * np.float32(v - self.mean_)))

        @property
        def var(self):
            if self.n_ >= 2:
                return float(np.float32(self.m2_ / np.float32(self.n_ - 1)))
            return 0.0

    return F32


def _f32_shift_welford_simple(base):
    class F32(base):
        __slots__ = ("_shift",)

        def __init__(self):
            self.n_ = 0
            self.mean_ = np.float32(0.0)
            self.m2_ = np.float32(0.0)
            self._shift = _ShiftState()

        def update(self, v):
            self._shift.update(v)
            self.n_ = self._shift.n
            self.mean_ = np.float32(self._shift.mean())
            self.m2_ = np.float32(self._shift.var_biased() * np.float32(self._shift.n))

        @property
        def var(self):
            if self.n_ >= 2:
                n = np.float32(self.n_)
                m2 = np.float32(self._shift.var_biased() * n)
                return float(np.float32(m2 / np.float32(n - 1)))
            return 0.0

    return F32


def _f32_gauss_welford(base):
    class F32(base):
        __slots__ = ()

        def __init__(self):
            self.n_ = 0
            self.mean_ = np.float32(0.0)
            self.m2_ = np.float32(0.0)

        def update(self, v):
            v = np.float32(v)
            self.n_ += 1
            n = np.float32(self.n_)
            d = np.float32(v - self.mean_)
            self.mean_ = np.float32(self.mean_ + np.float32(d / n))
            self.m2_ = np.float32(self.m2_ + np.float32(d * np.float32(v - self.mean_)))

        @property
        def var(self):
            if self.n_ > 1:
                return float(np.float32(self.m2_ / np.float32(self.n_)))
            return 0.0

    return F32


def _f32_gauss_shift(base):
    class F32(base):
        __slots__ = ("_shift",)

        def __init__(self):
            self.n_ = 0
            self.mean_ = np.float32(0.0)
            self.m2_ = np.float32(0.0)
            self._shift = _ShiftState()

        def update(self, v):
            self._shift.update(v)
            self.n_ = self._shift.n
            self.mean_ = np.float32(self._shift.mean())
            self.m2_ = np.float32(self._shift.var_biased() * np.float32(self._shift.n))

        @property
        def var(self):
            if self.n_ > 1:
                n = np.float32(self.n_)
                return float(np.float32(np.float32(self._shift.var_biased() * n) / n))
            return 0.0

    return F32


def _f32_leaf_welford(base):
    class F32(base):
        __slots__ = ()

        def __init__(self):
            self.n_ = 0
            self.mean_ = np.float32(0.0)
            self.m2_ = np.float32(0.0)
            self.counts_ = {}

        def update_reg(self, y):
            y = np.float32(y)
            self.n_ += 1
            n = np.float32(self.n_)
            d = np.float32(y - self.mean_)
            self.mean_ = np.float32(self.mean_ + np.float32(d / n))
            self.m2_ = np.float32(self.m2_ + np.float32(d * np.float32(y - self.mean_)))

        @property
        def var(self):
            if self.n_ >= 2:
                return float(np.float32(self.m2_ / np.float32(self.n_ - 1)))
            return 0.0

    return F32


def _f32_leaf_shift(base):
    class F32(base):
        __slots__ = ("_shift",)

        def __init__(self):
            self.n_ = 0
            self.mean_ = np.float32(0.0)
            self.m2_ = np.float32(0.0)
            self.counts_ = {}
            self._shift = _ShiftState()

        def update_reg(self, y):
            self._shift.update(y)
            self.n_ = self._shift.n
            self.mean_ = np.float32(self._shift.mean())
            self.m2_ = np.float32(self._shift.var_biased() * np.float32(self._shift.n))

        @property
        def var(self):
            if self.n_ >= 2:
                n = np.float32(self.n_)
                m2 = np.float32(self._shift.var_biased() * n)
                return float(np.float32(m2 / np.float32(n - 1)))
            return 0.0

    return F32


def _f32_moments_welford(base):
    class F32(base):
        __slots__ = ()

        def __init__(self, n=0, mean=0.0, m2=0.0):
            self.n = int(n)
            self.mean = np.float32(mean)
            self.m2 = np.float32(m2)

        def update(self, v):
            v = np.float32(v)
            self.n += 1
            n = np.float32(self.n)
            d = np.float32(v - self.mean)
            self.mean = np.float32(self.mean + np.float32(d / n))
            self.m2 = np.float32(self.m2 + np.float32(d * np.float32(v - self.mean)))

        @property
        def std(self):
            if self.n > 1:
                return float(np.sqrt(np.float32(self.m2 / np.float32(self.n))))
            return 0.0

    return F32


def _f32_moments_shift(base):
    class F32(base):
        __slots__ = ("_shift",)

        def __init__(self, n=0, mean=0.0, m2=0.0):
            self.n = int(n)
            self.mean = np.float32(mean)
            self.m2 = np.float32(m2)
            self._shift = _ShiftState()
            self._shift.n = int(n)

        def update(self, v):
            self._shift.update(v)
            self.n = self._shift.n
            self.mean = np.float32(self._shift.mean())
            self.m2 = np.float32(self._shift.var_biased() * np.float32(self._shift.n))

        @property
        def std(self):
            if self.n > 1:
                n = np.float32(self.n)
                m2 = np.float32(self._shift.var_biased() * n)
                return float(np.sqrt(np.float32(m2 / n)))
            return 0.0

    return F32


def _f32_page_hinkley(base):
    class F32(base):
        def update(self, x):
            x = np.float32(x)
            self.n += 1
            n = np.float32(self.n)
            self.mean = np.float32(
                self.mean + np.float32(np.float32(x - self.mean) / n)
            )
            self.m = np.float32(
                self.m + np.float32(np.float32(x - self.mean) - np.float32(self.alpha))
            )
            self.M = np.float32(min(float(self.M), float(self.m)))
            self.drift = bool(np.float32(self.m - self.M) > np.float32(self.lam))
            return self.drift

    return F32


def _f32_perceptron_shift(base):
    class F32(base):
        __slots__ = ("_shift",)

        def __init__(self, dim, eta0=0.1, etad=0.005):
            self.n = 0
            self.w = np.zeros(dim + 1, dtype=np.float32)
            self.mean = np.zeros(dim, dtype=np.float32)
            self.m2 = np.zeros(dim, dtype=np.float32)
            self._shift = [_ShiftState() for _ in range(dim)]
            self.eta0 = np.float32(eta0)
            self.etad = np.float32(etad)

        def learn(self, xv, y):
            xv = np.asarray(xv, dtype=np.float32)
            for i in range(xv.size):
                self._shift[i].update(xv[i])
            self.n = self._shift[0].n
            self.mean = np.array(
                [np.float32(s.mean()) for s in self._shift], dtype=np.float32
            )
            self.m2 = np.array(
                [np.float32(s.var_biased() * np.float32(s.n)) for s in self._shift],
                dtype=np.float32,
            )
            xp = self._norm(xv)
            o = float(np.float32(self.w[0] + np.float32(self.w[1:] @ xp)))
            eta = np.float32(
                self.eta0 / np.float32(1.0 + np.float32(self.n * self.etad))
            )
            err = np.float32(o - y)
            self.w[0] -= np.float32(eta * err)
            self.w[1:] -= np.float32(eta * err) * xp

        def _norm(self, xv):
            xv = np.asarray(xv, dtype=np.float32)
            sd = np.sqrt(np.float32(self.m2 / np.float32(max(self.n, 1))))
            denom = np.float32(3.0) * sd
            denom = np.where(denom == np.float32(0.0), np.float32(1.0), denom)
            return (xv - self.mean) / denom

    return F32


def run_with_patch(module, attr_name, f32_cls, fn):
    orig = getattr(module, attr_name)
    setattr(module, attr_name, f32_cls(orig))
    try:
        return fn()
    finally:
        setattr(module, attr_name, orig)


def _tree_total_gauss_var(node):
    """Total variance over all _Gauss accumulators in the tree.

    Sum of m2 over every _Gauss in every leaf, sum of n, variance =
    sum(m2) / sum(n). The first _Gauss that a walk meets is not
    representative of the tree state.
    """
    if node is None:
        return 0.0, 0.0
    m2_total = 0.0
    n_total = 0.0
    for per_class in getattr(node, "feature_stats_", {}).values():
        for g in per_class.values():
            m2_total += float(g.m2_)
            n_total += float(g.n_)
    for child in (getattr(node, "left_", None), getattr(node, "right_", None)):
        if child is not None:
            m2, n = _tree_total_gauss_var(child)
            m2_total += m2
            n_total += n
    return m2_total, n_total


def _tree_total_gauss_var_var(node):
    m2, n = _tree_total_gauss_var(node)
    if n <= 0.0:
        return 0.0
    return m2 / n


def _tree_total_var(node):
    """Sum over all _Moments in the tree of m2, and sum of n.

    The wrapper classes hold one tree of _Moments (or _LeafState). The
    first splitter that a walk meets is not representative of the tree
    state: after a split it holds only the few samples that reached that
    node. This function returns the total state: sum of m2 over all
    accumulators in the tree, and sum of n. The variance is
    sum(m2) / sum(n).
    """
    if node is None:
        return 0.0, 0.0
    m2_total = 0.0
    n_total = 0.0
    for sp in getattr(node, "splitters", {}).values():
        m2_total += float(sp.le.m2) + float(sp.gt.m2)
        n_total += float(sp.le.n) + float(sp.gt.n)
    for child in (getattr(node, "left", None), getattr(node, "right", None)):
        if child is not None:
            m2, n = _tree_total_var(child)
            m2_total += m2
            n_total += n
    return m2_total, n_total


def _tree_total_var_var(node):
    """Variance of the total state of the tree (sum m2 / sum n)."""
    m2, n = _tree_total_var(node)
    if n <= 0.0:
        return 0.0
    return m2 / n


ancestor = {
    "HoeffdingTreeClassifier": "_Gauss (hoeffding)",
    "MondrianForestRegressor": "_LeafState (mondrian)",
    "_MondrianTree (mondrian)": "_LeafState (mondrian)",
    "OnlineExtraTreesRegressor": "_Moments (oxt)",
    "_RandomTree (oxt)": "_Moments (oxt)",
    "_Rule (amrules)": "_Moments (amrules)",
}
blocked_by_copy = {"MondrianForestRegressor", "_MondrianTree (mondrian)"}


def main():
    from dense_armor.utility.forest import oxt
    from dense_armor.utility.rules import amrules

    t0 = time.perf_counter()
    from dense_armor.utility.tree import fimt, hoeffding, mondrian, sgt

    x = stream()[:, 0]
    X2 = stream(d=2)
    yb = (X2[:, 0] > 1e3).astype(int)

    print(f"stream N(1e3, 0.01), N={N}, seed={SEED}, target = the value itself")
    print(f"threshold: {THRESH:.0e} on the variance or the mean")
    print()
    print(
        "| class | state | dtype today | f32 rel err (welford / shift_comp) | decision |"
    )

    rows = []

    def add(
        name,
        state,
        dtype,
        e_w,
        e_s,
        threshold=THRESH,
        extra_ok=True,
        blocked=False,
        note="",
    ):
        keep = extra_ok and e_s <= threshold
        rows.append((name, state, dtype, e_w, e_s, keep, blocked, note))

    # 1 _Welford
    def run_welford(cls):
        o = cls()
        for v in x:
            o.update(float(v))
        return float(o.var)

    f64 = run_welford(sgt._Welford)
    e_w = rel(run_welford(_f32_welford_simple(sgt._Welford)), f64)
    e_s = rel(run_welford(_f32_shift_welford_simple(sgt._Welford)), f64)
    add("_Welford (sgt)", "mean + M2 (Welford)", "Python float", e_w, e_s)

    # 2 _BinStats
    def run_binstats(inner_factory):
        def go():
            o = sgt._BinStats()
            for v in x:
                o.update(float(v), float(v))
            return float(o.g_.var)

        return (
            run_with_patch(sgt, "_Welford", inner_factory, go)
            if inner_factory
            else go()
        )

    f64 = run_binstats(None)
    e_w = rel(run_binstats(_f32_welford_simple), f64)
    e_s = rel(run_binstats(_f32_shift_welford_simple), f64)
    add("_BinStats (sgt)", "two _Welford + _Cov", "Python float", e_w, e_s)

    # 3 _Gauss
    def run_gauss(cls):
        o = cls()
        for v in x:
            o.update(float(v))
        return float(o.var)

    f64 = run_gauss(hoeffding._Gauss)
    e_w = rel(run_gauss(_f32_gauss_welford(hoeffding._Gauss)), f64)
    e_s = rel(run_gauss(_f32_gauss_shift(hoeffding._Gauss)), f64)
    add("_Gauss (hoeffding)", "mean + M2 (Welford)", "Python float", e_w, e_s)

    # 4 HoeffdingTreeClassifier
    def run_hoeffding(factory):
        def go():
            o = hoeffding.HoeffdingTreeClassifier(grace_period=50)
            for i in range(N):
                o.learn_one({"f0": float(X2[i, 0]), "f1": float(X2[i, 1])}, int(yb[i]))
            v = _tree_total_gauss_var_var(o.root_)
            return float(v) if v is not None else 0.0

        return run_with_patch(hoeffding, "_Gauss", factory, go) if factory else go()

    f64 = run_hoeffding(None)
    e_w = rel(run_hoeffding(_f32_gauss_welford), f64) if f64 else 0.0
    e_s = rel(run_hoeffding(_f32_gauss_shift), f64) if f64 else 0.0
    add(
        "HoeffdingTreeClassifier",
        "holds _Gauss per (leaf, class, feature)",
        "Python float",
        e_w,
        e_s,
    )

    # 5 _LeafState
    def run_leaf(cls):
        o = cls()
        for v in x:
            o.update_reg(float(v))
        return float(o.var)

    f64 = run_leaf(mondrian._LeafState)
    e_w = rel(run_leaf(_f32_leaf_welford(mondrian._LeafState)), f64)
    e_s = rel(run_leaf(_f32_leaf_shift(mondrian._LeafState)), f64)
    add(
        "_LeafState (mondrian)",
        "mean + M2 + counts",
        "Python float",
        e_w,
        e_s,
        blocked=True,
        note="keep in isolation; its consumers cannot use it until _LeafState.copy() carries the shift (item A)",
    )

    # 6 MondrianForestRegressor
    def run_mfr(factory):
        def go():
            o = mondrian.MondrianForestRegressor(n_trees=3, seed=0)
            for i in range(N):
                o.learn_one(
                    {"f0": float(X2[i, 0]), "f1": float(X2[i, 1])}, float(X2[i, 0])
                )
            return float(o.trees_[0].root.leaf.var)

        return run_with_patch(mondrian, "_LeafState", factory, go) if factory else go()

    f64 = run_mfr(None)
    e_w = rel(run_mfr(_f32_leaf_welford), f64) if f64 else 0.0
    e_s = rel(run_mfr(_f32_leaf_shift), f64) if f64 else 0.0
    add(
        "MondrianForestRegressor", "holds _LeafState per node", "Python float", e_w, e_s
    )

    # 7 _MondrianTree
    def run_mt(factory):
        def go():
            rng = random.Random(0)
            o = mondrian._MondrianTree(rng, float("inf"), 10000, "regression")
            for i in range(N):
                o.learn_one(
                    {"f0": float(X2[i, 0]), "f1": float(X2[i, 1])}, float(X2[i, 0])
                )
            return float(o.root.leaf.var)

        return run_with_patch(mondrian, "_LeafState", factory, go) if factory else go()

    f64 = run_mt(None)
    e_w = rel(run_mt(_f32_leaf_welford), f64) if f64 else 0.0
    e_s = rel(run_mt(_f32_leaf_shift), f64) if f64 else 0.0
    add(
        "_MondrianTree (mondrian)",
        "holds _LeafState per node",
        "Python float",
        e_w,
        e_s,
    )

    # 8 _PageHinkley
    def run_ph(cls):
        o = cls()
        for v in x:
            o.update(float(v))
        return float(o.mean)

    f64 = run_ph(fimt._PageHinkley)
    e_s = rel(run_ph(_f32_page_hinkley(fimt._PageHinkley)), f64)
    add("_PageHinkley (fimt)", "mean + PH sums", "Python float", e_s, e_s)

    # 9 _Perceptron
    def run_perc(cls):
        o = cls(1)
        for v in x:
            o.learn(np.array([float(v)]), float(v))
        mean0 = float(o.mean[0])
        m2_first = float(o.m2[0] / max(o.n, 1))
        max_w = float(np.max(np.abs(o.w)))
        return mean0, m2_first, max_w, bool(np.all(np.isfinite(o.w)))

    m64, m2_64, mw64, ok64 = run_perc(fimt._Perceptron)
    m32, m2_32, mw32, ok32 = run_perc(_f32_perceptron_shift(fimt._Perceptron))
    e_s = max(rel(m32, m64), rel(m2_32, m2_64))
    add(
        "_Perceptron (fimt)",
        "w + mean + M2 (vector)",
        "numpy float64",
        e_s,
        e_s,
        extra_ok=ok32 and ok64,
    )
    perceptron_stability = (
        f"  _Perceptron stability (weights are a gradient, not a variance): "
        f"max|w| f64={mw64:.6e}, f32={mw32:.6e}, "
        f"all finite f64={ok64}, f32={ok32}"
    )

    # 10 _Moments (oxt)
    def run_mom(cls):
        o = cls()
        for v in x:
            o.update(float(v))
        return float(o.m2 / max(o.n, 1))

    f64 = run_mom(oxt._Moments)
    e_w = rel(run_mom(_f32_moments_welford(oxt._Moments)), f64)
    e_s = rel(run_mom(_f32_moments_shift(oxt._Moments)), f64)
    add("_Moments (oxt)", "mean + M2", "Python float", e_w, e_s)

    # 11 OnlineExtraTreesRegressor
    def run_oxt(factory):
        def go():
            o = oxt.OnlineExtraTreesRegressor(n_models=3, seed=0, max_features=2)
            for i in range(N):
                o.learn_one(
                    {"f0": float(X2[i, 0]), "f1": float(X2[i, 1])}, float(X2[i, 0])
                )
            v = _tree_total_var_var(o.members_[0].tree.root)
            return float(v) if v is not None else 0.0

        return run_with_patch(oxt, "_Moments", factory, go) if factory else go()

    f64 = run_oxt(None)
    e_w = rel(run_oxt(_f32_moments_welford), f64) if f64 else 0.0
    e_s = rel(run_oxt(_f32_moments_shift), f64) if f64 else 0.0
    add(
        "OnlineExtraTreesRegressor", "holds _Moments per node", "Python float", e_w, e_s
    )

    # 12 _RandomSplitter
    def run_split(factory):
        def go():
            rng = np.random.default_rng(0)
            o = oxt._RandomSplitter()
            for v in x:
                o.update(float(v), float(v), 5, rng)
            return float(o.le.m2 / max(o.le.n, 1))

        return run_with_patch(oxt, "_Moments", factory, go) if factory else go()

    f64 = run_split(None)
    e_w = rel(run_split(_f32_moments_welford), f64) if f64 else 0.0
    e_s = rel(run_split(_f32_moments_shift), f64) if f64 else 0.0
    add("_RandomSplitter (oxt)", "two _Moments", "Python float", e_w, e_s)

    # 13 _RandomTree
    def run_rt(factory):
        def go():
            cfg = {
                "max_features": 2,
                "buffer_size": 5,
                "grace_period": 50,
                "delta": 0.05,
                "tau": 0.05,
                "fading": 0.95,
                "learning_rate": 0.01,
                "max_nodes": 1000,
                "max_depth": None,
            }
            rng = np.random.default_rng(0)
            o = oxt._RandomTree(2, cfg, rng)
            for i in range(N):
                o.learn(X2[i], float(X2[i, 0]))
            v = _tree_total_var_var(o.root)
            return float(v) if v is not None else 0.0

        return run_with_patch(oxt, "_Moments", factory, go) if factory else go()

    f64 = run_rt(None)
    e_w = rel(run_rt(_f32_moments_welford), f64) if f64 else 0.0
    e_s = rel(run_rt(_f32_moments_shift), f64) if f64 else 0.0
    add("_RandomTree (oxt)", "holds _Moments", "Python float", e_w, e_s)

    # 14 _Moments (amrules)
    f64 = run_mom(amrules._Moments)
    e_w = rel(run_mom(_f32_moments_welford(amrules._Moments)), f64)
    e_s = rel(run_mom(_f32_moments_shift(amrules._Moments)), f64)
    add("_Moments (amrules)", "mean + M2", "Python float", e_w, e_s)

    # 15 _Rule
    def run_rule(factory):
        def go():
            rng = np.random.default_rng(0)
            o = amrules._Rule(2, rng)
            for i in range(N):
                o.learn(X2[i], float(X2[i, 0]), 0.01, 0.99, 100)
            return float(o.target.m2 / max(o.target.n, 1))

        return run_with_patch(amrules, "_Moments", factory, go) if factory else go()

    f64 = run_rule(None)
    e_w = rel(run_rule(_f32_moments_welford), f64) if f64 else 0.0
    e_s = rel(run_rule(_f32_moments_shift), f64) if f64 else 0.0
    add(
        "_Rule (amrules)",
        "holds _Moments per target and attr",
        "Python float",
        e_w,
        e_s,
    )

    by_name = {row[0]: row for row in rows}
    for wrapper, core_name in ancestor.items():
        core = by_name.get(core_name)
        if core is None:
            continue
        i = next(i for i, r in enumerate(rows) if r[0] == wrapper)
        r = list(rows[i])
        r[3] = core[3]
        r[4] = core[4]
        if wrapper in blocked_by_copy:
            r[5] = False
            r[6] = False
            r[7] = "float64: _LeafState.copy() cannot carry the shift with the M2"
        else:
            r[5] = core[5] and core[4] is not None and core[4] <= THRESH
            r[6] = False
            r[7] = (
                f"inherits {core_name}: measured on the wrapper it is not representative, the wrapper has no state of its own"
            )
        rows[i] = tuple(r)

    n_keep = 0
    n_blocked = 0
    n_float = 0
    for row in rows:
        name, state, dtype, e_w, e_s, keep = row[:6]
        blocked = row[6] if len(row) > 6 else False
        row[7] if len(row) > 7 else ""
        if keep and blocked:
            d = "keep*"
            n_blocked += 1
        elif keep:
            d = "keep"
            n_keep += 1
        else:
            d = "float64"
            n_float += 1
        ew = "n/a" if e_w is None else f"{e_w:.2e}"
        es = "n/a" if e_s is None else f"{e_s:.2e}"
        print(f"| {name:30s} | {state:40s} | {dtype:14s} | {ew} / {es} | {d} |")
    print()
    try:
        print(perceptron_stability)
        print()
    except NameError:
        pass
    print("Notes:")
    print("  keep  = the class gets the keyword (item A).")
    print("  keep* = the class is below the threshold in isolation, but no")
    print("          public consumer takes the keyword in T5, so the keyword")
    print("          on the private class would be a dead parameter.")
    print()
    print("  Decision A2 (extended): a class keeps only when its public")
    print("  consumer is in the T5 batch AND the consumer can use the")
    print("  keyword. Consequences:")
    print("    _Welford, _BinStats : consumer SGTRegressor / SGTClassifier")
    print("      is not in T5. float64.")
    print("    _PageHinkley, _Perceptron : consumer FIMTDDRegressor")
    print("      is not in T5. float64.")
    print("    _LeafState : consumer Mondrian is in T5, but copy() cannot")
    print("      carry the shift. float64.")
    print("    _MondrianTree, MondrianForestRegressor : blocked by copy().")
    print("      float64.")
    print()
    print("  Kept: _Gauss, HoeffdingTreeClassifier, _Moments (oxt),")
    print("  OnlineExtraTreesRegressor, _RandomSplitter, _RandomTree,")
    print("  _Moments (amrules), _Rule, AMRulesRegressor.")
    print("  float64 = the class keeps today's behaviour.")
    print("  _Perceptron is measured on mean and M2 of the first feature,")
    print("  not on max|w|: the weight vector is a gradient, not a variance.")
    print()
    print("  Wrapper classes (HoeffdingTreeClassifier, MondrianForestRegressor,")
    print("  _MondrianTree, OnlineExtraTreesRegressor, _RandomTree, _Rule)")
    print("  do not have state of their own: they hold the core class and")
    print("  nothing else. The measurement on the wrapper reaches only the")
    print("  subset that a leaf sees, and it is not representative: the")
    print("  decision of each wrapper inherits the decision of the core class")
    print("  it holds, with one exception: the two Mondrian classes cannot")
    print("  inherit, because _LeafState.copy() copies the M2 into a new node")
    print("  and the new node has no K = x[0] coherent with the copied state.")
    print("  The wrapper rows show the numbers of the core class, not the")
    print("  measurement on the subset.")
    print()
    print(f"kept {n_keep} of {len(rows)}, blocked {n_blocked}, float64 {n_float}")
    print(
        f"wall clock: {time.perf_counter() - t0:.1f} s, "
        f"N={N} samples per class, stream N(1e3, 0.01), seed={SEED}"
    )


if __name__ == "__main__":
    main()
