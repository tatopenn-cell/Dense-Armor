"""Adaptive Model Rules (AMRules) for regression on data streams.

A rule is an "if" part, a conjunction of conditions on the features
(``x2 <= 1.5 and x0 > 3``), and a "then" part, a model that predicts the
target for the samples the rule covers. The rule set starts with one
default rule that covers everything, and learns one sample at a time
(Duarte, Gama and Bifet 2016).

The pieces, with the place where the paper states them (page numbers are
those of the PDF, which is the journal article, ACM TKDD 10(3), article
30):

- Learning a rule set (Algorithm 1, page 6). Every sample updates the
  rules that cover it, or the first of them when the set is ordered; a
  sample that no rule covers updates the default rule. After ``n_min``
  samples a rule tries to expand, and the default rule too: if it
  expands, the expanded rule joins the set and a new default rule
  starts.
- Expansion (Algorithm 2 and Section 3.2, pages 6-7). Every candidate
  split ``attribute <= value`` is scored by the variance ratio

      VR = 1 - |E_L|/|E| * var(E_L)/var(E) - |E_R|/|E| * var(E_R)/var(E),

  where ``E`` is the set of samples seen by the rule since its last
  expansion and ``E_L``, ``E_R`` the parts on each side of the split.
  The best attribute is chosen when the ratio ``r`` of the second best
  ``VR`` to the best one satisfies ``r + eps < 1``, with the Hoeffding
  bound ``eps = sqrt(R^2 ln(1/delta) / (2 n))``, ``R = 1``; when ``eps``
  falls below ``tau`` the best attribute is chosen anyway.
- Prediction (Section 3.3, page 8). A rule predicts with the mean of its
  target, with a linear model, or with the one of the two whose mean
  absolute error, weighted by a fading factor ``alpha``, is lower. The
  linear model works on standardised attributes and target, learns by
  the delta rule, and its output is denormalised. An ordered set uses the
  first covering rule; an unordered set averages the covering rules; a
  sample with no covering rule is predicted by the default rule.
- Change detection (Section 3.4, pages 8-9). The Page-Hinkley test
  watches the absolute error of every rule; a rule whose error rises
  significantly is removed.
- Anomalies (Section 3.5, pages 9-10). Optionally, a sample is ignored
  by a rule when ``mean_j log(P_j / (1 - P_j))`` is below ``t``, with
  ``P_j`` a Cantelli-type bound on the probability of the value of
  feature ``j`` given the rule.

Differences from the paper, and choices it does not fix, stated as my
choice:

- Algorithm 2 prints the test ``VR(Xa) - VR(Xb) > eps``, while the text
  (page 7) compares the ratio of the two ``VR`` with ``1 - eps``. The
  ratio of the text is implemented.
- The delta rule is printed as ``w <- w + eta (y_hat - y) x`` (page 8).
  That moves the weights up the error; the descent step
  ``w <- w - eta (y_hat - y) x`` is implemented.
- ``learning_rate`` has no value in the paper: 0.01 is my choice.
- The paper keeps the split statistics of a numeric attribute in an
  extended binary search tree with at most 50 split points (page 8). Here
  a sorted list of at most ``max_split_points`` values keeps the target
  sums of each value; once it is full, a new value is added to the nearest
  one.
- The side of the expansion is not stated: the side with the smaller
  target variance becomes the new condition. The expanded rule starts from
  the target statistics of that side, and the new default rule from those
  of the other side.
- The Page-Hinkley test uses the running mean of the errors at each step
  and is reset when a rule is expanded.
- ``max_rules`` and ``max_literals`` bound the size of the model; the
  paper has no limit.

Examples:
    >>> import numpy as np
    >>> from dense_armor.utility.rules.amrules import AMRulesRegressor
    >>> rng = np.random.default_rng(0)
    >>> model = AMRulesRegressor(n_min=100, seed=0)
    >>> for _ in range(1500):
    ...     a = float(rng.uniform(-1.0, 1.0))
    ...     _ = model.learn_one({"a": a, "b": float(rng.uniform(-1.0, 1.0))}, 5.0 * (a > 0.0))
    >>> round(model.predict_one({"a": 0.5, "b": 0.0}), 1)
    5.0
    >>> round(model.predict_one({"a": -0.5, "b": 0.0}), 1)
    0.0

References:
    Duarte, J., Gama, J., Bifet, A. (2016). Adaptive model rules from
        high-speed data streams. ACM Transactions on Knowledge Discovery
        from Data 10(3), article 30. Algorithms 1 and 2, Sections 3.1 to
        3.5.
"""

import copy
import math
from bisect import bisect_left
from typing import Any

import numpy as np

from dense_armor.roles import Regressor

_PREDICTIONS = ("adaptive", "mean", "perceptron")
_EPS = 1e-12


def _as_dict(x: Any) -> dict:
    if hasattr(x, "to_dict"):
        return dict(x.to_dict())
    try:
        return dict(x)
    except (TypeError, ValueError):
        return {}


class _Moments:
    """Running count, mean and sum of squared deviations."""

    __slots__ = ("m2", "mean", "n")

    def __init__(self, n: int = 0, mean: float = 0.0, m2: float = 0.0) -> None:
        self.n = n
        self.mean = mean
        self.m2 = m2

    def update(self, v: float) -> None:
        self.n += 1
        d = v - self.mean
        self.mean += d / self.n
        self.m2 += d * (v - self.mean)

    @property
    def std(self) -> float:
        return math.sqrt(max(self.m2, 0.0) / self.n) if self.n > 1 else 0.0


def _side(n: float, s: float, q: float) -> _Moments:
    if n <= 0:
        return _Moments()
    return _Moments(int(n), s / n, max(q - s * s / n, 0.0))


class _Splits:
    """Candidate split values of one attribute with the target sums at each."""

    __slots__ = ("n", "q", "s", "xs")

    def __init__(self) -> None:
        self.xs: list = []
        self.n: list = []
        self.s: list = []
        self.q: list = []

    def add(self, x: float, y: float, cap: int) -> None:
        i = bisect_left(self.xs, x)
        if i < len(self.xs) and self.xs[i] == x:
            j = i
        elif len(self.xs) < cap:
            self.xs.insert(i, x)
            self.n.insert(i, 0)
            self.s.insert(i, 0.0)
            self.q.insert(i, 0.0)
            j = i
        elif i == 0:
            j = 0
        elif i == len(self.xs):
            j = i - 1
        else:
            j = i if self.xs[i] - x < x - self.xs[i - 1] else i - 1
        self.n[j] += 1
        self.s[j] += y
        self.q[j] += y * y

    def best(self, total_n: float, total_s: float, total_q: float) -> tuple | None:
        if len(self.xs) < 2:
            return None
        n = np.cumsum(np.asarray(self.n, dtype=float))[:-1]
        s = np.cumsum(np.asarray(self.s, dtype=float))[:-1]
        q = np.cumsum(np.asarray(self.q, dtype=float))[:-1]
        rn, rs, rq = total_n - n, total_s - s, total_q - q
        ok = (n > 0) & (rn > 0)
        var_e = total_q / total_n - (total_s / total_n) ** 2
        n_, rn_ = np.where(ok, n, 1.0), np.where(ok, rn, 1.0)
        var_l = np.maximum(q / n_ - (s / n_) ** 2, 0.0)
        var_r = np.maximum(rq / rn_ - (rs / rn_) ** 2, 0.0)
        vr = 1.0 - (n / total_n) * var_l / var_e - (rn / total_n) * var_r / var_e
        vr = np.where(ok, vr, -np.inf)
        k = int(np.argmax(vr))
        return (
            float(vr[k]),
            self.xs[k],
            (float(n[k]), float(s[k]), float(q[k])),
            (float(rn[k]), float(rs[k]), float(rq[k])),
        )


class _Rule:
    """One rule: conditions, target statistics, linear model, error monitors."""

    def __init__(self, d: int, rng: np.random.Generator) -> None:
        self.d = d
        self.literals: list = []
        self.target = _Moments()
        self.attrs = [_Moments() for _ in range(d)]
        self.w = rng.uniform(-1.0, 1.0, d + 1)
        self.t_mean = self.n_mean = self.t_perc = self.n_perc = 0.0
        self.n_total = 0
        self.reset_window()
        self.reset_change_test()

    def reset_window(self) -> None:
        self.e_n = 0
        self.e_s = 0.0
        self.e_q = 0.0
        self.splits = [_Splits() for _ in range(self.d)]

    def reset_change_test(self) -> None:
        self.ph_n = 0
        self.ph_mean = 0.0
        self.ph_sum = 0.0
        self.ph_min = 0.0

    def covers(self, v: np.ndarray) -> bool:
        return all((v[j] <= c) if le else (v[j] > c) for j, le, c in self.literals)

    def _standardise(self, v: np.ndarray) -> np.ndarray:
        mu = np.array([a.mean for a in self.attrs])
        sd = np.array([a.std if a.n > 1 and a.std > _EPS else 1.0 for a in self.attrs])
        return (v - mu) / sd

    def _target_scale(self) -> float:
        sd = self.target.std
        return sd if sd > _EPS else 1.0

    def predict_mean(self) -> float:
        return self.target.mean if self.target.n > 0 else 0.0

    def predict_perceptron(self, v: np.ndarray) -> float:
        z = np.append(self._standardise(v), 1.0)
        out = float(self.w @ z) * self._target_scale() + self.target.mean
        return out if math.isfinite(out) else self.predict_mean()

    def predict(self, v: np.ndarray, strategy: str) -> float:
        if strategy == "mean":
            return self.predict_mean()
        if strategy == "perceptron":
            return self.predict_perceptron(v)
        if self.n_mean == 0.0 or self.n_perc == 0.0:
            return self.predict_mean()
        if self.t_perc / self.n_perc < self.t_mean / self.n_mean:
            return self.predict_perceptron(v)
        return self.predict_mean()

    def change_detected(self, err: float, phi: float, lam: float) -> bool:
        self.ph_n += 1
        self.ph_mean += (err - self.ph_mean) / self.ph_n
        self.ph_sum += err - self.ph_mean - phi
        self.ph_min = min(self.ph_min, self.ph_sum)
        return self.ph_sum - self.ph_min > lam

    def is_anomaly(self, v: np.ndarray, t: float, m_min: int) -> bool:
        if self.n_total <= m_min:
            return False
        score = 0.0
        for j in range(self.d):
            sd = self.attrs[j].std
            k = abs(v[j] - self.attrs[j].mean)
            p = 2.0 * sd * sd / (sd * sd + k * k) if sd < k else 1.0
            p = min(max(p, _EPS), 1.0 - _EPS)
            score += math.log(p) - math.log(1.0 - p)
        return score / self.d < t

    def learn(
        self, v: np.ndarray, y: float, eta: float, alpha: float, cap: int
    ) -> None:
        pm = self.predict_mean()
        pp = self.predict_perceptron(v)
        self.t_mean = alpha * self.t_mean + abs(pm - y)
        self.n_mean = alpha * self.n_mean + 1.0
        self.t_perc = alpha * self.t_perc + abs(pp - y)
        self.n_perc = alpha * self.n_perc + 1.0
        if self.target.n > 1:
            z = np.append(self._standardise(v), 1.0)
            yn = (y - self.target.mean) / self._target_scale()
            w = self.w - eta * (float(self.w @ z) - yn) * z
            self.w = w if np.isfinite(w).all() else np.zeros(self.d + 1)
        for j in range(self.d):
            self.attrs[j].update(float(v[j]))
            self.splits[j].add(float(v[j]), y, cap)
        self.target.update(y)
        self.n_total += 1
        self.e_n += 1
        self.e_s += y
        self.e_q += y * y

    def split_candidate(self, delta: float, tau: float) -> tuple | None:
        n = float(self.e_n)
        var_e = self.e_q / n - (self.e_s / n) ** 2
        if var_e <= _EPS:
            return None
        found = []
        for j in range(self.d):
            b = self.splits[j].best(n, self.e_s, self.e_q)
            if b is not None:
                found.append((b[0], j, b[1], b[2], b[3]))
        found.sort(key=lambda c: c[0], reverse=True)
        if not found or found[0][0] <= 0.0:
            return None
        second = max(found[1][0], 0.0) if len(found) > 1 else 0.0
        eps = math.sqrt(math.log(1.0 / delta) / (2.0 * n))
        if second / found[0][0] + eps < 1.0 or eps < tau:
            return found[0]
        return None

    def add_condition(self, j: int, c: float, left: tuple, right: tuple) -> _Moments:
        lm, rm = _side(*left), _side(*right)
        use_left = (lm.m2 / max(lm.n, 1)) <= (rm.m2 / max(rm.n, 1))
        self.literals.append((j, use_left, c))
        self.target = lm if use_left else rm
        self.reset_window()
        self.reset_change_test()
        return rm if use_left else lm


class AMRulesRegressor(Regressor):
    """Adaptive Model Rules regressor (Duarte, Gama and Bifet).

    Cost and memory. ``learn_one`` updates the rules that cover the sample:
    for each, one insertion in a sorted list per feature (at most
    ``max_split_points`` values) and one small linear model; every ``n_min``
    samples a rule scans its candidate splits. A model has at most
    ``max_rules`` rules of at most ``max_literals`` conditions and
    ``d * max_split_points`` stored split values each, so the state is
    constant in the length of the stream and linear in the number of
    features. There is no pure ``init_state`` / ``step`` form: the state is
    a set of rules of changing shape.

    Args:
        ordered_set: ``True`` learns and predicts with the first rule that
            covers a sample; ``False`` (my choice) with all of them.
        n_min: samples a rule sees before it tries to expand. Default 200.
        delta: one minus the confidence of the Hoeffding bound. Default
            1e-7.
        tau: tie threshold on the bound. Default 0.05.
        ph_lambda: threshold of the Page-Hinkley test. Default 35.0. It is
            in the unit of the absolute error of the target.
        ph_phi: magnitude of the changes the Page-Hinkley test allows.
            Default 0.005.
        alpha: fading factor of the error that chooses the prediction
            strategy. Default 0.99.
        learning_rate: step of the delta rule of the linear models.
            Default 0.01.
        prediction: ``"adaptive"``, ``"mean"`` or ``"perceptron"``.
        anomaly_threshold: threshold ``t`` of the anomaly score; a sample
            below it is ignored by the rule that finds it anomalous.
            ``None`` (my choice) turns the test off.
        m_min: samples a rule needs before it checks anomalies. Default 30.
        max_split_points: split values kept per feature and rule. Default
            50.
        max_rules: largest number of rules. Default 100.
        max_literals: largest number of conditions of a rule. Default 20.
        seed: seed of the generator of the initial weights.
        feature_keys: names of the features. ``None`` takes the sorted keys
            of the first sample.

    Raises:
        ValueError: on a non-positive ``n_min``, ``learning_rate``,
            ``ph_lambda``, ``max_rules``, ``max_literals`` or
            ``max_split_points``, on ``delta`` outside (0, 1), ``alpha``
            outside (0, 1), a negative ``tau``, ``ph_phi`` or ``m_min``, or
            an unknown ``prediction``.

    Examples:
        >>> from dense_armor.utility.rules.amrules import AMRulesRegressor
        >>> model = AMRulesRegressor(seed=0)
        >>> model.predict_one({"a": 1.0})
        0.0
        >>> _ = model.learn_one({"a": 1.0}, 2.0)
        >>> model.predict_one({"a": 1.0})
        2.0

    References:
        Duarte, Gama, Bifet (2016), Algorithms 1 and 2, Sections 3.1 to 3.5.
    """

    budget_s = 0.01
    memory_class = "O(window)"

    def __init__(
        self,
        ordered_set: bool = False,
        n_min: int = 200,
        delta: float = 1e-7,
        tau: float = 0.05,
        ph_lambda: float = 35.0,
        ph_phi: float = 0.005,
        alpha: float = 0.99,
        learning_rate: float = 0.01,
        prediction: str = "adaptive",
        anomaly_threshold: float | None = None,
        m_min: int = 30,
        max_split_points: int = 50,
        max_rules: int = 100,
        max_literals: int = 20,
        seed: int | None = None,
        feature_keys: list | None = None,
    ) -> None:
        if n_min < 1:
            raise ValueError(f"n_min must be >= 1, got {n_min}")
        if not 0.0 < delta < 1.0:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        if tau < 0.0:
            raise ValueError(f"tau must be >= 0, got {tau}")
        if ph_lambda <= 0.0:
            raise ValueError(f"ph_lambda must be > 0, got {ph_lambda}")
        if ph_phi < 0.0:
            raise ValueError(f"ph_phi must be >= 0, got {ph_phi}")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        if prediction not in _PREDICTIONS:
            raise ValueError(
                f"prediction must be one of {_PREDICTIONS}, got {prediction!r}"
            )
        if m_min < 0:
            raise ValueError(f"m_min must be >= 0, got {m_min}")
        if max_split_points < 1 or max_rules < 1 or max_literals < 1:
            raise ValueError(
                "max_split_points, max_rules and max_literals must be >= 1"
            )
        self.ordered_set = ordered_set
        self.n_min = n_min
        self.delta = delta
        self.tau = tau
        self.ph_lambda = ph_lambda
        self.ph_phi = ph_phi
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.prediction = prediction
        self.anomaly_threshold = anomaly_threshold
        self.m_min = m_min
        self.max_split_points = max_split_points
        self.max_rules = max_rules
        self.max_literals = max_literals
        self.seed = seed
        self.feature_keys = feature_keys
        self.rules_: list = []
        self.default_: _Rule | None = None
        self.n_seen_ = 0
        self.n_missing_ = 0
        self.n_anomalies_ = 0
        self.n_removed_ = 0
        self._keys: list | None = None
        self._rng = np.random.default_rng(seed)

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    def _vector(self, x: Any) -> np.ndarray | None:
        d = _as_dict(x)
        if not d:
            return None
        if self._keys is None:
            self._keys = (
                list(self.feature_keys) if self.feature_keys is not None else sorted(d)
            )
        try:
            v = np.array([float(d[k]) for k in self._keys], dtype=float)
        except (KeyError, TypeError, ValueError):
            return None
        return v if np.isfinite(v).all() else None

    def _expand_rule(self, rule: _Rule) -> None:
        if len(rule.literals) >= self.max_literals:
            return
        cand = rule.split_candidate(self.delta, self.tau)
        if cand is not None:
            _, j, c, left, right = cand
            rule.add_condition(j, c, left, right)

    def _expand_default(self) -> None:
        default = self.default_
        assert default is not None
        if len(self.rules_) >= self.max_rules:
            return
        cand = default.split_candidate(self.delta, self.tau)
        if cand is None:
            return
        _, j, c, left, right = cand
        grown = copy.deepcopy(default)
        other = grown.add_condition(j, c, left, right)
        self.rules_.append(grown)
        fresh = _Rule(default.d, self._rng)
        fresh.target = other
        self.default_ = fresh

    def learn_one(self, x: Any, y: Any, t: float | None = None) -> "AMRulesRegressor":
        """Update the rule set with one sample.

        A sample with a missing or non-finite value or target is skipped
        and counted in ``n_missing``.
        """
        self._time_step(t)
        try:
            target = float(y)
        except (TypeError, ValueError):
            target = math.nan
        v = self._vector(x) if math.isfinite(target) else None
        if v is None:
            self.n_missing_ += 1
            return self
        if self.default_ is None:
            self.default_ = _Rule(len(v), self._rng)
        self.n_seen_ += 1
        covered = False
        for rule in list(self.rules_):
            if not rule.covers(v):
                continue
            covered = True
            if self.anomaly_threshold is not None and rule.is_anomaly(
                v, self.anomaly_threshold, self.m_min
            ):
                self.n_anomalies_ += 1
            else:
                err = abs(rule.predict(v, self.prediction) - target)
                if rule.change_detected(err, self.ph_phi, self.ph_lambda):
                    self.rules_.remove(rule)
                    self.n_removed_ += 1
                else:
                    rule.learn(
                        v, target, self.learning_rate, self.alpha, self.max_split_points
                    )
                    if rule.e_n % self.n_min == 0:
                        self._expand_rule(rule)
            if self.ordered_set:
                break
        if not covered:
            default = self.default_
            default.learn(
                v, target, self.learning_rate, self.alpha, self.max_split_points
            )
            if default.e_n % self.n_min == 0:
                self._expand_default()
        return self

    def _covering(self, v: np.ndarray) -> list:
        rules = [r for r in self.rules_ if r.covers(v)]
        if self.ordered_set:
            rules = rules[:1]
        return rules or [self.default_]

    def predict_one(
        self, x: Any, t: float | None = None, return_std: bool = False
    ) -> Any:
        """Prediction of the covering rules, 0.0 before the first sample.

        With ``return_std`` the result is a pair: the prediction and the
        mean of the standard deviations of the targets of the covering
        rules.
        """
        v = self._vector(x) if self.default_ is not None else None
        if v is None:
            return (0.0, 0.0) if return_std else 0.0
        rules = self._covering(v)
        out = float(np.mean([r.predict(v, self.prediction) for r in rules]))
        if return_std:
            return out, float(np.mean([r.target.std for r in rules]))
        return out

    def explain_one(self, x: Any) -> list:
        """Conditions of the rules that cover ``x`` ([] for the default rule)."""
        v = self._vector(x) if self.default_ is not None else None
        if v is None or self._keys is None:
            return []
        keys = self._keys
        return [
            [
                {"feature": keys[j], "op": "<=" if le else ">", "value": c}
                for j, le, c in r.literals
            ]
            for r in self._covering(v)
            if r is not self.default_
        ]
