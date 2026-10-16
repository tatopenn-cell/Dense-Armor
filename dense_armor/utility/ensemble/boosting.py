"""Online boosting (Oza and Russell 2001).

The online AdaBoost of Oza and Russell (2001), Section 3 and Figure 2.
Each member ``m`` has two accumulators, ``lambda_sc_m`` for the samples
it classified correctly and ``lambda_sw_m`` for the wrong ones, and a
weight ``lambda_d`` for the incoming sample. The weight of the sample
for the next member is multiplied by ``N / (2 lambda_s)`` where ``N``
is the number of samples seen so far and ``lambda_s`` is the relevant
accumulator, exactly as in Figure 2 of the paper.

This is the online approximation of AdaBoost.M1 (Freund and Schapire
1997, Figure 1): the paper states the equivalence of the expected
weights, not a convergence rate, and the same is claimed here. No
error bound is stated, because neither paper states one for the online
version.

The prediction is the weighted vote ``argmax_y sum_{m: h_m(x) = y}
log(1 / beta_m)`` with ``beta_m = eps_m / (1 - eps_m)`` and
``eps_m = lambda_sw_m / (lambda_sc_m + lambda_sw_m)``, as Figure 2
prescribes.

References:
    Oza, N. C., Russell, S. (2001). Online bagging and boosting. In
        Artificial Intelligence and Statistics 2001, 105-112.
    Freund, Y., Schapire, R. E. (1997). A decision-theoretic
        generalization of on-line learning and an application to
        boosting. Journal of Computer and System Sciences 55(1),
        119-139. Figure 1, AdaBoost.M1.
"""

import math
from typing import Any

import numpy as np

from dense_armor.roles import Classifier


def _as_dict(x: Any) -> dict:
    if hasattr(x, "to_dict"):
        return dict(x.to_dict())
    try:
        return dict(x)
    except (TypeError, ValueError):
        return {}


def _finite(d: dict) -> bool:
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


class OnlineBoostingClassifier(Classifier):
    """Online AdaBoost with a Poisson resample and sample weights.

    Args:
        model: a callable returning a fresh base classifier, or a single
            classifier to clone.
        n_models: number of members. Must be at least one.
        seed: seed of the Poisson draw. ``None`` uses a fresh state.

    Raises:
        ValueError: if ``n_models < 1``.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier
        >>> from dense_armor.utility.ensemble.boosting import OnlineBoostingClassifier
        >>> rng = np.random.default_rng(0)
        >>> clf = OnlineBoostingClassifier(
        ...     lambda: HoeffdingTreeClassifier(grace_period=20),
        ...     n_models=3, seed=0)
        >>> for _ in range(400):
        ...     x = float(rng.normal(0.0 if rng.random() < 0.5 else 4.0, 0.5))
        ...     y = int(x > 2.0)
        ...     _ = clf.learn_one({"x": x}, y)
        >>> clf.predict_one({"x": 0.1}) != clf.predict_one({"x": 4.0})
        True
    """

    budget_s = 0.05
    memory_class = "O(n_models * window)"

    def __init__(self, model, n_models: int = 10, seed: int | None = None) -> None:
        if n_models < 1:
            raise ValueError(f"n_models must be >= 1, got {n_models}")
        self.model = model
        self.n_models = int(n_models)
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.members_: list = []
        self.lambda_sc_: list = []
        self.lambda_sw_: list = []
        self.classes_: set = set()
        self.n_seen_ = 0
        self.n_missing_ = 0

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    def _unit_test_skips(self) -> set:
        return set()

    def _new_member(self):
        m = self.model
        if isinstance(m, type) or (callable(m) and not hasattr(m, "learn_one")):
            return m()
        return m.clone()

    def learn_one(self, x: Any, y: Any, t: float | None = None) -> "OnlineBoostingClassifier":
        d = _as_dict(x)
        if not d or y is None or not _finite(d):
            self.n_missing_ += 1
            return self
        if not self.members_:
            self.members_ = [self._new_member() for _ in range(self.n_models)]
            self.lambda_sc_ = [0.0] * self.n_models
            self.lambda_sw_ = [0.0] * self.n_models
        self.classes_.add(y)
        lambda_d = 1.0
        for m_i, m in enumerate(self.members_):
            k = int(self._rng.poisson(lambda_d))
            correct = True
            for _ in range(k):
                m.learn_one(d, y)
            try:
                p = m.predict_one(d)
            except (AttributeError, TypeError):
                p = None
            correct = p == y
            if correct:
                self.lambda_sc_[m_i] += lambda_d
                denom = self.lambda_sc_[m_i]
                if denom > 0:
                    lambda_d = lambda_d * self.n_seen_ / (2.0 * denom)
            else:
                self.lambda_sw_[m_i] += lambda_d
                denom = self.lambda_sw_[m_i]
                if denom > 0:
                    lambda_d = lambda_d * self.n_seen_ / (2.0 * denom)
            if not math.isfinite(lambda_d) or lambda_d <= 0.0:
                lambda_d = 1.0
        self.n_seen_ += 1
        return self

    def _beta(self, m_i: int) -> float:
        sc = self.lambda_sc_[m_i]
        sw = self.lambda_sw_[m_i]
        if sc + sw <= 0.0:
            return 1.0
        eps = sw / (sc + sw)
        eps = max(min(eps, 1.0 - 1e-12), 1e-12)
        return eps / (1.0 - eps)

    def predict_proba_one(self, x: Any, t: float | None = None) -> dict:
        """Weighted probability, ``sum log(1/beta)`` normalised."""
        if not self.members_:
            return {}
        d = _as_dict(x)
        weights: dict = {}
        for m_i, m in enumerate(self.members_):
            try:
                p = m.predict_one(d)
            except (AttributeError, TypeError):
                p = None
            if p is None:
                continue
            w = math.log(1.0 / max(self._beta(m_i), 1e-12))
            weights[p] = weights.get(p, 0.0) + max(w, 0.0)
        z = sum(weights.values())
        if z <= 0:
            return {}
        return {c: v / z for c, v in weights.items()}

    def predict_one(self, x: Any, t: float | None = None) -> Any:
        """Weighted vote, ties go to the smallest class."""
        if not self.members_:
            return None
        d = _as_dict(x)
        weights: dict = {}
        for m_i, m in enumerate(self.members_):
            try:
                p = m.predict_one(d)
            except (AttributeError, TypeError):
                p = None
            if p is None:
                continue
            w = math.log(1.0 / max(self._beta(m_i), 1e-12))
            weights[p] = weights.get(p, 0.0) + max(w, 0.0)
        if not weights:
            return None
        return max(sorted(weights), key=lambda c: weights[c])
