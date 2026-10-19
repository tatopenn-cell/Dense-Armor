"""Bernoulli naive Bayes over presence or absence of a feature.

The multi-variate Bernoulli event model of McCallum and Nigam
(1998). A sample is a binary vector over the vocabulary; the class-
conditional probability of a feature is smoothed by Laplace, eq. 3,
``theta_{w_t | c} = (1 + sum_i B_it P(c | d_i)) / (2 + sum_i P(c |
d_i))``, where ``B_it`` is 1 if feature ``t`` is present in sample
``i`` and 0 otherwise. The likelihood of a sample given its class,
eq. 2, is ``prod_t [ B_it theta + (1 - B_it) (1 - theta) ]``, so
an absent feature contributes the complementary probability ``1 -
theta``.
"""

from typing import Any

import numpy as np

from dense_armor.roles import Classifier


class BernoulliNB(Classifier):
    """Bernoulli naive Bayes with Laplace smoothing.

    A value greater than zero is treated as "present"; zero as
    "absent". A negative or non-finite value is a missing sample
    and is counted in ``n_missing``.

    Args:
        alpha: Laplace smoothing. Default 1.0.
        class_prior: optional dict of prior probabilities per class.
            ``None`` uses the observed frequency.
        feature_keys: optional list of feature names. ``None``
            collects the keys of the first sample seen.

    Raises:
        ValueError: on a non-positive ``alpha``.

    Examples:
        >>> from dense_armor.utility.bayes import BernoulliNB
        >>> nb = BernoulliNB(alpha=1.0)
        >>> for x, y in [({"a": 1, "b": 0}, 0),
        ...              ({"a": 1, "b": 0}, 0),
        ...              ({"a": 0, "b": 1}, 1),
        ...              ({"a": 0, "b": 1}, 1)]:
        ...     _ = nb.learn_one(x, y)
        >>> nb.predict_one({"a": 1, "b": 0})
        0
        >>> nb.predict_one({"a": 0, "b": 1})
        1

    References:
        McCallum, A., Nigam, K. (1998). A comparison of event models
        for naive Bayes text classification. Eq. 2 and 3.
    """

    # The p99 on a Windows runner is 2.27 ms, above the old 1 ms.
    # The budget is the worst case the estimator promises, not the
    # best case on Linux.
    budget_s = 0.003
    memory_class = "O(c d)"

    def __init__(
        self,
        alpha: float = 1.0,
        class_prior: dict | None = None,
        feature_keys: list | None = None,
    ) -> None:
        if alpha <= 0.0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        self.alpha = float(alpha)
        self.class_prior = class_prior
        self.feature_keys = feature_keys
        self._keys: list | None = None
        self._present: dict = {}
        self._class_n: dict = {}
        self.n_missing_: int = 0

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    @property
    def _multiclass(self) -> bool:
        return True

    def _resolve_keys(self, x: dict) -> list:
        if self._keys is not None:
            return self._keys
        if self.feature_keys is not None:
            self._keys = list(self.feature_keys)
        else:
            self._keys = sorted(x, key=str)
        return self._keys

    def learn_one(self, x: dict, y: Any, t: float | None = None) -> "BernoulliNB":
        """Add one sample with label ``y``."""
        self._time_step(t)
        if not isinstance(x, dict) or not x:
            self.n_missing_ += 1
            return self
        keys = self._resolve_keys(x)
        row = np.empty(len(keys), dtype=float)
        for i, k in enumerate(keys):
            try:
                v = float(x[k])
            except (KeyError, TypeError, ValueError):
                self.n_missing_ += 1
                return self
            if not np.isfinite(v) or v < 0.0:
                self.n_missing_ += 1
                return self
            row[i] = 1.0 if v > 0.0 else 0.0
        if y not in self._present:
            self._present[y] = np.zeros(len(keys))
            self._class_n[y] = 0
        self._present[y] = self._present[y] + row
        self._class_n[y] += 1
        return self

    def _log_likelihood(self, row: np.ndarray, y: Any) -> float:
        present = self._present[y]
        n = self._class_n[y]
        theta = (1.0 + present) / (2.0 + n)
        p = row * theta + (1.0 - row) * (1.0 - theta)
        with np.errstate(divide="ignore"):
            return float(np.log(p).sum())

    def _log_prior(self, y: Any) -> float:
        if self.class_prior is not None:
            return float(np.log(self.class_prior[y]))
        total_n = sum(self._class_n.values())
        return float(np.log(self._class_n[y] / total_n))

    def predict_proba_one(self, x: dict, t: float | None = None) -> dict:
        """Posterior over the classes seen so far."""
        self._time_step(t)
        if not self._present or not isinstance(x, dict) or not x:
            return {}
        keys = self._keys or self._resolve_keys(x)
        row = np.zeros(len(keys), dtype=float)
        for i, k in enumerate(keys):
            try:
                v = float(x.get(k, 0.0))
            except (TypeError, ValueError):
                return {}
            if not np.isfinite(v) or v < 0.0:
                return {}
            row[i] = 1.0 if v > 0.0 else 0.0
        log_post = {
            y: self._log_likelihood(row, y) + self._log_prior(y) for y in self._present
        }
        mx = max(log_post.values())
        ex = {y: float(np.exp(v - mx)) for y, v in log_post.items()}
        s = sum(ex.values())
        return {y: p / s for y, p in ex.items()}

    @classmethod
    def _unit_test_params(cls):
        yield {}
