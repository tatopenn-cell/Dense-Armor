"""Multinomial naive Bayes over feature counts.

The class-conditional probability of a feature is the fraction of
its occurrences that fall in that class, smoothed by Laplace. The
formulation is the multinomial event model of McCallum and Nigam
(1998), eq. 5: ``P(d | c) = P(|d|) |d|! prod_t (theta_{w_t | c})^
{N_it} / N_it!``, with ``theta_{w_t | c}`` estimated by eq. 5
continued, ``(1 + sum_i N_it P(c | d_i)) / (|V| + sum_t sum_i N_it
P(c | d_i))``. The proportional form used here drops the factorials
and the document-length prior, which are independent of the class
in the paper's assumption.
"""

from typing import Any

import numpy as np

from dense_armor.roles import Classifier


class MultinomialNB(Classifier):
    """Multinomial naive Bayes with Laplace smoothing.

    Args:
        alpha: Laplace smoothing added to every feature count.
            Default 1.0.
        class_prior: optional dict of prior probabilities per class.
            ``None`` uses the observed frequency.
        feature_keys: optional list of feature names. ``None``
            collects the keys of the first sample seen.

    Raises:
        ValueError: on a non-positive ``alpha``.

    Examples:
        >>> from dense_armor.utility.bayes import MultinomialNB
        >>> nb = MultinomialNB(alpha=1.0)
        >>> for x, y in [({"a": 3, "b": 0}, 0),
        ...              ({"a": 2, "b": 1}, 0),
        ...              ({"a": 0, "b": 3}, 1),
        ...              ({"a": 0, "b": 4}, 1)]:
        ...     _ = nb.learn_one(x, y)
        >>> nb.predict_one({"a": 5, "b": 0})
        0
        >>> nb.predict_one({"a": 0, "b": 5})
        1

    References:
        McCallum, A., Nigam, K. (1998). A comparison of event models
        for naive Bayes text classification. AAAI Workshop. Eq. 5.
    """

    budget_s = 0.001
    memory_class = "O(d)"

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
        self._counts: dict = {}
        self._class_totals: dict = {}
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

    def _unit_test_skips(self) -> set:
        return set()

    def learn_one(self, x: dict, y: Any, t: float | None = None) -> "MultinomialNB":
        """Add one sample with label ``y``.

        A sample with a missing key or a non-finite value is
        skipped and counted in ``n_missing``.
        """
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
            row[i] = v
        if y not in self._counts:
            self._counts[y] = np.zeros(len(keys))
            self._class_totals[y] = 0.0
            self._class_n[y] = 0
        self._counts[y] = self._counts[y] + row
        self._class_totals[y] = self._class_totals[y] + float(row.sum())
        self._class_n[y] += 1
        return self

    def _log_likelihood(self, row: np.ndarray, y: Any) -> float:
        counts = self._counts[y]
        total = self._class_totals[y]
        n_feat = counts.size
        theta = (counts + self.alpha) / (total + self.alpha * n_feat)
        with np.errstate(divide="ignore"):
            log_theta = np.log(theta)
        return float(row @ log_theta)

    def _log_prior(self, y: Any) -> float:
        if self.class_prior is not None:
            return float(np.log(self.class_prior[y]))
        total_n = sum(self._class_n.values())
        return float(np.log(self._class_n[y] / total_n))

    def predict_proba_one(self, x: dict, t: float | None = None) -> dict:
        """Posterior over the classes seen so far."""
        self._time_step(t)
        if not self._counts:
            return {}
        if not isinstance(x, dict) or not x:
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
            row[i] = v
        log_post = {
            y: self._log_likelihood(row, y) + self._log_prior(y) for y in self._counts
        }
        mx = max(log_post.values())
        ex = {y: float(np.exp(v - mx)) for y, v in log_post.items()}
        s = sum(ex.values())
        return {y: p / s for y, p in ex.items()}

    @classmethod
    def _unit_test_params(cls):
        yield {}
