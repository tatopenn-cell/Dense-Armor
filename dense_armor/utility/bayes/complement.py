"""Complement naive Bayes (Rennie et al. 2003).

The class-conditional probability of a feature is estimated from
the samples of every class except the one being scored, eq. 6 in
Rennie, Shih, Teevan and Karger (2003):

    theta_hat_{~c i} = (N_{~c i} + alpha_i) / (N_{~c} + alpha),

where ``N_{~c i}`` is the number of times feature ``i`` appears in
samples whose label is not ``c``, and ``N_{~c}`` is the total count
of features in those samples. The paper writes the classification
weight as ``w_ci = log theta_hat_{~c i}`` and selects the class with
the smallest score (``argmin``), because a feature that appears
often in the complement is a weak signal for ``c``. This
implementation uses the equivalent sign-flipped form
``w_ci = -log theta_hat_{~c i}`` with the standard ``argmax`` over
the resulting scores, so that ``predict_one`` and
``predict_proba_one`` behave like the other classifiers of the
library. The weights are then normalised so that their absolute
values sum to one per class, the normalisation step described after
eq. 6 in the paper. The score does not include a class prior, by
design.
"""

from typing import Any

import numpy as np

from dense_armor.roles import Classifier


class ComplementNB(Classifier):
    """Complement naive Bayes with Laplace smoothing.

    Args:
        alpha: Laplace smoothing. Default 1.0.
        norm: if True, normalise the weight vector of each class so
            that the absolute values sum to one. Default True.
        feature_keys: optional list of feature names. ``None``
            collects the keys of the first sample seen.

    In ``predict_proba_one`` a key missing from ``x`` counts as 0.0,
    while ``learn_one`` discards a sample with a missing key
    (``n_missing_``); ``CategoricalNB`` returns ``{}`` instead.

    Raises:
        ValueError: on a non-positive ``alpha``.

    Examples:
        >>> from dense_armor.utility.bayes import ComplementNB
        >>> nb = ComplementNB(alpha=1.0)
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
        Rennie, J. D. M., Shih, L., Teevan, J., Karger, D. R. (2003).
        Tackling the poor assumptions of naive Bayes text classifiers.
        ICML. Eq. 6 and the following normalisation step.
    """

    budget_s = 0.001
    memory_class = "O(c d)"

    def __init__(
        self, alpha: float = 1.0, norm: bool = True, feature_keys: list | None = None
    ) -> None:
        if alpha <= 0.0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        self.alpha = float(alpha)
        self.norm = bool(norm)
        self.feature_keys = feature_keys
        self._keys: list | None = None
        self._counts: dict = {}
        self._totals: dict = {}
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

    def learn_one(self, x: dict, y: Any, t: float | None = None) -> "ComplementNB":
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
            row[i] = v
        if y not in self._counts:
            self._counts[y] = np.zeros(len(keys))
            self._totals[y] = 0.0
            self._class_n[y] = 0
        self._counts[y] = self._counts[y] + row
        self._totals[y] = self._totals[y] + float(row.sum())
        self._class_n[y] += 1
        return self

    def _weights(self) -> dict:
        classes = list(self._counts)
        n_feat = next(iter(self._counts.values())).size
        grand_counts = np.zeros(n_feat)
        grand_total = 0.0
        for y in classes:
            grand_counts = grand_counts + self._counts[y]
            grand_total += self._totals[y]
        out = {}
        for y in classes:
            comp_counts = grand_counts - self._counts[y]
            comp_total = grand_total - self._totals[y]
            theta = (comp_counts + self.alpha) / (comp_total + self.alpha * n_feat)
            with np.errstate(divide="ignore"):
                w = -np.log(theta)
            if self.norm:
                denom = np.abs(w).sum()
                if denom > 0.0:
                    w = w / denom
            out[y] = w
        return out

    def predict_proba_one(self, x: dict, t: float | None = None) -> dict:
        """Softmax over the complement scores."""
        self._time_step(t)
        if not self._counts or not isinstance(x, dict) or not x:
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
        weights = self._weights()
        scores = {y: float(row @ weights[y]) for y in weights}
        mx = max(scores.values())
        ex = {y: float(np.exp(s - mx)) for y, s in scores.items()}
        s = sum(ex.values())
        return {y: p / s for y, p in ex.items()}

    @classmethod
    def _unit_test_params(cls):
        yield {}
