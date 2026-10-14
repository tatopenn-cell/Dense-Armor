"""Categorical naive Bayes over discrete feature values.

Each feature takes one of a finite set of categories; the model
stores a count per (class, feature, category). The class-conditional
probability of a category is smoothed by Laplace. Unlike the
multinomial model, the value of a feature is a label, not a count,
so the likelihood is ``prod_t P(x_t | c)`` with each factor one of
a multinomial over the categories of feature ``t``.
"""

from collections.abc import Hashable
from typing import Any

import numpy as np

from dense_armor.roles import Classifier


class CategoricalNB(Classifier):
    """Categorical naive Bayes with Laplace smoothing.

    Args:
        alpha: Laplace smoothing added to every (class, feature,
            category) count. Default 1.0.
        class_prior: optional dict of prior probabilities per class.
            ``None`` uses the observed frequency.
        feature_keys: optional list of feature names. ``None``
            collects the keys of the first sample seen.

    Raises:
        ValueError: on a non-positive ``alpha``.

    Examples:
        >>> from dense_armor.utility.bayes import CategoricalNB
        >>> nb = CategoricalNB(alpha=1.0)
        >>> for x, y in [({"color": "red"}, 0),
        ...              ({"color": "red"}, 0),
        ...              ({"color": "blue"}, 1),
        ...              ({"color": "blue"}, 1)]:
        ...     _ = nb.learn_one(x, y)
        >>> nb.predict_one({"color": "red"})
        0
        >>> nb.predict_one({"color": "blue"})
        1
    """

    budget_s = 0.001
    memory_class = "O(c d v)"

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
        self._categories: dict = {}
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

    def _check_value(self, v: Any) -> Hashable | None:
        if v is None:
            return None
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v

    def learn_one(self, x: dict, y: Any, t: float | None = None) -> "CategoricalNB":
        """Add one sample with label ``y``."""
        self._time_step(t)
        if not isinstance(x, dict) or not x:
            self.n_missing_ += 1
            return self
        keys = self._resolve_keys(x)
        cats = []
        for k in keys:
            if k not in x:
                self.n_missing_ += 1
                return self
            cat = self._check_value(x[k])
            if cat is None:
                self.n_missing_ += 1
                return self
            cats.append(cat)
        if y not in self._counts:
            self._counts[y] = [{} for _ in keys]
            self._categories[y] = [set() for _ in keys]
            self._class_n[y] = 0
        for i, cat in enumerate(cats):
            self._counts[y][i][cat] = self._counts[y][i].get(cat, 0) + 1
            self._categories[y][i].add(cat)
        self._class_n[y] += 1
        return self

    def _all_categories(self, i: int) -> list:
        cats = set()
        for y in self._categories:
            cats.update(self._categories[y][i])
        return sorted(cats, key=str)

    def _log_likelihood(self, cats: list, y: Any) -> float:
        total = 0.0
        n = self._class_n[y]
        for i, cat in enumerate(cats):
            all_cats = self._all_categories(i)
            v = len(all_cats)
            c = self._counts[y][i].get(cat, 0)
            p = (c + self.alpha) / (n + self.alpha * v)
            total += float(np.log(p))
        return total

    def _log_prior(self, y: Any) -> float:
        if self.class_prior is not None:
            return float(np.log(self.class_prior[y]))
        total_n = sum(self._class_n.values())
        return float(np.log(self._class_n[y] / total_n))

    def predict_proba_one(self, x: dict, t: float | None = None) -> dict:
        """Posterior over the classes seen so far."""
        self._time_step(t)
        if not self._counts or not isinstance(x, dict) or not x:
            return {}
        keys = self._keys or self._resolve_keys(x)
        cats = []
        for k in keys:
            if k not in x:
                return {}
            cat = self._check_value(x[k])
            if cat is None:
                return {}
            cats.append(cat)
        log_post = {
            y: self._log_likelihood(cats, y) + self._log_prior(y) for y in self._counts
        }
        mx = max(log_post.values())
        ex = {y: float(np.exp(v - mx)) for y, v in log_post.items()}
        s = sum(ex.values())
        return {y: p / s for y, p in ex.items()}

    @classmethod
    def _unit_test_params(cls):
        yield {}
