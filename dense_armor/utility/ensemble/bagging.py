"""Online bagging (Oza and Russell 2001).

Each member of the ensemble sees the same stream, but for each arriving
sample it trains ``k`` times, where ``k`` is drawn from a ``Poisson(1)``
distribution. The paper shows that this is equivalent, in expectation,
to training each member on a bootstrap sample of the batch bagging
algorithm: the number of times an example appears in a bootstrap sample
of size ``N`` has, in the limit ``N -> infinity``, a ``Poisson(1)``
distribution (Oza and Russell 2001, Section 2, equation for the
Poisson sampling process). The equivalence is exact on the expected
counts of the contingency tables the base model builds, so a base
model whose hypothesis depends on those counts gets, on average, the
same hypothesis as in the batch version.

The prediction is an unweighted vote of the members: the class with
the most votes wins, and the probabilities are averaged over the
members that expose them.

References:
    Oza, N. C., Russell, S. (2001). Online bagging and boosting. In
        Artificial Intelligence and Statistics 2001, 105-112.
"""

import math
from typing import Any

import numpy as np

from dense_armor.utility.ensemble._common import as_dict, finite, new_member
from dense_armor.roles import Classifier


def as_dict(x: Any) -> dict:
    if hasattr(x, "to_dict"):
        return dict(x.to_dict())
    try:
        return dict(x)
    except (TypeError, ValueError):
        return {}



class OnlineBaggingClassifier(Classifier):
    """Bagging of online classifiers with a Poisson(1) resample.

    Every member is trained on the same stream, but for each arriving
    sample the class draws ``k ~ Poisson(1)`` and calls the member's
    ``learn_one`` ``k`` times. The prediction is a hard vote of the
    members; the probabilities are the mean of the members that expose
    ``predict_proba_one``. See Oza and Russell (2001), Section 2.

    Args:
        model: a callable returning a fresh base classifier, or a single
            classifier instance to clone. Every member is built with
            this.
        n_models: number of members. Must be at least one.
        seed: seed of the Poisson draw. ``None`` uses a fresh random
            state.

    Raises:
        ValueError: if ``n_models < 1``.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier
        >>> from dense_armor.utility.ensemble.bagging import OnlineBaggingClassifier
        >>> rng = np.random.default_rng(0)
        >>> clf = OnlineBaggingClassifier(
        ...     lambda: HoeffdingTreeClassifier(grace_period=20), n_models=3, seed=0)
        >>> for _ in range(400):
        ...     x = float(rng.normal(0.0 if rng.random() < 0.5 else 4.0, 0.5))
        ...     y = int(x > 2.0)
        ...     _ = clf.learn_one({"x": x}, y)
        >>> clf.predict_one({"x": 0.1}) != clf.predict_one({"x": 4.0})
        True
    """

    budget_s = 5e-3
    memory_class = "O(n_models * window)"

    def __init__(self, model, n_models: int = 10, seed: int | None = None) -> None:
        if n_models < 1:
            raise ValueError(f"n_models must be >= 1, got {n_models}")
        self.model = model
        self.n_models = int(n_models)
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.members_: list = []
        self.classes_: set = set()
        self.n_seen_ = 0
        self.n_missing_ = 0

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    def _unit_test_skips(self) -> set:
        return set()

    def _new_member(self):
        return new_member(self.model)

    def learn_one(
        self, x: Any, y: Any, t: float | None = None
    ) -> "OnlineBaggingClassifier":
        """Train every member ``k ~ Poisson(1)`` times on the sample."""
        d = as_dict(x)
        if not d or y is None or not finite(d):
            self.n_missing_ += 1
            return self
        if not self.members_:
            self.members_ = [self._new_member() for _ in range(self.n_models)]
        self.classes_.add(y)
        for m in self.members_:
            k = int(self._rng.poisson(1.0))
            for _ in range(k):
                m.learn_one(d, y)
        self.n_seen_ += 1
        return self

    def predict_proba_one(self, x: Any, t: float | None = None) -> dict:
        """Mean of the members' probability dicts."""
        if not self.members_:
            return {}
        d = as_dict(x)
        acc: dict = {}
        n_votes = 0
        for m in self.members_:
            try:
                p = m.predict_proba_one(d)
            except (AttributeError, TypeError):
                p = None
            if not p:
                continue
            n_votes += 1
            for c, v in p.items():
                acc[c] = acc.get(c, 0.0) + float(v)
        if n_votes == 0:
            return {}
        return {c: v / n_votes for c, v in acc.items()}

    def predict_one(self, x: Any, t: float | None = None) -> Any:
        """Hard vote of the members; ties go to the smallest class."""
        if not self.members_:
            return None
        d = as_dict(x)
        votes: dict = {}
        for m in self.members_:
            try:
                p = m.predict_one(d)
            except (AttributeError, TypeError):
                p = None
            if p is None:
                continue
            votes[p] = votes.get(p, 0) + 1
        if not votes:
            return None
        return max(sorted(votes), key=lambda c: votes[c])
