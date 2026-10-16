"""Voting classifier: hard vote or mean of probabilities.

The class has no paper: the hard vote is the majority rule, the soft
vote is the mean of the members' probability dicts. The definition is
stated in this docstring, with no equation number because the two rules
are elementary.

Args:
    models: a list of classifiers. Every member is built by cloning the
        corresponding entry when it exposes ``clone``, otherwise by
        calling it as a factory.
    mode: ``"hard"`` for the majority vote, ``"soft"`` for the mean of
        the members' probabilities.
    weights: optional per-member weights, one per entry of ``models``.
        ``None`` uses equal weights.

Raises:
    ValueError: on an unknown ``mode`` or on a ``weights`` sequence of
        the wrong length.

Examples:
    >>> from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier
    >>> from dense_armor.utility.ensemble.voting import VotingClassifier
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> v = VotingClassifier([HoeffdingTreeClassifier(grace_period=20)] * 3,
    ...                       mode="hard")
    >>> for _ in range(200):
    ...     x = float(rng.normal(0.0 if rng.random() < 0.5 else 4.0, 0.5))
    ...     y = int(x > 2.0)
    ...     _ = v.learn_one({"x": x}, y)
    >>> v.predict_one({"x": 4.0})
    1
"""

from typing import Any

from dense_armor.roles import Classifier
from dense_armor.utility.ensemble._common import as_dict, finite, new_member


class VotingClassifier(Classifier):
    """Hard vote or mean of probabilities over a fixed list of classifiers.

    Args:
        models: the base classifiers (or factories).
        mode: ``"hard"`` or ``"soft"``.
        weights: optional per-member weights.

    Raises:
        ValueError: on a bad ``mode`` or a mismatched ``weights``.

    Examples:
        >>> from dense_armor.utility.ensemble.voting import VotingClassifier
        >>> from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier
        >>> v = VotingClassifier([HoeffdingTreeClassifier()], mode="soft")
        >>> v.mode
        'soft'
    """

    budget_s = 5e-3
    memory_class = "O(n_models * window)"

    def __init__(self, models, mode: str = "hard", weights=None) -> None:
        if mode not in ("hard", "soft"):
            raise ValueError(f"mode must be 'hard' or 'soft', got {mode!r}")
        if not models:
            raise ValueError("models must not be empty")
        if weights is not None and len(weights) != len(models):
            raise ValueError(
                f"weights has {len(weights)} entries, models has {len(models)}"
            )
        self.models = list(models)
        self.mode = mode
        self.weights = None if weights is None else [float(w) for w in weights]
        self.members_: list = []
        self.classes_: set = set()
        self.n_seen_ = 0
        self.n_missing_ = 0

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    def _unit_test_skips(self) -> set:
        return set()

    def _new_member(self, m):
        return new_member(m)

    def _ensure_members(self) -> None:
        if not self.members_:
            self.members_ = [self._new_member(m) for m in self.models]

    def _w(self, i: int) -> float:
        if self.weights is None:
            return 1.0
        return self.weights[i]

    def learn_one(self, x: Any, y: Any, t: float | None = None) -> "VotingClassifier":
        d = as_dict(x)
        if not d or y is None or not finite(d):
            self.n_missing_ += 1
            return self
        self._ensure_members()
        self.classes_.add(y)
        for m in self.members_:
            try:
                m.learn_one(d, y)
            except (AttributeError, TypeError):
                pass
        self.n_seen_ += 1
        return self

    def predict_proba_one(self, x: Any, t: float | None = None) -> dict:
        self._ensure_members()
        d = as_dict(x)
        if self.mode == "hard":
            votes: dict = {}
            for i, m in enumerate(self.members_):
                try:
                    p = m.predict_one(d)
                except (AttributeError, TypeError):
                    p = None
                if p is None:
                    continue
                votes[p] = votes.get(p, 0.0) + self._w(i)
            z = sum(votes.values())
            if z <= 0:
                return {}
            return {c: v / z for c, v in votes.items()}
        acc: dict = {}
        w_sum = 0.0
        for i, m in enumerate(self.members_):
            try:
                p = m.predict_proba_one(d)
            except (AttributeError, TypeError):
                p = None
            if not p:
                continue
            w_i = self._w(i)
            w_sum += w_i
            for c, v in p.items():
                acc[c] = acc.get(c, 0.0) + w_i * float(v)
        if w_sum <= 0:
            return {}
        return {c: v / w_sum for c, v in acc.items()}

    def predict_one(self, x: Any, t: float | None = None) -> Any:
        p = self.predict_proba_one(x, t=t)
        if not p:
            return None
        return max(sorted(p), key=lambda c: p[c])
