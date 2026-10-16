"""Stacking classifier: a meta-model on the members' predictions.

The class has no paper cited in this folder. The definition, stated
here in words: a list of base classifiers is trained on the stream; the
meta-model is trained on the labels of the base classifiers' predictions
on the *same* sample, but only after the base members have made their
prediction and *before* they have seen the label of that sample. The
order matters: if the meta-model saw the labels of the members' training
samples, it would learn from leaked information and the ensemble would
look better on the training stream than it is. The class enforces the
order, and a test checks it.

The meta-model takes as input a dict of the members' predictions, one
key per member: ``m0``, ``m1``, ... The value is the class returned by
``predict_one`` of that member on the current sample. The label is the
true one. The meta-model never sees the raw input ``x``.

Raises:
    ValueError: on an empty ``models``.

Examples:
    >>> from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier
    >>> from dense_armor.utility.ensemble.stacking import StackingClassifier
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> base = HoeffdingTreeClassifier(grace_period=20)
    >>> meta = HoeffdingTreeClassifier(grace_period=20)
    >>> s = StackingClassifier([base, base.clone()], meta)
    >>> for _ in range(200):
    ...     x = float(rng.normal(0.0 if rng.random() < 0.5 else 4.0, 0.5))
    ...     y = int(x > 2.0)
    ...     _ = s.learn_one({"x": x}, y)
    >>> s.predict_one({"x": 4.0})
    1
"""

from typing import Any

from dense_armor.roles import Classifier
from dense_armor.utility.ensemble._common import as_dict, finite, new_member


class StackingClassifier(Classifier):
    """Stacking of a meta-model on top of a list of base classifiers.

    Args:
        models: the base classifiers (or factories).
        meta_model: the classifier that learns from the members'
            predictions, one key per member, and never sees the raw
            input or a label before the members have predicted.

    Raises:
        ValueError: on an empty ``models``.

    Examples:
        >>> from dense_armor.utility.ensemble.stacking import StackingClassifier
        >>> from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier
        >>> s = StackingClassifier([HoeffdingTreeClassifier()],
        ...                        HoeffdingTreeClassifier())
        >>> s.n_seen_
        0
    """

    budget_s = 5e-3
    memory_class = "O(n_models * window)"

    def __init__(self, models, meta_model) -> None:
        if not models:
            raise ValueError("models must not be empty")
        self.models = list(models)
        self.meta_model = meta_model
        self.members_: list = []
        self.meta_: Any = None
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

    def _ensure(self) -> None:
        if not self.members_:
            self.members_ = [self._new_member(m) for m in self.models]
        if self.meta_ is None:
            self.meta_ = self._new_member(self.meta_model)

    def _meta_input(self, d: dict) -> dict:
        out = {}
        for i, m in enumerate(self.members_):
            try:
                p = m.predict_one(d)
            except (AttributeError, TypeError):
                p = None
            if p is not None:
                out[f"m{i}"] = p
        return out

    def learn_one(self, x: Any, y: Any, t: float | None = None) -> "StackingClassifier":
        d = as_dict(x)
        if not d or y is None or not finite(d):
            self.n_missing_ += 1
            return self
        self._ensure()
        self.classes_.add(y)
        meta_x = self._meta_input(d)
        if meta_x:
            self.meta_.learn_one(meta_x, y)
        for m in self.members_:
            try:
                m.learn_one(d, y)
            except (AttributeError, TypeError):
                pass
        self.n_seen_ += 1
        return self

    def predict_proba_one(self, x: Any, t: float | None = None) -> dict:
        self._ensure()
        d = as_dict(x)
        meta_x = self._meta_input(d)
        if not meta_x:
            return {}
        try:
            return self.meta_.predict_proba_one(meta_x)
        except (AttributeError, NotImplementedError):
            return {}

    def predict_one(self, x: Any, t: float | None = None) -> Any:
        self._ensure()
        d = as_dict(x)
        meta_x = self._meta_input(d)
        if not meta_x:
            return None
        try:
            return self.meta_.predict_one(meta_x)
        except (AttributeError, NotImplementedError):
            return None
