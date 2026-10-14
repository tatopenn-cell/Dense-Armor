"""Binary-to-multiclass strategies for Dense-Armor classifiers.

Three wrappers, each taking a callable ``binary_factory`` that returns
a fresh binary classifier exposing the ``Classifier`` interface
(``learn_one`` / ``predict_proba_one`` / ``predict_one``):

- OneVsRestClassifier: one binary problem per class, class i against
  the rest. Fuerkranz (2002), Definition 2 (unordered / one-against-all
  class binarization).
- OneVsOneClassifier: one binary problem per pair of classes, voting.
  Fuerkranz (2002), Definition 4 (round robin / pairwise binarization);
  the simple voting of Section 3.
- OutputCodeClassifier: a code matrix with one row per class, one
  column per binary problem; the predicted class is the row closest to
  the binary predictions in Hamming distance. Dietterich and Bakiri
  (1995), Section 2 ("distributed output code"): each row is a class
  codeword, each column a binary problem, decoding by the closest
  codeword.

All three wrappers take the first sample's keys as the feature order,
adding new classes on the fly. A sample with NaN or a missing key is
skipped and counted in ``n_missing``.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from dense_armor.roles import Classifier


class _BaseMulticlass(Classifier):
    """Shared state and helpers of the three strategies."""

    def __init__(self, binary_factory: Callable[[], Classifier]) -> None:
        if not callable(binary_factory):
            raise TypeError("binary_factory must be callable")
        self.binary_factory = binary_factory
        self._keys: list | None = None
        self._classes: list = []
        self.n_missing_: int = 0

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    @property
    def _multiclass(self) -> bool:
        return True

    @classmethod
    def _unit_test_params(cls):
        from dense_armor.utility.linear.glm import LogisticRegression
        from dense_armor.utility.optim import SGD

        def factory():
            return LogisticRegression(optimizer=SGD(lr=0.1), intercept=True)

        yield {"binary_factory": factory}

    def _unit_test_skips(self) -> set:
        """No pure form is provided: the state is not a fixed-size array.

        The state of these estimators is a dict of counters (Bayes),
        a sliding window (neighbors), or N independent models
        (multiclass, multioutput). None of them fits the
        ``init_state`` / ``step(state, v)`` shape and none would pass
        ``jax.lax.scan``. The two checks are skipped explicitly.
        """
        return {"check_step_is_pure", "check_step_is_jittable"}

    def _resolve_keys(self, x: dict) -> list:
        if self._keys is not None:
            return self._keys
        self._keys = sorted(x, key=str)
        return self._keys

    def _check(self, x: Any) -> dict | None:
        if not isinstance(x, dict) or not x:
            return None
        if self._keys is None:
            self._resolve_keys(x)
        keys = self._keys
        if keys is None:
            return None
        for k in keys:
            if k not in x:
                return None
            try:
                v = float(x[k])
            except (TypeError, ValueError):
                return None
            if not np.isfinite(v):
                return None
        return x

    def _ensure_class(self, y: Any) -> None:
        if y not in self._classes:
            self._classes.append(y)


class OneVsRestClassifier(_BaseMulticlass):
    """One-vs-all multiclass wrapper.

    One binary problem per class, class ``c`` against every other class
    seen so far. At prediction time the class with the highest
    ``predict_proba_one`` for the positive label wins.

    Args:
        binary_factory: callable returning a fresh binary classifier.

    Examples:
        >>> from dense_armor.utility.multiclass import OneVsRestClassifier
        >>> from dense_armor.utility.linear.glm import LogisticRegression
        >>> from dense_armor.utility.optim import SGD
        >>> def factory():
        ...     return LogisticRegression(optimizer=SGD(lr=0.5))
        >>> clf = OneVsRestClassifier(binary_factory=factory)
        >>> for _ in range(30):
        ...     _ = clf.learn_one({"x": 0.0, "y": 0.0}, "A")
        ...     _ = clf.learn_one({"x": 1.0, "y": 1.0}, "B")
        ...     _ = clf.learn_one({"x": -1.0, "y": 1.0}, "C")
        >>> clf.predict_one({"x": 0.05, "y": 0.05})
        'A'
        >>> clf.predict_one({"x": 1.05, "y": 1.05})
        'B'

    References:
        Fuerkranz, J. (2002). Round Robin Classification. JMLR 2,
        721-747. Definition 2.
    """

    budget_s = 0.005
    memory_class = "O(c)"

    def __init__(self, binary_factory: Callable[[], Classifier]) -> None:
        super().__init__(binary_factory)
        self._binary: dict = {}

    def learn_one(
        self, x: dict, y: Any, t: float | None = None
    ) -> "OneVsRestClassifier":
        """Train every binary classifier: 1 if its class matches ``y``, else 0."""
        self._time_step(t)
        if self._check(x) is None:
            self.n_missing_ += 1
            return self
        self._ensure_class(y)
        for c in self._classes:
            if c not in self._binary:
                self._binary[c] = self.binary_factory()
            self._binary[c].learn_one(x, 1 if c == y else 0)
        return self

    def predict_proba_one(self, x: dict, t: float | None = None) -> dict:
        """Score every class by ``p(class = 1)``, normalised."""
        self._time_step(t)
        if not self._classes or self._check(x) is None:
            return {}
        scores = {}
        for c in self._classes:
            p = self._binary[c].predict_proba_one(x)
            scores[c] = float(p.get(1, 0.0))
        s = sum(scores.values())
        if s <= 0.0:
            return {c: 1.0 / len(self._classes) for c in self._classes}
        return {c: v / s for c, v in scores.items()}


class OneVsOneClassifier(_BaseMulticlass):
    """Round-robin (pairwise) multiclass wrapper.

    One binary problem per pair of classes, ``c (c - 1) / 2`` in total.
    Each binary classifier votes for the class it prefers; the class with
    the most votes wins. Ties are broken in favour of the class seen
    first.

    Args:
        binary_factory: callable returning a fresh binary classifier.

    Examples:
        >>> from dense_armor.utility.multiclass import OneVsOneClassifier
        >>> from dense_armor.utility.linear.glm import LogisticRegression
        >>> from dense_armor.utility.optim import SGD
        >>> def factory():
        ...     return LogisticRegression(optimizer=SGD(lr=0.5))
        >>> clf = OneVsOneClassifier(binary_factory=factory)
        >>> for _ in range(30):
        ...     _ = clf.learn_one({"x": 0.0}, "A")
        ...     _ = clf.learn_one({"x": 1.0}, "B")
        ...     _ = clf.learn_one({"x": -1.0}, "C")
        >>> clf.predict_one({"x": 0.05})
        'A'
        >>> clf.predict_one({"x": 1.05})
        'B'

    References:
        Fuerkranz, J. (2002). Round Robin Classification. JMLR 2,
        721-747. Definition 4 and Section 3.
    """

    budget_s = 0.01
    memory_class = "O(c^2)"

    def __init__(self, binary_factory: Callable[[], Classifier]) -> None:
        super().__init__(binary_factory)
        self._binary: dict = {}

    def _pair_key(self, a: Any, b: Any) -> tuple:
        return (a, b) if str(a) <= str(b) else (b, a)

    def learn_one(
        self, x: dict, y: Any, t: float | None = None
    ) -> "OneVsOneClassifier":
        """Train the binary classifier of every pair that contains ``y``."""
        self._time_step(t)
        if self._check(x) is None:
            self.n_missing_ += 1
            return self
        self._ensure_class(y)
        for a in self._classes:
            if a == y:
                continue
            key = self._pair_key(a, y)
            if key not in self._binary:
                self._binary[key] = self.binary_factory()
            left = 1 if key[0] == y else 0
            self._binary[key].learn_one(x, left)
        return self

    def predict_proba_one(self, x: dict, t: float | None = None) -> dict:
        """Voting: each pair gives one point to the class it prefers."""
        self._time_step(t)
        if not self._classes or self._check(x) is None:
            return {}
        votes = {c: 0.0 for c in self._classes}
        for key, clf in self._binary.items():
            p = clf.predict_proba_one(x)
            if not p:
                continue
            if p.get(1, 0.0) >= 0.5:
                votes[key[0]] += 1.0
            else:
                votes[key[1]] += 1.0
        s = sum(votes.values())
        if s <= 0.0:
            return {c: 1.0 / len(self._classes) for c in self._classes}
        return {c: v / s for c, v in votes.items()}


class OutputCodeClassifier(_BaseMulticlass):
    """Error-correcting output code wrapper.

    Every class is assigned a codeword (a row of +1 / -1 bits), and one
    binary classifier is trained per column: the class is positive if
    its bit is +1, negative if it is -1. A query is classified by the
    Hamming distance between the column predictions and each codeword;
    the closest row wins.

    Args:
        binary_factory: callable returning a fresh binary classifier.
        code_size: number of bits per codeword, i.e. the number of
            binary classifiers. Default 6.
        seed: random seed for the initial code matrix. Default 0.

    Examples:
        >>> from dense_armor.utility.multiclass import OutputCodeClassifier
        >>> from dense_armor.utility.linear.glm import LogisticRegression
        >>> from dense_armor.utility.optim import SGD
        >>> def factory():
        ...     return LogisticRegression(optimizer=SGD(lr=0.5))
        >>> clf = OutputCodeClassifier(binary_factory=factory, code_size=8)
        >>> for _ in range(30):
        ...     _ = clf.learn_one({"x": 0.0}, "A")
        ...     _ = clf.learn_one({"x": 1.0}, "B")
        ...     _ = clf.learn_one({"x": -1.0}, "C")
        >>> clf.predict_one({"x": 0.05})
        'A'

    References:
        Dietterich, T. G., Bakiri, G. (1995). Solving multiclass learning
        problems via error-correcting output codes. JAIR 2, 263-286.
        Section 2.
    """

    budget_s = 0.01
    memory_class = "O(c * code_size)"

    def __init__(
        self,
        binary_factory: Callable[[], Classifier],
        code_size: int = 6,
        seed: int = 0,
    ) -> None:
        super().__init__(binary_factory)
        if code_size < 2:
            raise ValueError(f"code_size must be >= 2, got {code_size}")
        self.code_size = int(code_size)
        self.seed = int(seed)
        self._codes: dict = {}

    def _new_code(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed + len(self._codes))
        bits = rng.integers(0, 2, size=self.code_size) * 2 - 1
        return bits.astype(float)

    def _ensure_class(self, y: Any) -> None:
        if y not in self._codes:
            self._codes[y] = self._new_code()

    def learn_one(
        self, x: dict, y: Any, t: float | None = None
    ) -> "OutputCodeClassifier":
        self._time_step(t)
        if self._check(x) is None:
            self.n_missing_ += 1
            return self
        self._ensure_class(y)
        code = self._codes[y]
        if not hasattr(self, "_binary"):
            self._binary = [self.binary_factory() for _ in range(self.code_size)]
        for j in range(self.code_size):
            self._binary[j].learn_one(x, 1 if code[j] > 0 else 0)
        return self

    def predict_proba_one(self, x: dict, t: float | None = None) -> dict:
        self._time_step(t)
        if not self._codes or self._check(x) is None:
            return {}
        if not hasattr(self, "_binary"):
            return {}
        bits = np.zeros(self.code_size)
        for j in range(self.code_size):
            p = self._binary[j].predict_proba_one(x)
            bits[j] = 1.0 if p.get(1, 0.0) >= 0.5 else -1.0
        dists = {}
        for c, code in self._codes.items():
            dists[c] = float(np.sum(bits != code))
        weights = {c: float(np.exp(-d / self.code_size)) for c, d in dists.items()}
        s = sum(weights.values())
        return {c: w / s for c, w in weights.items()}
