"""Multi-output classifiers and regressors on a dict of targets.

Every estimator takes ``x`` and a dict of targets ``y = {name: value}``
and returns a dict of predictions, one per target. The interface
follows ``Classifier`` / ``Regressor`` for the single-target case;
``predict_one(x)`` returns the dict of targets, ``predict_proba_one(x)``
(for the classifier variants) returns ``{target: p(target=1 | x)}``.

Five classes:

- ``PerOutputClassifier`` / ``PerOutputRegressor``: independent models,
  one per target. This is the single-target baseline (ST) of
  Spyromitros-Xioufis et al. (2016), Section 2.
- ``ClassifierChain`` / ``RegressorChain``: one model per target,
  chained in an order; the model for target ``j`` is trained on
  ``x`` plus the **true** values of the previous targets of the chain
  (so ``hj: X x Y^{j-1} -> Y_j``), and at prediction time it receives
  the **predicted** values of the previous targets. Dembczynski et al.
  (2010), Section 4.1 (the original classifier chain of Read et al.
  2009), and Spyromitros-Xioufis et al. (2016), Section 2.2
  (Regressor Chains). The training/prediction discrepancy on the
  previous targets is the well-known limitation of the chaining method
  (Spyromitros-Xioufis et al. 2016, "discrepancy of the values of the
  additional input variables between training and prediction").
- ``ProbabilisticClassifierChain``: estimates the joint

      Px(y) = f1(x) * prod_{i=2}^{m} fi(x, y1, ..., y_{i-1})    (11-12)

  of Dembczynski et al. (2010), and predicts the joint mode

      h*(x) = arg max_y Px(y)                                    (10)

  which is the Bayes-optimal decision for the subset zero-one loss.
  Exact inference enumerates the ``2**m`` labelings; the paper notes
  that this limits the method to "not more than about 15" labels.
  ``predict_proba_one`` returns the marginals P(Y_i = 1 | x).

The order of the chain is a hyperparameter. ``order=None`` uses a
random permutation with the given ``seed``; a list of target names
fixes the order explicitly. Dembczynski et al. (2010), Section 4.3,
note that the product rule does not depend on the order, but two
different chains give different results because they train different
models; the effect on the stream used in this repository is measured
in ``test/multioutput/test_multioutput.py``.

A sample with NaN or a missing key is skipped and counted in
``n_missing``.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from dense_armor.roles import Classifier, Regressor


class _BaseMultiOutput:
    """Shared helpers: feature keys, target order, NaN check."""

    def __init__(
        self, factory: Callable[[], Any], order: list | None, seed: int
    ) -> None:
        if not callable(factory):
            raise TypeError("factory must be callable")
        self.factory = factory
        self.order = None if order is None else list(order)
        self.seed = int(seed)
        self._keys: list | None = None
        self._targets: list = []
        self.n_missing_: int = 0

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    def _resolve_keys(self, x: dict) -> list:
        if self._keys is not None:
            return self._keys
        self._keys = sorted(x, key=str)
        return self._keys

    def _check_x(self, x: Any) -> dict | None:
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

    def _check_y(self, y: Any) -> dict | None:
        if not isinstance(y, dict) or not y:
            return None
        for v in y.values():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return None
            if not np.isfinite(fv):
                return None
        return y

    def _ensure_targets(self, y: dict) -> None:
        for name in y:
            if name not in self._targets:
                self._targets.append(name)

    def _chain_order(self) -> list:
        if self.order is not None:
            for name in self.order:
                if name not in self._targets:
                    raise ValueError(f"order contains unknown target {name!r}")
            missing = [n for n in self._targets if n not in self.order]
            return list(self.order) + missing
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(self._targets))
        return [self._targets[i] for i in perm]


class PerOutputClassifier(_BaseMultiOutput, Classifier):
    """One binary classifier per target, learned independently.

    The single-target baseline (Spyromitros-Xioufis et al. 2016,
    Section 2). Each target is a separate binary problem trained on
    ``x`` alone.

    Args:
        factory: callable returning a fresh binary ``Classifier``.
        order: unused, kept for interface symmetry with the chains.
        seed: unused.

    Examples:
        >>> from dense_armor.utility.multioutput import PerOutputClassifier
        >>> from dense_armor.utility.linear.glm import LogisticRegression
        >>> from dense_armor.utility.optim import SGD
        >>> def factory():
        ...     return LogisticRegression(optimizer=SGD(lr=0.5))
        >>> clf = PerOutputClassifier(factory=factory)
        >>> for _ in range(20):
        ...     _ = clf.learn_one({"a": 0.0}, {"y0": 0, "y1": 1})
        ...     _ = clf.learn_one({"a": 1.0}, {"y0": 1, "y1": 0})
        >>> clf.predict_one({"a": 0.05})
        {'y0': 0, 'y1': 1}
    """

    budget_s = 0.005
    memory_class = "O(m * d)"

    def __init__(
        self,
        factory: Callable[[], Classifier],
        order: list | None = None,
        seed: int = 0,
    ) -> None:
        _BaseMultiOutput.__init__(self, factory, order, seed)
        self._models: dict = {}

    def _unit_test_skips(self) -> set:
        """No pure form is provided: the state is not a fixed-size array.

        The state of these estimators is a dict of counters (Bayes),
        a sliding window (neighbors), or N independent models
        (multiclass, multioutput). None of them fits the
        ``init_state`` / ``step(state, v)`` shape and none would pass
        ``jax.lax.scan``. The two checks are skipped explicitly.
        """
        return {"check_step_is_pure", "check_step_is_jittable"}

    @classmethod
    def _unit_test_params(cls):
        from dense_armor.utility.linear.glm import LogisticRegression
        from dense_armor.utility.optim import SGD

        def factory():
            return LogisticRegression(optimizer=SGD(lr=0.1), intercept=True)

        yield {"factory": factory}

    def learn_one(
        self, x: dict, y: dict, t: float | None = None
    ) -> "PerOutputClassifier":
        if self._check_x(x) is None or self._check_y(y) is None:
            self.n_missing_ += 1
            return self
        self._ensure_targets(y)
        for name, value in y.items():
            if name not in self._models:
                self._models[name] = self.factory()
            self._models[name].learn_one(x, int(value))
        return self

    def predict_proba_one(self, x: dict, t: float | None = None) -> dict:
        if not self._models or self._check_x(x) is None:
            return {}
        out = {}
        for name, model in self._models.items():
            p = model.predict_proba_one(x)
            out[name] = float(p.get(1, 0.0))
        return out

    def predict_one(self, x: dict, t: float | None = None) -> dict:
        proba = self.predict_proba_one(x, t)
        if not proba:
            return {}
        return {name: (1 if p >= 0.5 else 0) for name, p in proba.items()}


class PerOutputRegressor(_BaseMultiOutput, Regressor):
    """One regressor per target, learned independently.

    The single-target baseline for multi-target regression
    (Spyromitros-Xioufis et al. 2016, Section 2).

    Args:
        factory: callable returning a fresh ``Regressor``.
        order: unused.
        seed: unused.

    Examples:
        >>> from dense_armor.utility.multioutput import PerOutputRegressor
        >>> from dense_armor.utility.linear.glm import LinearRegression
        >>> from dense_armor.utility.optim import SGD
        >>> def factory():
        ...     return LinearRegression(optimizer=SGD(lr=0.5), intercept=True)
        >>> reg = PerOutputRegressor(factory=factory)
        >>> for i in range(30):
        ...     t = float(i) / 30.0
        ...     _ = reg.learn_one({"x": t}, {"y0": 0.2, "y1": 0.8})
        >>> out = reg.predict_one({"x": 0.5})
        >>> round(out["y0"], 1), round(out["y1"], 1)
        (0.2, 0.8)
    """

    budget_s = 0.005
    memory_class = "O(m * d)"

    def __init__(
        self, factory: Callable[[], Regressor], order: list | None = None, seed: int = 0
    ) -> None:
        _BaseMultiOutput.__init__(self, factory, order, seed)
        self._models: dict = {}

    def _unit_test_skips(self) -> set:
        """No pure form is provided: the state is not a fixed-size array.

        The state of these estimators is a dict of counters (Bayes),
        a sliding window (neighbors), or N independent models
        (multiclass, multioutput). None of them fits the
        ``init_state`` / ``step(state, v)`` shape and none would pass
        ``jax.lax.scan``. The two checks are skipped explicitly.
        """
        return {"check_step_is_pure", "check_step_is_jittable"}

    @classmethod
    def _unit_test_params(cls):
        from dense_armor.utility.linear.glm import LinearRegression
        from dense_armor.utility.optim import SGD

        def factory():
            return LinearRegression(optimizer=SGD(lr=0.1), intercept=True)

        yield {"factory": factory}

    def learn_one(
        self, x: dict, y: dict, t: float | None = None
    ) -> "PerOutputRegressor":
        if self._check_x(x) is None or self._check_y(y) is None:
            self.n_missing_ += 1
            return self
        self._ensure_targets(y)
        for name, value in y.items():
            if name not in self._models:
                self._models[name] = self.factory()
            self._models[name].learn_one(x, float(value))
        return self

    def predict_one(
        self,
        x: dict,
        t: float | None = None,
        return_std: bool = False,
    ) -> dict:
        if not self._models or self._check_x(x) is None:
            return {}
        out = {}
        for name, model in self._models.items():
            out[name] = float(model.predict_one(x))
        return out


class ClassifierChain(_BaseMultiOutput, Classifier):
    """Classifier chain: chained binary classifiers.

    One binary classifier per target, in a fixed order. At training
    the classifier for target ``j`` sees ``x`` plus the **true** values
    of the targets of the previous links (Dembczynski et al. 2010,
    Section 4.1, and Read et al. 2009). At prediction time the model
    for target ``j`` sees ``x`` plus the **predicted** values of the
    previous links. The discrepancy between the two is a known
    limitation of the chaining method (Spyromitros-Xioufis et al.
    2016, Section 2.2).

    The order of the chain is a hyperparameter: different orders train
    different models and give different predictions, even though the
    product rule does not depend on the order (Dembczynski et al.
    2010, Section 4.3).

    Args:
        factory: callable returning a fresh binary ``Classifier``.
        order: explicit list of target names, or ``None`` for a random
            permutation with the given ``seed``.
        seed: seed for the random permutation when ``order`` is None.

    Examples:
        >>> from dense_armor.utility.multioutput import ClassifierChain
        >>> from dense_armor.utility.linear.glm import LogisticRegression
        >>> from dense_armor.utility.optim import SGD
        >>> def factory():
        ...     return LogisticRegression(optimizer=SGD(lr=0.5))
        >>> clf = ClassifierChain(factory=factory, order=["y0", "y1"])
        >>> for _ in range(20):
        ...     _ = clf.learn_one({"a": 0.0}, {"y0": 0, "y1": 0})
        ...     _ = clf.learn_one({"a": 1.0}, {"y0": 1, "y1": 1})
        >>> clf.predict_one({"a": 1.05})
        {'y0': 1, 'y1': 1}

    References:
        Dembczynski, K., Cheng, W., Huellermeier, E. (2010). Bayes
        optimal multilabel classification via probabilistic classifier
        chains. ICML. Section 4.1.
        Spyromitros-Xioufis, E., Tsoumakas, G., Groves, W., Vlahavas,
        I. (2016). Multi-target regression via input space expansion.
        Machine Learning 104(1), 55-98. Section 2.2.
    """

    budget_s = 0.01
    memory_class = "O(m * d + m^2)"

    def __init__(
        self,
        factory: Callable[[], Classifier],
        order: list | None = None,
        seed: int = 0,
    ) -> None:
        _BaseMultiOutput.__init__(self, factory, order, seed)
        self._models: list = []
        self._order_used: list | None = None

    def _unit_test_skips(self) -> set:
        """No pure form is provided: the state is not a fixed-size array.

        The state of these estimators is a dict of counters (Bayes),
        a sliding window (neighbors), or N independent models
        (multiclass, multioutput). None of them fits the
        ``init_state`` / ``step(state, v)`` shape and none would pass
        ``jax.lax.scan``. The two checks are skipped explicitly.
        """
        return {"check_step_is_pure", "check_step_is_jittable"}

    @classmethod
    def _unit_test_params(cls):
        from dense_armor.utility.linear.glm import LogisticRegression
        from dense_armor.utility.optim import SGD

        def factory():
            return LogisticRegression(optimizer=SGD(lr=0.1), intercept=True)

        yield {"factory": factory}

    def _expanded(self, x: dict, prev: dict) -> dict:
        return {**x, **{f"__prev_{k}": float(v) for k, v in prev.items()}}

    def _prepare(self, y: dict) -> list:
        order = self._chain_order()
        if self._order_used is None:
            self._order_used = list(order)
            self._models = [self.factory() for _ in order]
        return order

    def learn_one(self, x: dict, y: dict, t: float | None = None) -> "ClassifierChain":
        if self._check_x(x) is None or self._check_y(y) is None:
            self.n_missing_ += 1
            return self
        self._ensure_targets(y)
        order = self._prepare(y)
        prev: dict = {}
        for j, name in enumerate(order):
            self._models[j].learn_one(self._expanded(x, prev), int(y[name]))
            prev[name] = int(y[name])
        return self

    def predict_proba_one(self, x: dict, t: float | None = None) -> dict:
        if self._order_used is None or self._check_x(x) is None:
            return {}
        pred: dict = {}
        proba: dict = {}
        for j, name in enumerate(self._order_used):
            p = self._models[j].predict_proba_one(self._expanded(x, pred))
            p1 = float(p.get(1, 0.0))
            proba[name] = p1
            pred[name] = 1 if p1 >= 0.5 else 0
        return proba

    def predict_one(self, x: dict, t: float | None = None) -> dict:
        proba = self.predict_proba_one(x, t)
        if not proba:
            return {}
        return {name: (1 if p >= 0.5 else 0) for name, p in proba.items()}


class RegressorChain(_BaseMultiOutput, Regressor):
    """Regressor chain: chained regressors.

    One regressor per target, in a fixed order. At training the
    regressor for target ``j`` sees ``x`` plus the **true** values of
    the targets of the previous links (``hj: X x R^{j-1} -> R``). At
    prediction it sees ``x`` plus the **predicted** values. Spyromitros
    -Xioufis et al. (2016), Section 2.2.

    Args:
        factory: callable returning a fresh ``Regressor``.
        order: explicit list of target names, or ``None`` for random.
        seed: seed for the random permutation.

    Examples:
        >>> from dense_armor.utility.multioutput import RegressorChain
        >>> from dense_armor.utility.linear.glm import LinearRegression
        >>> from dense_armor.utility.optim import SGD
        >>> def factory():
        ...     return LinearRegression(optimizer=SGD(lr=0.5), intercept=True)
        >>> reg = RegressorChain(factory=factory, order=["y0", "y1"])
        >>> for i in range(30):
        ...     t = float(i) / 30.0
        ...     _ = reg.learn_one({"x": t}, {"y0": 0.2, "y1": 0.8})
        >>> out = reg.predict_one({"x": 0.5})
        >>> round(out["y0"], 1), round(out["y1"], 1)
        (0.2, 0.8)

    References:
        Spyromitros-Xioufis, E., et al. (2016). Machine Learning 104(1),
        55-98. Section 2.2.
    """

    budget_s = 0.01
    memory_class = "O(m * d + m^2)"

    def __init__(
        self, factory: Callable[[], Regressor], order: list | None = None, seed: int = 0
    ) -> None:
        _BaseMultiOutput.__init__(self, factory, order, seed)
        self._models: list = []
        self._order_used: list | None = None

    def _unit_test_skips(self) -> set:
        """No pure form is provided: the state is not a fixed-size array.

        The state of these estimators is a dict of counters (Bayes),
        a sliding window (neighbors), or N independent models
        (multiclass, multioutput). None of them fits the
        ``init_state`` / ``step(state, v)`` shape and none would pass
        ``jax.lax.scan``. The two checks are skipped explicitly.
        """
        return {"check_step_is_pure", "check_step_is_jittable"}

    @classmethod
    def _unit_test_params(cls):
        from dense_armor.utility.linear.glm import LinearRegression
        from dense_armor.utility.optim import SGD

        def factory():
            return LinearRegression(optimizer=SGD(lr=0.1), intercept=True)

        yield {"factory": factory}

    def _expanded(self, x: dict, prev: dict) -> dict:
        return {**x, **{f"__prev_{k}": float(v) for k, v in prev.items()}}

    def _prepare(self, y: dict) -> list:
        order = self._chain_order()
        if self._order_used is None:
            self._order_used = list(order)
            self._models = [self.factory() for _ in order]
        return order

    def learn_one(self, x: dict, y: dict, t: float | None = None) -> "RegressorChain":
        if self._check_x(x) is None or self._check_y(y) is None:
            self.n_missing_ += 1
            return self
        self._ensure_targets(y)
        order = self._prepare(y)
        prev: dict = {}
        for j, name in enumerate(order):
            self._models[j].learn_one(self._expanded(x, prev), float(y[name]))
            prev[name] = float(y[name])
        return self

    def predict_one(
        self,
        x: dict,
        t: float | None = None,
        return_std: bool = False,
    ) -> dict:
        if self._order_used is None or self._check_x(x) is None:
            return {}
        out: dict = {}
        for j, name in enumerate(self._order_used):
            v = float(self._models[j].predict_one(self._expanded(x, out)))
            out[name] = v
        return out


class ProbabilisticClassifierChain(_BaseMultiOutput, Classifier):
    """Probabilistic classifier chain (Bayes-optimal for 0/1 subset loss).

    Estimates the joint Px(y) by the product rule of Dembczynski et al.
    (2010), eq. 11-12:

        Px(y) = f1(x) * prod_{i=2}^{m} fi(x, y1, ..., y_{i-1})

    and predicts the joint mode (eq. 10):

        h*(x) = arg max_y Px(y)

    which is the Bayes-optimal decision for the subset zero-one loss.
    The exact argmax enumerates the ``2**m`` labelings; the paper notes
    that this limits the method to "not more than about 15" labels.

    ``predict_proba_one`` returns the marginals P(Y_i = 1 | x),
    obtained by summing the joint over all labelings. The full joint
    is available through :meth:`joint_proba_one`.

    Args:
        factory: callable returning a fresh binary ``Classifier``.
        order: explicit list of target names, or ``None`` for random.
        seed: seed for the random permutation.

    Examples:
        >>> from dense_armor.utility.multioutput import (
        ...     ProbabilisticClassifierChain,
        ... )
        >>> from dense_armor.utility.linear.glm import LogisticRegression
        >>> from dense_armor.utility.optim import SGD
        >>> def factory():
        ...     return LogisticRegression(optimizer=SGD(lr=0.5))
        >>> clf = ProbabilisticClassifierChain(
        ...     factory=factory, order=["y0", "y1"]
        ... )
        >>> for _ in range(30):
        ...     _ = clf.learn_one({"a": 0.0}, {"y0": 0, "y1": 0})
        ...     _ = clf.learn_one({"a": 1.0}, {"y0": 1, "y1": 1})
        >>> clf.predict_one({"a": 1.05})
        {'y0': 1, 'y1': 1}

    References:
        Dembczynski, K., Cheng, W., Huellermeier, E. (2010). Bayes
        optimal multilabel classification via probabilistic classifier
        chains. ICML. Section 4, eq. 10, 11, 12.
    """

    budget_s = 0.05
    memory_class = "O(m * d + m^2)"

    def __init__(
        self,
        factory: Callable[[], Classifier],
        order: list | None = None,
        seed: int = 0,
    ) -> None:
        _BaseMultiOutput.__init__(self, factory, order, seed)
        self._models: list = []
        self._order_used: list | None = None

    def _unit_test_skips(self) -> set:
        """No pure form is provided: the state is not a fixed-size array.

        The state of these estimators is a dict of counters (Bayes),
        a sliding window (neighbors), or N independent models
        (multiclass, multioutput). None of them fits the
        ``init_state`` / ``step(state, v)`` shape and none would pass
        ``jax.lax.scan``. The two checks are skipped explicitly.
        """
        return {"check_step_is_pure", "check_step_is_jittable"}

    @classmethod
    def _unit_test_params(cls):
        from dense_armor.utility.linear.glm import LogisticRegression
        from dense_armor.utility.optim import SGD

        def factory():
            return LogisticRegression(optimizer=SGD(lr=0.1), intercept=True)

        yield {"factory": factory}

    def _expanded(self, x: dict, prev: dict) -> dict:
        return {**x, **{f"__prev_{k}": int(v) for k, v in prev.items()}}

    def _prepare(self, y: dict) -> list:
        order = self._chain_order()
        if self._order_used is None:
            self._order_used = list(order)
            self._models = [self.factory() for _ in order]
        return order

    def learn_one(
        self, x: dict, y: dict, t: float | None = None
    ) -> "ProbabilisticClassifierChain":
        if self._check_x(x) is None or self._check_y(y) is None:
            self.n_missing_ += 1
            return self
        self._ensure_targets(y)
        order = self._prepare(y)
        prev: dict = {}
        for j, name in enumerate(order):
            self._models[j].learn_one(self._expanded(x, prev), int(y[name]))
            prev[name] = int(y[name])
        return self

    def _likelihood_for_labeling(self, x: dict, bits: list) -> float:
        assert self._order_used is not None
        prev: dict = {}
        logp = 0.0
        for j, name in enumerate(self._order_used):
            p = self._models[j].predict_proba_one(self._expanded(x, prev))
            p1 = float(min(max(p.get(1, 0.0), 1e-12), 1.0 - 1e-12))
            b = bits[j]
            logp += float(np.log(p1) if b == 1 else np.log(1.0 - p1))
            prev[name] = b
        return float(np.exp(logp))

    def _all_labelings(self, m: int):
        for i in range(2**m):
            yield [(i >> k) & 1 for k in range(m)]

    def joint_proba_one(self, x: dict, t: float | None = None) -> dict:
        """Full joint Px(y) over all 2**m labelings, as {tuple: prob}."""
        if self._order_used is None or self._check_x(x) is None:
            return {}
        m = len(self._order_used)
        if m > 15:
            raise ValueError(f"exact joint needs 2**m evaluations, m = {m} > 15")
        out = {}
        for bits in self._all_labelings(m):
            out[tuple(bits)] = self._likelihood_for_labeling(x, bits)
        return out

    def predict_proba_one(self, x: dict, t: float | None = None) -> dict:
        joint = self.joint_proba_one(x, t)
        if not joint or self._order_used is None:
            return {}
        marg = {}
        for j, name in enumerate(self._order_used):
            s = 0.0
            for bits, p in joint.items():
                if bits[j] == 1:
                    s += p
            marg[name] = s
        return marg

    def predict_one(self, x: dict, t: float | None = None) -> dict:
        joint = self.joint_proba_one(x, t)
        if not joint or self._order_used is None:
            return {}
        best_bits = max(joint, key=lambda k: joint[k])
        return {name: int(b) for name, b in zip(self._order_used, best_bits)}
