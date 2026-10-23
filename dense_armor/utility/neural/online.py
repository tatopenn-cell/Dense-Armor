"""Online estimators of a network: one sample at a time.

``OnlineNetRegressor`` and ``OnlineNetClassifier`` wrap a model with
``forward``, ``backward`` and ``params()`` and learn through
``learn_one(x, y)`` with the same operations, in the same order, as one
batch of ``Trainer.fit`` of N04 with a batch of one sample: zero the
gradients, forward, loss, backward, learning rate from a schedule, the
weight decay of N04, the update of every ``Param``. A sample is one
unit of the schedule (our choice: the schedule reads ``n_seen_``
directly, so a stream of 1000 samples advances the schedule by 1000;
this is stated in the docstring of ``learn_one``).
"""

from __future__ import annotations

import copy
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from dense_armor.roles import Classifier, Regressor
from dense_armor.utility.neural.clip import clip_grad_norm
from dense_armor.utility.neural.params import Param
from dense_armor.utility.neural.schedules import ConstantLR


def _read_features(
    x: Any,
    feature_keys: list[str] | None,
) -> tuple[jnp.ndarray | None, list[str] | None]:
    """Read ``x`` into a ``float32`` vector, or None on a missing value.

    Returns ``(vec, keys)``: the vector of ``float32`` and the keys used
    (which may be ``None`` if ``feature_keys`` is None and ``x`` is not
    a dict). If a key is missing, None or NaN, returns ``(None, keys)``.
    """
    if not isinstance(x, dict):
        return None, feature_keys
    keys = feature_keys if feature_keys is not None else sorted(x.keys())
    vals: list[float] = []
    for k in keys:
        if k not in x:
            return None, keys
        v = x[k]
        if v is None:
            return None, keys
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return None, keys
        if not math.isfinite(fv):
            return None, keys
        vals.append(fv)
    return jnp.asarray(vals, dtype=jnp.float32), keys


class _EqWrap:
    """Wrap a ``get_params`` value so two clones compare equal."""

    __slots__ = ("obj",)

    def __init__(self, obj: Any) -> None:
        self.obj = obj

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _EqWrap):
            other = other.obj
        return _values_equal(self.obj, other)

    def __hash__(self) -> int:
        return hash(id(self))

    def __repr__(self) -> str:
        return repr(self.obj)


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, _EqWrap):
        a = a.obj
    if isinstance(b, _EqWrap):
        b = b.obj
    if type(a) is not type(b):
        return False
    if hasattr(a, "params") and hasattr(b, "params"):
        pa = a.params()
        pb = b.params()
        if len(pa) != len(pb):
            return False
        for x, y in zip(pa, pb):
            if not jnp.array_equal(x.value, y.value):
                return False
        return True
    if hasattr(a, "__dict__") and hasattr(b, "__dict__"):
        da = a.__dict__
        db = b.__dict__
        if set(da) != set(db):
            return False
        for k in da:
            if not _values_equal(da[k], db[k]):
                return False
        return True
    return bool(a == b)


def _unwrap(v: Any) -> Any:
    return v.obj if isinstance(v, _EqWrap) else v


class OnlineNetRegressor(Regressor):
    """A network trained one sample at a time.

    ``learn_one(x, y)`` performs the same update as one batch of
    ``Trainer.fit`` of the ``Trainer`` with a batch of one sample:
    ``zero_grad()`` on every ``Param``, ``forward``, ``loss.forward``,
    ``backward``, the learning rate ``schedule(n_seen_)`` written to
    the ``lr`` of the optimizer copy of every ``Param``, the weight
    decay of the ``Trainer`` (L2 or decoupled, only on ``Param`` with
    two or more dimensions), the update, then ``n_seen_ += 1``. A
    sample is one unit of the schedule (our choice).

    Args:
        model: any object with ``forward(x, training=True)``,
            ``backward(g)`` and ``params()``.
        loss: any object with ``forward(pred, y)`` and ``backward()``.
        optimizer: a prototype of ``dense_armor.utility.optim``; one
            ``copy.deepcopy`` is made for each ``Param``.
        feature_keys: the keys of the dict read as features, in order;
            if None, the sorted keys of the first dict are used.
        schedule: a learning-rate schedule; default ``ConstantLR``
            with the learning rate of ``optimizer``.
        weight_decay: the weight-decay coefficient, default 0.
        decay: ``"l2"`` (default) or ``"decoupled"``.

    Examples:
        >>> import jax
        >>> from dense_armor.utility.neural.online import OnlineNetRegressor
        >>> from dense_armor.utility.neural.layers import Dense
        >>> from dense_armor.utility.neural.losses import MSE
        >>> from dense_armor.utility.optim import SGD
        >>> m = Dense(2, 1, jax.random.PRNGKey(0))
        >>> r = OnlineNetRegressor(m, MSE(), SGD(0.1))
        >>> r.learn_one({"a": 1.0, "b": 0.0}, 1.0)
        OnlineNetRegressor(n_seen_=1, n_missing_=0)
        >>> r.n_seen_
        1
    """

    budget_s = 0.06
    memory_class = "O(1)"

    def __init__(
        self,
        model: Any,
        loss: Any,
        optimizer: Any,
        feature_keys: Any = None,
        schedule: Any = None,
        weight_decay: float = 0.0,
        decay: str = "l2",
        clip_norm: float | None = None,
    ) -> None:
        if decay not in ("l2", "decoupled"):
            raise ValueError(f"decay must be 'l2' or 'decoupled', got {decay!r}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
        if clip_norm is not None and clip_norm <= 0:
            raise ValueError(f"clip_norm must be > 0 or None, got {clip_norm}")
        model = _unwrap(model)
        loss = _unwrap(loss)
        optimizer = _unwrap(optimizer)
        if schedule is not None:
            schedule = _unwrap(schedule)
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.feature_keys = list(feature_keys) if feature_keys is not None else None
        self.schedule = schedule if schedule is not None else ConstantLR(optimizer.lr)
        self.weight_decay = float(weight_decay)
        self.decay = decay
        self.clip_norm = clip_norm
        self.model_ = copy.deepcopy(model)
        self.optimizers_ = [copy.deepcopy(optimizer) for _ in model.params()]
        self.n_seen_ = 0
        self.n_missing_ = 0
        self.feature_keys_: list[str] | None = None

    def __repr__(self) -> str:
        return (
            f"OnlineNetRegressor(n_seen_={self.n_seen_}, n_missing_={self.n_missing_})"
        )

    def _apply_weight_decay(self, p: Param, epoch_t: float) -> None:
        if self.weight_decay <= 0 or p.value.ndim < 2:
            return
        if self.decay == "l2":
            p.grad = p.grad + self.weight_decay * p.value
        else:
            eta_t = self.schedule.multiplier(epoch_t)
            p.value = p.value - eta_t * self.weight_decay * p.value

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Constructor parameters, wrapped so two clones compare equal."""
        d = super().get_params(deep=deep)
        for k in ("model", "loss", "optimizer", "schedule"):
            if k in d:
                d[k] = _EqWrap(d[k])
        return d

    def learn_one(self, x: dict, y: Any, t: float | None = None) -> OnlineNetRegressor:
        """One online step; returns ``self``."""
        del t
        try:
            yv = float(y)
        except (TypeError, ValueError):
            self.n_missing_ += 1
            return self
        if not math.isfinite(yv):
            self.n_missing_ += 1
            return self
        if self.feature_keys is None and self.feature_keys_ is None:
            if not isinstance(x, dict):
                self.n_missing_ += 1
                return self
            self.feature_keys_ = sorted(x.keys())
        keys = (
            self.feature_keys if self.feature_keys is not None else self.feature_keys_
        )
        vec, keys_used = _read_features(x, keys)
        if vec is None:
            self.n_missing_ += 1
            return self
        if self.feature_keys is None:
            self.feature_keys_ = keys_used
        lr_t = self.schedule(float(self.n_seen_))
        for opt in self.optimizers_:
            opt.lr = lr_t
        for p in self.model_.params():
            p.zero_grad()
        pred = self.model_.forward(vec, training=True)
        y_arr = jnp.asarray([yv], dtype=jnp.float32)
        self.loss.forward(pred.reshape(-1, 1), y_arr.reshape(-1, 1))
        self.model_.backward(self.loss.backward())
        if self.clip_norm is not None:
            clip_grad_norm(self.model_.params(), self.clip_norm)
        for p, opt in zip(self.model_.params(), self.optimizers_):
            self._apply_weight_decay(p, float(self.n_seen_))
            p.update(opt)
        self.n_seen_ += 1
        return self

    def predict_one(
        self,
        x: dict,
        t: float | None = None,
        return_std: bool = False,
    ) -> Any:
        """Predict from the current weights; returns a float."""
        del t
        keys = (
            self.feature_keys if self.feature_keys is not None else self.feature_keys_
        )
        if keys is None:
            return 0.0 if not return_std else (0.0, 0.0)
        vec, _ = _read_features(x, keys)
        if vec is None:
            return 0.0 if not return_std else (0.0, 0.0)
        pred = self.model_.forward(vec, training=False)
        v = float(jnp.asarray(pred).reshape(-1)[0])
        if return_std:
            return v, 0.0
        return v

    def state_dict(self) -> dict[str, Any]:
        """Include the model, the optimizer copies and the counters."""
        return {
            "model_": copy.deepcopy(self.model_),
            "optimizers_": copy.deepcopy(self.optimizers_),
            "n_seen_": self.n_seen_,
            "n_missing_": self.n_missing_,
            "feature_keys_": (
                list(self.feature_keys_) if self.feature_keys_ is not None else None
            ),
        }

    def load_state_dict(self, state: dict[str, Any]) -> OnlineNetRegressor:
        """Restore from ``state_dict()``."""
        self.model_ = copy.deepcopy(state["model_"])
        self.optimizers_ = copy.deepcopy(state["optimizers_"])
        self.n_seen_ = state["n_seen_"]
        self.n_missing_ = state["n_missing_"]
        self.feature_keys_ = (
            list(state["feature_keys_"]) if state["feature_keys_"] is not None else None
        )
        return self

    @classmethod
    def _unit_test_params(cls):
        import jax

        from dense_armor.utility.neural.layers import Dense
        from dense_armor.utility.neural.losses import MSE
        from dense_armor.utility.optim import SGD

        yield {
            "model": Dense(3, 1, jax.random.PRNGKey(0)),
            "loss": MSE(),
            "optimizer": SGD(0.01),
            "feature_keys": ("a", "b", "c"),
        }

    def _unit_test_skips(self) -> set:
        return set()


class OnlineNetClassifier(Classifier):
    """A network classifier trained one sample at a time.

    Same operations, in the same order, as one batch of ``Trainer.fit``
    of N04 with a batch of one sample. The learning rate is
    ``schedule(n_seen_)`` (a sample is one unit of the schedule, our
    choice). ``predict_proba_one`` uses ``BCEWithLogits`` for two
    classes and ``SoftmaxCrossEntropy`` for more.

    Args:
        model: any object with ``forward``, ``backward``, ``params()``.
        loss: ``BCEWithLogits`` (two classes) or ``SoftmaxCrossEntropy``.
        optimizer: a prototype of ``dense_armor.utility.optim``.
        classes: the class labels, at least two.
        feature_keys: the keys of the dict read as features, in order.
        schedule: a learning-rate schedule; default ``ConstantLR``.
        weight_decay: the weight-decay coefficient, default 0.
        decay: ``"l2"`` (default) or ``"decoupled"``.

    Examples:
        >>> import jax
        >>> from dense_armor.utility.neural.online import OnlineNetClassifier
        >>> from dense_armor.utility.neural.layers import Dense
        >>> from dense_armor.utility.neural.losses import BCEWithLogits
        >>> from dense_armor.utility.optim import SGD
        >>> m = Dense(2, 1, jax.random.PRNGKey(0))
        >>> c = OnlineNetClassifier(m, BCEWithLogits(), SGD(0.1), classes=[0, 1])
        >>> c.learn_one({"a": 1.0, "b": 0.0}, 1)
        OnlineNetClassifier(n_seen_=1, n_missing_=0, classes=[0, 1])
        >>> sorted(c.predict_proba_one({"a": 1.0, "b": 0.0}).keys())
        [0, 1]
    """

    budget_s = 0.01
    memory_class = "O(1)"

    def __init__(
        self,
        model: Any,
        loss: Any,
        optimizer: Any,
        classes: Any,
        feature_keys: Any = None,
        schedule: Any = None,
        weight_decay: float = 0.0,
        decay: str = "l2",
        clip_norm: float | None = None,
    ) -> None:
        if decay not in ("l2", "decoupled"):
            raise ValueError(f"decay must be 'l2' or 'decoupled', got {decay!r}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
        if clip_norm is not None and clip_norm <= 0:
            raise ValueError(f"clip_norm must be > 0 or None, got {clip_norm}")
        classes_list = list(classes)
        if len(classes_list) < 2:
            raise ValueError(
                f"classes needs at least two labels, got {len(classes_list)}"
            )
        model = _unwrap(model)
        loss = _unwrap(loss)
        optimizer = _unwrap(optimizer)
        if schedule is not None:
            schedule = _unwrap(schedule)
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.classes = classes_list
        self.feature_keys = list(feature_keys) if feature_keys is not None else None
        self.schedule = schedule if schedule is not None else ConstantLR(optimizer.lr)
        self.weight_decay = float(weight_decay)
        self.decay = decay
        self.clip_norm = clip_norm
        self.model_ = copy.deepcopy(model)
        self.optimizers_ = [copy.deepcopy(optimizer) for _ in model.params()]
        self.n_seen_ = 0
        self.n_missing_ = 0
        self.feature_keys_: list[str] | None = None

    def __repr__(self) -> str:
        return (
            f"OnlineNetClassifier(n_seen_={self.n_seen_}, "
            f"n_missing_={self.n_missing_}, classes={self.classes!r})"
        )

    def _apply_weight_decay(self, p: Param, epoch_t: float) -> None:
        if self.weight_decay <= 0 or p.value.ndim < 2:
            return
        if self.decay == "l2":
            p.grad = p.grad + self.weight_decay * p.value
        else:
            eta_t = self.schedule.multiplier(epoch_t)
            p.value = p.value - eta_t * self.weight_decay * p.value

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Constructor parameters, wrapped so two clones compare equal."""
        d = super().get_params(deep=deep)
        for k in ("model", "loss", "optimizer", "schedule"):
            if k in d:
                d[k] = _EqWrap(d[k])
        return d

    def _target_vector(self, y: Any) -> jnp.ndarray:
        if len(self.classes) == 2:
            return jnp.asarray(
                [1.0 if y == self.classes[1] else 0.0], dtype=jnp.float32
            )
        idx = self.classes.index(y)
        return jnp.asarray([idx], dtype=jnp.int32)

    def _proba_from_logits(self, logits: jnp.ndarray) -> list[float]:
        if len(self.classes) == 2:
            p1 = float(jax.nn.sigmoid(logits.reshape(-1)[0]))
            vals = [1.0 - p1, p1]
        else:
            probs = np.asarray(jax.nn.softmax(logits))
            vals = [float(p) for p in probs]
        s = sum(vals)
        if s > 0.0:
            vals = [v / s for v in vals]
        return vals

    def learn_one(self, x: dict, y: Any, t: float | None = None) -> OnlineNetClassifier:
        """One online step; returns ``self``."""
        del t
        if y not in self.classes:
            self.n_missing_ += 1
            return self
        if self.feature_keys is None and self.feature_keys_ is None:
            if not isinstance(x, dict):
                self.n_missing_ += 1
                return self
            self.feature_keys_ = sorted(x.keys())
        keys = (
            self.feature_keys if self.feature_keys is not None else self.feature_keys_
        )
        vec, keys_used = _read_features(x, keys)
        if vec is None:
            self.n_missing_ += 1
            return self
        if self.feature_keys is None:
            self.feature_keys_ = keys_used
        lr_t = self.schedule(float(self.n_seen_))
        for opt in self.optimizers_:
            opt.lr = lr_t
        for p in self.model_.params():
            p.zero_grad()
        pred = self.model_.forward(vec, training=True)
        if len(self.classes) == 2:
            logits = pred.reshape(-1, 1)
        else:
            logits = pred.reshape(1, -1)
        self.loss.forward(logits, self._target_vector(y))
        g = self.loss.backward()
        self.model_.backward(g.reshape(-1))
        if self.clip_norm is not None:
            clip_grad_norm(self.model_.params(), self.clip_norm)
        for p, opt in zip(self.model_.params(), self.optimizers_):
            self._apply_weight_decay(p, float(self.n_seen_))
            p.update(opt)
        self.n_seen_ += 1
        return self

    def predict_proba_one(self, x: dict, t: float | None = None) -> dict:
        """Class probabilities; empty dict if no sample has been learned."""
        del t
        keys = (
            self.feature_keys if self.feature_keys is not None else self.feature_keys_
        )
        if keys is None:
            return {}
        vec, _ = _read_features(x, keys)
        if vec is None:
            return {}
        pred = self.model_.forward(vec, training=False)
        probs = self._proba_from_logits(jnp.asarray(pred).reshape(-1))
        return dict(zip(self.classes, probs))

    def predict_one(self, x: dict, t: float | None = None) -> Any:
        """The class with the largest probability, or None if no sample has been learned."""
        probs = self.predict_proba_one(x, t=t)
        if not probs:
            return None
        return max(probs, key=lambda k: probs[k])

    def state_dict(self) -> dict[str, Any]:
        """Include the model, the optimizer copies and the counters."""
        return {
            "model_": copy.deepcopy(self.model_),
            "optimizers_": copy.deepcopy(self.optimizers_),
            "n_seen_": self.n_seen_,
            "n_missing_": self.n_missing_,
            "feature_keys_": (
                list(self.feature_keys_) if self.feature_keys_ is not None else None
            ),
        }

    def load_state_dict(self, state: dict[str, Any]) -> OnlineNetClassifier:
        """Restore from ``state_dict()``."""
        self.model_ = copy.deepcopy(state["model_"])
        self.optimizers_ = copy.deepcopy(state["optimizers_"])
        self.n_seen_ = state["n_seen_"]
        self.n_missing_ = state["n_missing_"]
        self.feature_keys_ = (
            list(state["feature_keys_"]) if state["feature_keys_"] is not None else None
        )
        return self

    @classmethod
    def _unit_test_params(cls):
        import jax

        from dense_armor.utility.neural.layers import Dense
        from dense_armor.utility.neural.losses import BCEWithLogits
        from dense_armor.utility.optim import SGD

        yield {
            "model": Dense(3, 1, jax.random.PRNGKey(0)),
            "loss": BCEWithLogits(),
            "optimizer": SGD(0.01),
            "classes": (0, 1),
            "feature_keys": ("a", "b", "c"),
        }

    def _unit_test_skips(self) -> set:
        return set()
