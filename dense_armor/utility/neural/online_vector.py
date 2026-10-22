"""Online learner for vector targets."""

from __future__ import annotations

import copy
from typing import Any

import jax.numpy as jnp

from dense_armor.roles import Regressor
from dense_armor.utility.neural.clip import clip_grad_norm
from dense_armor.utility.neural.online import _EqWrap, _read_features, _unwrap
from dense_armor.utility.neural.params import Param
from dense_armor.utility.neural.schedules import ConstantLR


class OnlineNetVector(Regressor):
    """A network trained one sample at a time, on vector targets.

    ``learn_one(x, y)`` performs the same update as one batch of
    ``Trainer.fit`` with a batch of one sample: ``zero_grad()`` on every
    ``Param``, ``forward``, ``loss.forward``, ``backward``, the clipping
    of :func:`clip_grad_norm` if ``clip_norm`` is set, the weight decay
    of the ``Trainer`` (L2 or decoupled, only on ``Param`` with two or
    more dimensions), the learning rate ``schedule(n_seen_)`` written to
    every optimizer, the update, then ``n_seen_ += 1``. If the loss has
    a ``params()`` method (as ``MatryoshkaLoss`` does), its ``Param``
    are trained too with one ``copy.deepcopy(optimizer)`` each.

    Args:
        model: any object with ``forward(x, training=True)``,
            ``backward(g)`` and ``params()``.
        loss: any object with ``forward(pred, y)`` and ``backward()``.
            If it has ``params()``, those ``Param`` are trained.
        optimizer: a prototype of ``dense_armor.utility.optim``; one
            ``copy.deepcopy`` is made for each ``Param``.
        feature_keys: the keys of the dict read as features, in order.
            If ``None``, the sorted keys of the first dict are used.
        schedule: a learning-rate schedule; default ``ConstantLR`` with
            the learning rate of ``optimizer``.
        weight_decay: the weight-decay coefficient, default 0.
        decay: ``"l2"`` (default) or ``"decoupled"``.
        clip_norm: if not None, the global L2 norm of the gradients of
            the model and of the loss is clipped at ``clip_norm``
            before the weight decay and the update.

    Raises:
        ValueError: on ``decay`` not in ``("l2", "decoupled")``, on a
            negative ``weight_decay``, or on a non-positive
            ``clip_norm``.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.layers import Dense
        >>> from dense_armor.utility.neural.losses import MSE
        >>> from dense_armor.utility.neural.online_vector import OnlineNetVector
        >>> from dense_armor.utility.optim import SGD
        >>> m = Dense(2, 1, jax.random.PRNGKey(0))
        >>> r = OnlineNetVector(m, MSE(), SGD(0.01))
        >>> _ = r.learn_one(jnp.array([1.0, 2.0]), 3.0)
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
        n_model = len(list(model.params()))
        n_loss = 0
        if hasattr(loss, "params"):
            n_loss = len(list(loss.params()))
        self.optimizers_ = [copy.deepcopy(optimizer) for _ in range(n_model + n_loss)]
        self.n_seen_ = 0
        self.n_missing_ = 0
        self.feature_keys_: list[str] | None = None

    def __repr__(self) -> str:
        return f"OnlineNetVector(n_seen_={self.n_seen_}, n_missing_={self.n_missing_})"

    def _apply_weight_decay(self, p: Param, epoch_t: float) -> None:
        if self.weight_decay <= 0 or p.value.ndim < 2:
            return
        if self.decay == "l2":
            p.grad = p.grad + self.weight_decay * p.value
        else:
            eta_t = self.schedule.multiplier(epoch_t)
            p.value = p.value - eta_t * self.weight_decay * p.value

    def _all_params(self) -> list[Param]:
        ps = list(self.model_.params())
        if hasattr(self.loss, "params"):
            ps += list(self.loss.params())
        return ps

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Constructor parameters, wrapped so two clones compare equal."""
        d = super().get_params(deep=deep)
        for k in ("model", "loss", "optimizer", "schedule"):
            if k in d:
                d[k] = _EqWrap(d[k])
        return d

    def _read_x(self, x: Any) -> jnp.ndarray | None:
        if isinstance(x, dict):
            keys = (
                self.feature_keys
                if self.feature_keys is not None
                else self.feature_keys_
            )
            if keys is None:
                self.feature_keys_ = sorted(x.keys())
                keys = self.feature_keys_
            vec, keys_used = _read_features(x, keys)
            if vec is None:
                return None
            if not bool(jnp.all(jnp.isfinite(vec))):
                return None
            if self.feature_keys is None:
                self.feature_keys_ = keys_used
            return vec
        try:
            arr = jnp.asarray(x, dtype=jnp.float32)
        except (TypeError, ValueError):
            return None
        if not bool(jnp.all(jnp.isfinite(arr))):
            return None
        return arr

    def learn_one(self, x: Any, y: Any, t: float | None = None) -> OnlineNetVector:
        """One online step; returns ``self``."""
        del t
        x_arr = self._read_x(x)
        if x_arr is None:
            self.n_missing_ += 1
            return self
        if y is None:
            self.n_missing_ += 1
            return self
        try:
            y_arr = jnp.asarray(y)
        except (TypeError, ValueError):
            self.n_missing_ += 1
            return self
        if not bool(jnp.all(jnp.isfinite(y_arr))):
            self.n_missing_ += 1
            return self

        lr_t = self.schedule(float(self.n_seen_))
        for opt in self.optimizers_:
            opt.lr = lr_t

        all_params = self._all_params()
        for p in all_params:
            p.zero_grad()

        pred = self.model_.forward(x_arr[None], training=True)
        self.loss.forward(pred, y_arr[None])
        self.model_.backward(self.loss.backward())

        if self.clip_norm is not None:
            clip_grad_norm(all_params, self.clip_norm)

        for p, opt in zip(all_params, self.optimizers_):
            self._apply_weight_decay(p, float(self.n_seen_))
            p.update(opt)

        self.n_seen_ += 1
        return self

    def predict_one(
        self,
        x: Any,
        t: float | None = None,
        return_std: bool = False,
    ) -> Any:
        """Predict from the current weights."""
        del t
        if isinstance(x, dict):
            keys = (
                self.feature_keys
                if self.feature_keys is not None
                else self.feature_keys_
            )
            if keys is None:
                out = jnp.zeros(1)
            else:
                vec, _ = _read_features(x, keys)
                if vec is None:
                    out = jnp.zeros(1)
                else:
                    out = self.model_.forward(vec[None], training=False)[0]
        else:
            try:
                x_arr = jnp.asarray(x, dtype=jnp.float32)
            except (TypeError, ValueError):
                out = jnp.zeros(1)
            else:
                out = self.model_.forward(x_arr[None], training=False)[0]
        arr = jnp.asarray(out)
        if arr.size == 1:
            val = float(arr.reshape(-1)[0])
            return (val, 0.0) if return_std else val
        return (out, 0.0) if return_std else out

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

    def load_state_dict(self, state: dict[str, Any]) -> OnlineNetVector:
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
