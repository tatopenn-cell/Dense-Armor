"""The standard training loop: mini-batches, schedules, weight decay,
early stopping.

``Trainer(model, loss, optimizer, schedule=None, weight_decay=0.0,
decay="l2")`` runs epochs over a model with ``forward``, ``backward``
and ``params()``. The optimizer is a prototype: the trainer makes one
``copy.deepcopy`` of it for each ``Param`` (the object form keeps the
state of one array). The default schedule is ``ConstantLR(optimizer.lr)``.

The weight decay applies only to the ``Param`` whose ``value`` has two
or more dimensions (the weights). ``decay="l2"`` adds ``weight_decay *
value`` to the gradient before the update (Algorithm 1, line 6, page 3
of Loshchilov and Hutter 2019); ``decay="decoupled"`` subtracts
``eta_t * weight_decay * value`` from the weights **before the
optimizer update**, using the previous value (Algorithm 1, line 9, and
Algorithm 2, line 12, of the same paper: ``theta_t = theta_{t-1} -
m_t - eta_t * lambda * theta_{t-1}``). With standard SGD the two forms
are equivalent when the L2 coefficient is ``lambda / alpha`` for a
decoupled decay ``lambda`` (Proposition 1, page 2).

Early stopping: with a validation set and ``patience``, the training
stops at the end of the epoch after which the validation loss has not
improved by ``min_delta`` for ``patience`` consecutive epochs. The
weights of the best epoch are restored. This is the textbook rule,
declared as our choice.
"""

from __future__ import annotations

import copy
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from dense_armor.utility.neural.batching import minibatches
from dense_armor.utility.neural.clip import clip_grad_norm
from dense_armor.utility.neural.compiled import (
    _init_state as init_state,
)
from dense_armor.utility.neural.compiled import (
    _read_values as read_values,
)
from dense_armor.utility.neural.compiled import (
    _write_values as write_values,
)
from dense_armor.utility.neural.compiled import (
    make_step,
)
from dense_armor.utility.neural.params import Param
from dense_armor.utility.neural.schedules import ConstantLR


class Trainer:
    """Train a model on a dataset, with mini-batches and schedules.

    Args:
        model: any object with ``forward(x, training=True)``,
            ``backward(g)``, ``params()`` and ``zero_grad()``.
        loss: any object with ``forward(pred, target)`` and
            ``backward()``.
        optimizer: a prototype of ``dense_armor.utility.optim``
            (``SGD``, ``Adam``); one ``deepcopy`` is made for each
            ``Param``.
        schedule: a learning-rate schedule; default ``ConstantLR``
            with the learning rate of ``optimizer``.
        weight_decay: the weight-decay coefficient, default 0.
        decay: ``"l2"`` (default) or ``"decoupled"``.
        clip_norm: the global L2 clipping threshold, or ``None``.
        compile: if True, the whole step is a single ``jax.jit``
            function with a vectorised optimizer over the pytree of
            the parameters; if False (default) the Python loop is used
            and its results do not change.

    Raises:
        ValueError: if ``decay`` is neither ``"l2"`` nor ``"decoupled"``,
            or if ``weight_decay`` is negative.

    Examples:
        >>> import jax
        >>> from dense_armor.utility.neural.trainer import Trainer
        >>> from dense_armor.utility.neural.models import mlp
        >>> from dense_armor.utility.neural.losses import MSE
        >>> from dense_armor.utility.optim import SGD
        >>> m = mlp([2, 4, 1], jax.random.PRNGKey(0))
        >>> t = Trainer(m, MSE(), SGD(0.01))
        >>> X = jax.numpy.ones((4, 2))
        >>> y = jax.numpy.zeros((4, 1))
        >>> _ = t.fit(X, y, epochs=2, batch_size=2, key=jax.random.PRNGKey(0))
        >>> len(t.history_["train_loss"])
        2
    """

    def __init__(
        self,
        model: Any,
        loss: Any,
        optimizer: Any,
        schedule: Any = None,
        weight_decay: float = 0.0,
        decay: str = "l2",
        clip_norm: float | None = None,
        compile: bool = False,
    ) -> None:
        if decay not in ("l2", "decoupled"):
            raise ValueError(f"decay must be 'l2' or 'decoupled', got {decay!r}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
        self.model = model
        self.loss = loss
        self.optimizer_prototype = optimizer
        self.schedule = schedule if schedule is not None else ConstantLR(optimizer.lr)
        self.weight_decay = float(weight_decay)
        self.decay = decay
        if clip_norm is not None and clip_norm <= 0:
            raise ValueError(f"clip_norm must be > 0 or None, got {clip_norm}")
        self.clip_norm = clip_norm
        self.compile = bool(compile)
        self._compiled_values: list | None = None
        self._compiled_opt_state: Any = None
        self._compiled_step: Any = None
        self.optimizers_ = [copy.deepcopy(optimizer) for _ in model.params()]
        self.history_: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }
        self.best_epoch_: int | None = None
        self.stopped_epoch_: int | None = None

    def _apply_weight_decay(self, p: Param, epoch_t: float) -> None:
        if self.weight_decay <= 0 or p.value.ndim < 2:
            return
        if self.decay == "l2":
            p.grad = p.grad + self.weight_decay * p.value
        else:
            eta_t = self.schedule.multiplier(epoch_t)
            p.value = p.value - eta_t * self.weight_decay * p.value

    def fit(
        self,
        X: Any,
        y: Any,
        epochs: int,
        batch_size: int,
        key: Any,
        X_val: Any = None,
        y_val: Any = None,
        patience: int | None = None,
        min_delta: float = 0.0,
        shuffle: bool = True,
    ) -> Trainer:
        """Run epochs of training; return ``self``.

        Args:
            X, y: arrays of shape ``(n, ...)`` and ``(n, ...)``.
            epochs: the number of epochs, at least 1.
            batch_size: the mini-batch size.
            key: a ``jax.random`` key.
            X_val, y_val: optional validation set.
            patience: stop when the val_loss has not improved for this
                many epochs; requires the validation set.
            min_delta: the improvement threshold.
            shuffle: whether to shuffle each epoch.

        Raises:
            ValueError: on ``epochs < 1``, ``patience`` without a
                validation set, ``patience < 1``, or ``len(X) != len(y)``.
        """
        if epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {epochs}")
        if len(X) != len(y):
            raise ValueError(f"X and y have different lengths: {len(X)} and {len(y)}")
        if patience is not None:
            if X_val is None or y_val is None:
                raise ValueError("patience requires X_val and y_val")
            if patience < 1:
                raise ValueError(f"patience must be at least 1, got {patience}")

        self.history_ = {"train_loss": [], "val_loss": [], "lr": []}
        self.best_epoch_ = None
        self.stopped_epoch_ = None
        best_val = float("inf")
        best_weights: list[np.ndarray] | None = None
        no_improve = 0
        n = len(X)

        if self.compile:
            self._compiled_values = read_values(self.model)
            if self._compiled_step is None:
                self._compiled_opt_state = init_state(
                    self.optimizer_prototype, self._compiled_values
                )
                self._compiled_step = make_step(
                    self.model,
                    self.loss,
                    self.optimizer_prototype,
                    self.schedule,
                    self.weight_decay,
                    self.decay,
                    self.clip_norm,
                )
            values = self._compiled_values
            opt_state = self._compiled_opt_state
            step = self._compiled_step
        else:
            values = None
            opt_state = None
            step = None

        for epoch in range(epochs):
            if shuffle:
                key, sub = jax.random.split(key)
            else:
                sub = None
            batches = list(minibatches(n, batch_size, key=sub))
            n_batches = len(batches)
            first_lr = self.schedule(epoch + 0.0 / n_batches)
            if not self.compile:
                for opt in self.optimizers_:
                    opt.lr = first_lr
            total_loss = 0.0
            total_size = 0
            total_loss_device: Any = jnp.float32(0.0) if self.compile else None
            for b_idx, idx in enumerate(batches):
                epoch_t = epoch + b_idx / n_batches
                if self.compile:
                    values, opt_state, batch_loss = step(
                        values, opt_state, X[idx], y[idx], epoch_t
                    )
                    self._compiled_values = values
                    self._compiled_opt_state = opt_state
                    write_values(self.model, values)
                    total_loss_device = total_loss_device + batch_loss * len(idx)
                    total_size += len(idx)
                else:
                    lr_t = self.schedule(epoch_t)
                    for opt in self.optimizers_:
                        opt.lr = lr_t
                    for p in self.model.params():
                        p.zero_grad()
                    pred = self.model.forward(X[idx], training=True)
                    batch_loss = self.loss.forward(pred, y[idx])
                    self.model.backward(self.loss.backward())
                    if self.clip_norm is not None:
                        clip_grad_norm(self.model.params(), self.clip_norm)
                    for p, opt in zip(self.model.params(), self.optimizers_):
                        self._apply_weight_decay(p, epoch_t)
                        p.update(opt)
                    total_loss += float(batch_loss) * len(idx)
                    total_size += len(idx)
            if self.compile:
                assert total_loss_device is not None
                train_loss = float(total_loss_device) / total_size
            else:
                train_loss = total_loss / total_size
            self.history_["train_loss"].append(train_loss)
            self.history_["lr"].append(first_lr)
            if X_val is not None and y_val is not None:
                val_loss = self.evaluate(X_val, y_val)
            else:
                val_loss = float("nan")
            self.history_["val_loss"].append(val_loss)
            if patience is not None:
                if val_loss < best_val - min_delta:
                    best_val = val_loss
                    if self.compile:
                        assert values is not None
                        best_weights = [np.asarray(v).copy() for v in values]
                    else:
                        best_weights = [
                            np.asarray(p.value).copy() for p in self.model.params()
                        ]
                    self.best_epoch_ = epoch
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        self.stopped_epoch_ = epoch
                        break
        if patience is not None and best_weights is not None:
            for p, w in zip(self.model.params(), best_weights):
                p.value = jnp.asarray(w)
        if self.stopped_epoch_ is None:
            self.stopped_epoch_ = epochs - 1
        return self

    def evaluate(self, X: Any, y: Any) -> float:
        """Return the loss of the whole set with ``training=False``."""
        pred = self.model.forward(X, training=False)
        return float(self.loss.forward(pred, y))

    def predict(self, X: Any) -> jax.Array:
        """Return ``model.forward(X, training=False)``."""
        return self.model.forward(X, training=False)
