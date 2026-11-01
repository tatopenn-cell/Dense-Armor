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
    make_step_parallel,
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
        compute_dtype: str | None = None,
        loss_scale: float | str | None = None,
        devices: int | list[int] | None = None,
    ) -> None:
        """Build a trainer.

        ``compute_dtype`` and ``loss_scale`` select mixed-precision
        training on the compiled step (``compile=True``). The defaults
        ``None`` and ``None`` keep float32 and today's behaviour.

        The mixed-precision machinery follows Micikevicius et al. 2017,
        Section 3 (page 2): a float32 master copy of the weights and of
        the optimizer state, and the cast of the compute to the dtype
        for the forward and the backward. Section 3.2 (pages 4-5) is the
        source of the loss scaling and of the skip of a step whose
        gradient contains an inf or a nan. The paper describes a
        **constant** scaling factor (§3.2 page 4, "the simplest one is
        to pick a constant scaling factor") and the skip on overflow
        (§3.2 page 5, "one option is to skip the weight update when an
        overflow is detected").

        The reductions of ``LayerNorm``, of the softmax of the
        attention and of the softmax cross-entropy stay in the compute
        dtype, not in float32. Section 3.3 page 5 of the paper
        recommends float32 for "large reductions (sums across elements
        of a vector) ... batch-normalization ... softmax layers". This
        library keeps them in the compute dtype: the reductions are
        inside ``norm.py``, ``attention.py`` and ``losses.py``, which
        this argument does not change. Float32 reductions need a change in
        those files and are not implemented.

        The dynamic scaling (grow after a window of good steps, halve
        and skip on overflow) is **not in the paper**: the paper mentions
        only the constant factor and the skip. It is the behaviour of
        ``torch.cuda.amp.GradScaler`` (``growth_interval=2000``,
        ``init_scale=65536.0``, ``growth_factor=2.0``,
        ``backoff_factor=0.5``, PyTorch documentation) and of
        ``apex.amp.GradScaler`` (NVIDIA, same defaults), adopted here as
        the field's de-facto standard, not as an arbitrary choice. The
        paper does not mention bfloat16 either; bfloat16 has the same
        exponent range as float32, so the loss scaling is not needed
        with it (our statement, not a formula of the paper).

        Args:
            model: the model.
            loss: the loss.
            optimizer: the object form of the optimizer.
            schedule: the schedule; ``ConstantLR(optimizer.lr)`` if None.
            weight_decay: the weight decay coefficient.
            decay: ``"l2"`` or ``"decoupled"``.
            clip_norm: the global L2 threshold, or ``None``.
            compile: use the compiled step of item N1.
            compute_dtype: ``None`` (float32), ``"bfloat16"`` or
                ``"float16"``. Requires ``compile=True``.
            loss_scale: ``None`` (no scaling), a positive float (constant
                scale), or ``"dynamic"`` (grow after 2000 good steps,
                halve and skip on overflow, the PyTorch/Apex default).
                Requires ``compile=True``.
            devices: ``None`` (one device, the step as before), a number
                of devices (the first ones), or a list of device ids
                (indices into ``jax.devices()``). The global batch of
                ``fit`` is split evenly over them, each device computes
                the gradient of its shard, the gradients are averaged
                before the clipping and the update, and the weights and
                the optimizer state stay replicated. At least 2 devices.
                Requires ``compile=True``. It can be combined with
                ``compute_dtype`` and ``loss_scale``.

        Raises:
            ValueError: on ``decay`` outside ``("l2", "decoupled")``,
                negative ``weight_decay``, non-positive ``clip_norm``,
                ``compute_dtype`` outside ``None``, ``"bfloat16"``,
                ``"float16"``, ``loss_scale`` outside ``None``, a
                positive float or ``"dynamic"``, either of the two
                given with ``compile=False``, ``devices`` with
                ``compile=False``, fewer than 2 devices, more devices
                than ``jax.device_count()``, repeated or unknown device
                ids.
        """
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
        if compute_dtype is not None and compute_dtype not in (
            "bfloat16",
            "float16",
        ):
            raise ValueError(
                f"compute_dtype must be None, 'bfloat16' or 'float16', "
                f"got {compute_dtype!r}"
            )
        self.compute_dtype = compute_dtype
        if loss_scale is None:
            _ls: Any = None
        elif isinstance(loss_scale, str):
            if loss_scale != "dynamic":
                raise ValueError(
                    f"loss_scale must be None, a positive float, or 'dynamic', "
                    f"got {loss_scale!r}"
                )
            _ls = "dynamic"
        else:
            _ls = float(loss_scale)
            if _ls <= 0.0:
                raise ValueError(f"loss_scale must be positive, got {_ls}")
        self.loss_scale = _ls
        if (compute_dtype is not None or _ls is not None) and not self.compile:
            raise ValueError("compute_dtype and loss_scale require compile=True")
        self.devices: list[int] | None
        self._n_devices: int
        if devices is not None:
            if not self.compile:
                raise ValueError("devices requires compile=True")
            available = jax.device_count()
            ids = list(range(devices)) if isinstance(devices, int) else list(devices)
            if len(ids) < 2:
                raise ValueError(f"devices requires at least 2 devices, got {len(ids)}")
            if len(ids) > available:
                raise ValueError(
                    f"devices={len(ids)} exceeds the available count {available}"
                )
            if len(set(ids)) != len(ids):
                raise ValueError(f"devices has repeated entries: {ids}")
            bad = [i for i in ids if not 0 <= i < available]
            if bad:
                raise ValueError(
                    f"device ids {bad} are not among the {available} available devices"
                )
            self.devices = ids
            self._n_devices = len(ids)
        else:
            self.devices = None
            self._n_devices = 1
        self._scale: float
        if isinstance(_ls, float):
            self._scale = _ls
        elif _ls == "dynamic":
            self._scale = float(2.0**16)
        else:
            self._scale = 1.0
        self._good_steps: int = 0
        self._compiled_values: list | None = None
        self._compiled_opt_state: Any = None
        self._compiled_step: Any = None
        self._compiled_mixed: Any = None
        self._compiled_parallel: Any = None
        self._compiled_parallel_mixed: Any = None
        self.optimizers_ = [copy.deepcopy(optimizer) for _ in model.params()]
        self.history_: dict[str, list[float]] = self._new_history()
        self.best_epoch_: int | None = None
        self.stopped_epoch_: int | None = None

    def _new_history(self) -> dict[str, list[float]]:
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "lr": []}
        if self.loss_scale is not None:
            history["loss_scale"] = []
        return history

    def _update_scale(self, finite: bool) -> None:
        """Dynamic loss scale: halve on an overflow, double after 2000 good steps."""
        if not finite:
            self._scale = max(self._scale / 2.0, 1.0)
            self._good_steps = 0
            return
        self._good_steps += 1
        if self._good_steps >= 2000:
            self._scale = self._scale * 2.0
            self._good_steps = 0

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
                validation set, ``patience < 1``, ``len(X) != len(y)``, or,
                with ``devices``, a ``batch_size`` or a last batch that the
                number of devices does not divide.
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

        self.history_ = self._new_history()
        self.best_epoch_ = None
        self.stopped_epoch_ = None
        best_val = float("inf")
        best_weights: list[np.ndarray] | None = None
        no_improve = 0
        n = len(X)

        mixed = self.compile and (
            self.compute_dtype is not None or self.loss_scale is not None
        )

        parallel = self.compile and self.devices is not None
        if parallel:
            n_dev = self._n_devices
            if batch_size % n_dev != 0:
                raise ValueError(
                    f"batch_size {batch_size} is not divisible by devices {n_dev}"
                )
            tail = n % batch_size
            if tail % n_dev != 0:
                raise ValueError(
                    f"the last batch has {tail} samples, not divisible by "
                    f"devices {n_dev}: use a number of samples that leaves a "
                    "multiple of the number of devices"
                )

        if self.compile:
            _dtype = jnp.float32 if mixed else None
            self._compiled_values = read_values(self.model, dtype=_dtype)
            if parallel:
                if (
                    self._compiled_parallel is None
                    and self._compiled_parallel_mixed is None
                ):
                    self._compiled_opt_state = init_state(
                        self.optimizer_prototype, self._compiled_values
                    )
                    if mixed:
                        self._compiled_parallel_mixed = make_step_parallel(
                            self.model,
                            self.loss,
                            self.optimizer_prototype,
                            self.schedule,
                            self.weight_decay,
                            self.decay,
                            self.clip_norm,
                            self._n_devices,
                            compute_dtype=self.compute_dtype,
                            loss_scale=self.loss_scale,
                            devices=self.devices,
                        )
                    else:
                        self._compiled_parallel = make_step_parallel(
                            self.model,
                            self.loss,
                            self.optimizer_prototype,
                            self.schedule,
                            self.weight_decay,
                            self.decay,
                            self.clip_norm,
                            self._n_devices,
                            devices=self.devices,
                        )
                n_dev = self._n_devices
                values = jax.tree_util.tree_map(
                    lambda v: jnp.broadcast_to(
                        jnp.asarray(v)[None], (n_dev,) + jnp.shape(v)
                    ),
                    self._compiled_values,
                )
                opt_state = jax.tree_util.tree_map(
                    lambda s: jnp.broadcast_to(
                        jnp.asarray(s)[None], (n_dev,) + jnp.shape(s)
                    ),
                    self._compiled_opt_state,
                )
                step = None
                step_parallel = self._compiled_parallel
                step_parallel_mixed = self._compiled_parallel_mixed
            elif self._compiled_step is None:
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
                    compute_dtype=self.compute_dtype,
                    loss_scale=self.loss_scale,
                )
                if mixed:
                    self._compiled_mixed = self._compiled_step.mixed
            if not parallel:
                values = self._compiled_values
                opt_state = self._compiled_opt_state
                step = self._compiled_step
                step_parallel = None
                step_parallel_mixed = None
        else:
            values = None
            opt_state = None
            step = None
            step_parallel = None
            step_parallel_mixed = None

        def sync_parallel() -> None:
            """Write the weights of the first device into the model, on the host."""
            values_plain = jax.tree_util.tree_map(lambda v: np.asarray(v[0]), values)
            opt_plain = jax.tree_util.tree_map(lambda s: np.asarray(s[0]), opt_state)
            self._compiled_values = values_plain
            self._compiled_opt_state = opt_plain
            if mixed:
                write_values(self.model, values_plain, dtype=jnp.float32)
            else:
                write_values(self.model, values_plain)

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
                if step_parallel_mixed is not None:
                    n_dev = self._n_devices
                    shard = len(idx) // n_dev
                    x_s = jnp.asarray(X[idx]).reshape(
                        (n_dev, shard) + tuple(X.shape[1:])
                    )
                    y_s = jnp.asarray(y[idx]).reshape(
                        (n_dev, shard) + tuple(y.shape[1:])
                    )
                    values, opt_state, batch_loss_d, finite_d = step_parallel_mixed(
                        values, opt_state, x_s, y_s, epoch_t, self._scale
                    )
                    if self.loss_scale == "dynamic":
                        self._update_scale(bool(finite_d[0]))
                    total_loss_device = total_loss_device + batch_loss_d[0] * len(idx)
                    total_size += len(idx)
                elif step_parallel is not None:
                    n_dev = self._n_devices
                    shard = len(idx) // n_dev
                    x_s = jnp.asarray(X[idx]).reshape(
                        (n_dev, shard) + tuple(X.shape[1:])
                    )
                    y_s = jnp.asarray(y[idx]).reshape(
                        (n_dev, shard) + tuple(y.shape[1:])
                    )
                    values, opt_state, batch_loss_d = step_parallel(
                        values, opt_state, x_s, y_s, epoch_t
                    )
                    total_loss_device = total_loss_device + batch_loss_d[0] * len(idx)
                    total_size += len(idx)
                elif mixed:
                    values, opt_state, batch_loss, finite = self._compiled_mixed(
                        values, opt_state, X[idx], y[idx], epoch_t, self._scale
                    )
                    self._compiled_values = values
                    self._compiled_opt_state = opt_state
                    write_values(self.model, values, dtype=jnp.float32)
                    if self.loss_scale == "dynamic":
                        self._update_scale(bool(finite))
                    total_loss_device = total_loss_device + batch_loss * len(idx)
                    total_size += len(idx)
                elif self.compile:
                    assert step is not None
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
            if (step_parallel is not None or step_parallel_mixed is not None) and (
                X_val is not None and y_val is not None
            ):
                sync_parallel()
            if self.compile:
                assert total_loss_device is not None
                train_loss = float(total_loss_device) / total_size
            else:
                train_loss = total_loss / total_size
            self.history_["train_loss"].append(train_loss)
            self.history_["lr"].append(first_lr)
            if self.loss_scale is not None:
                self.history_["loss_scale"].append(self._scale)
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
                        if step_parallel is not None or step_parallel_mixed is not None:
                            best_weights = [np.asarray(v[0]).copy() for v in values]
                        else:
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
        if step_parallel is not None or step_parallel_mixed is not None:
            sync_parallel()
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
