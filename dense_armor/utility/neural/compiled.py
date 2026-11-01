"""A compiled training step: forward, loss, backward, weight decay,
clipping and the optimizer update under one ``jax.jit``.

``Trainer(..., compile=True)`` uses :func:`make_step` in place of the
Python loop of :meth:`Trainer.fit`. The default ``compile=False`` is
the loop of :mod:`dense_armor.utility.neural.trainer` and its results
do not change.

The step is a pure function of its arguments: the values of the
parameters are passed as a list of ``jax`` arrays (one per ``Param``),
the optimizer state is a pytree with the same shape, and the function
returns the new values, the new state and the loss. The model is a
Python object captured by the closure; the layer caches it writes in
``forward`` are local to the traced function and do not cross the
``jit`` boundary (item 0, ``tools/audit_n1.py``).

The caller must write the new values back into the parameters with
:func:`write_values` after every step. Inside the trace the
``Param.value`` of each parameter is a tracer; if it is not replaced
with a concrete array between two calls, JAX raises
``UnexpectedTracerError`` on the second call. The write is cheap and
happens once per batch.

The arithmetic of every optimizer is the one of its object form
in :mod:`dense_armor.utility.optim` (``step(weights, gradient)``).
The decoupled weight decay is the one of Algorithm 1 (line 9, page 3)
and Algorithm 2 (line 12, page 3) of Loshchilov and Hutter 2019: the
term subtracted from the value is ``eta_t * lambda * theta_{t-1}``,
where ``eta_t`` is the schedule multiplier and ``lambda`` the weight
decay coefficient, not ``alpha * lambda * theta``. Proposition 1, page
2, states that with standard SGD the L2 form and the decoupled form
are equivalent when the L2 coefficient is ``lambda / alpha``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


@dataclass
class CompiledStep:
    """The compiled steps of a Trainer.

    Attributes:
        plain: ``(values, opt_state, x, y, epoch_t) -> (values, opt_state, loss)``.
        mixed: ``(values, opt_state, x, y, epoch_t, scale) -> (values,
            opt_state, loss, finite)``, the whole mixed-precision step in
            one compiled function, or ``None`` when the trainer is not in
            mixed-precision mode.
        grads: ``(values, x, y, scale) -> (grads_fp16, grads_fp32, loss)``,
            the gradient part of the mixed step alone, or ``None``.
    """

    plain: Callable | None
    mixed: Callable | None
    grads: Callable | None = None
    mixed_devices_inner: Callable | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the plain step; the mixed step is reached through ``mixed``."""
        if self.plain is None:
            raise TypeError("this step is mixed: call .mixed(...)")
        return self.plain(*args, **kwargs)


_DTYPE_MAP = {
    "bfloat16": jnp.bfloat16,
    "float16": jnp.float16,
}


def _cast(x, dtype):
    """Cast a list of arrays to a compute dtype (None = identity)."""
    if dtype is None:
        return x
    if isinstance(x, list):
        return [_cast(e, dtype) for e in x]
    return jnp.asarray(x, dtype=dtype)


def _optimizer_name(optimizer: Any) -> str:
    return type(optimizer).__name__


def _init_state(optimizer: Any, values: list) -> Any:
    """Return the initial optimizer state as a pytree of arrays."""
    name = _optimizer_name(optimizer)
    if name == "SGD":
        if optimizer.momentum == 0.0:
            return None
        return [jnp.zeros_like(v) for v in values]
    if name == "Adam":
        return {
            "m": [jnp.zeros_like(v) for v in values],
            "v": [jnp.zeros_like(v) for v in values],
            "t": jnp.int32(0),
        }
    if name == "RMSProp":
        return [jnp.zeros_like(v) for v in values]
    if name == "AdaGrad":
        return [jnp.zeros_like(v) for v in values]
    raise ValueError(f"compile=True does not support optimizer {name}")


def _apply_update(
    optimizer: Any,
    values: list,
    grads: list,
    opt_state: Any,
    lr: Any,
    eta: Any,
    weight_decay: Any,
    decay: str,
) -> tuple[list, Any]:
    """One step of the vectorised optimizer over all parameters.

    ``lr`` is the learning rate at this step, ``schedule(t)``;
    ``eta`` is the schedule multiplier at this step,
    ``schedule.multiplier(t)``. The two are the same for a constant
    schedule and differ when the schedule decays.

    The weight decay is applied first. ``decay="l2"`` adds
    ``weight_decay * value`` to every gradient whose value has two or
    more dimensions (Algorithm 1, line 6, page 3 of Loshchilov and
    Hutter 2019, as the Trainer does). ``decay="decoupled"`` subtracts
    ``eta * weight_decay * value`` from the value before the update:
    Algorithm 1, line 9, for SGDW, and Algorithm 2, line 12, for
    AdamW. The scale is ``eta`` and not ``lr``: the paper separates the
    schedule multiplier ``eta_t`` from the base learning rate
    ``alpha``, and the decoupled term is ``eta_t * lambda * theta``.
    """
    name = _optimizer_name(optimizer)
    if weight_decay > 0.0 and decay == "l2":
        grads = [
            g + weight_decay * v if v.ndim >= 2 else g for v, g in zip(values, grads)
        ]
    if weight_decay > 0.0 and decay == "decoupled":
        values = [v - eta * weight_decay * v if v.ndim >= 2 else v for v in values]
    if name == "SGD":
        if optimizer.momentum == 0.0:
            new_values = [v - lr * g for v, g in zip(values, grads)]
            return new_values, None
        u_old = opt_state
        u_new = [optimizer.momentum * u + lr * g for u, g in zip(u_old, grads)]
        if optimizer.nesterov:
            new_values = [
                v - lr * g - optimizer.momentum * u
                for v, g, u in zip(values, grads, u_new)
            ]
        else:
            new_values = [v - u for v, u in zip(values, u_new)]
        return new_values, u_new
    if name == "Adam":
        t = opt_state["t"] + 1
        m = [
            optimizer.beta1 * mi + (1.0 - optimizer.beta1) * g
            for mi, g in zip(opt_state["m"], grads)
        ]
        v = [
            optimizer.beta2 * vi + (1.0 - optimizer.beta2) * g * g
            for vi, g in zip(opt_state["v"], grads)
        ]
        b1t = 1.0 - jnp.float_power(optimizer.beta1, t)
        b2t = 1.0 - jnp.float_power(optimizer.beta2, t)
        new_values = [
            val - lr * (m_i / b1t) / (jnp.sqrt(v_i / b2t) + optimizer.eps)
            for val, m_i, v_i in zip(values, m, v)
        ]
        return new_values, {"m": m, "v": v, "t": t}
    if name == "RMSProp":
        e_new = [
            optimizer.rho * ei + (1.0 - optimizer.rho) * g * g
            for ei, g in zip(opt_state, grads)
        ]
        new_values = [
            val - lr * g / jnp.sqrt(ei + optimizer.eps)
            for val, g, ei in zip(values, grads, e_new)
        ]
        return new_values, e_new
    if name == "AdaGrad":
        g_new = [gi + g * g for gi, g in zip(opt_state, grads)]
        new_values = [
            val - lr * g / (jnp.sqrt(gi) + optimizer.eps)
            for val, g, gi in zip(values, grads, g_new)
        ]
        return new_values, g_new
    raise ValueError(f"compile=True does not support optimizer {name}")


def _write_values(model: Any, values: list, dtype: Any = None) -> None:
    """Set ``p.value`` of every ``Param`` from the flat list.

    Args:
        model: the model.
        values: the flat list of arrays.
        dtype: an optional jax dtype; if given, every value is cast.
    """
    for p, v in zip(model.params(), values):
        p.value = jnp.asarray(v, dtype=dtype) if dtype is not None else jnp.asarray(v)


def _read_values(model: Any, dtype: Any = None) -> list:
    """Return the current ``p.value`` of every ``Param`` as a flat list.

    Args:
        model: the model.
        dtype: an optional jax dtype; if given, every value is cast.
    """
    if dtype is not None:
        return [jnp.asarray(p.value, dtype=dtype) for p in model.params()]
    return [jnp.asarray(p.value) for p in model.params()]


def _read_grads(model: Any) -> list:
    """Return the current ``p.grad`` of every ``Param`` as a flat list."""
    return [jnp.asarray(p.grad) for p in model.params()]


def _clip_grads(grads: list, max_norm: float) -> list:
    """Global L2 clipping, the same arithmetic as ``clip_grad_norm``."""
    total_sq = 0.0
    for g in grads:
        total_sq = total_sq + jnp.sum(g * g)
    norm = jnp.sqrt(total_sq)
    scale = jnp.where(norm >= max_norm, max_norm / norm, 1.0)
    return [g * scale for g in grads]


def _stateful_blocker(model: Any) -> str | None:
    """Return the class name of the first stateful layer, or None.

    Dropout carries a PRNG key that advances at every training forward.
    BatchNorm carries a running mean and variance that change at every
    training forward. Both states must cross the jit boundary, so a
    compiled step on a model that contains them is not supported. The
    function walks the layer tree through the attributes that the
    models use (layers, inner, attn, ln1, ln2, ffn1, ffn2, dropout_attn,
    dropout_ffn).
    """
    stateful = {"Dropout", "BatchNorm", "GroupNorm", "InstanceNorm"}
    visited = set()

    def walk(layer: Any) -> str | None:
        if id(layer) in visited:
            return None
        visited.add(id(layer))
        cls = type(layer).__name__
        if cls in stateful:
            return cls
        children: list[Any] = []
        inner = getattr(layer, "layers", None) or []
        children.extend(inner)
        single = getattr(layer, "inner", None)
        if single is not None:
            children.append(single)
        for name in (
            "attn",
            "ln1",
            "ln2",
            "ffn1",
            "ffn2",
            "dropout_attn",
            "dropout_ffn",
        ):
            child = getattr(layer, name, None)
            if child is not None:
                children.append(child)
        for child in children:
            found = walk(child)
            if found is not None:
                return found
        return None

    return walk(model)


def make_step(
    model: Any,
    loss: Any,
    optimizer: Any,
    schedule: Any,
    weight_decay: float,
    decay: str,
    clip_norm: float | None,
    compute_dtype: str | None = None,
    loss_scale: float | str | None = None,
) -> CompiledStep:
    """Return the compiled step: ``step(values, opt_state, x, y, epoch_t)``.

    The inner function is jitted. ``lr`` and ``eta`` are computed
    outside the jit, from a Python float ``epoch_t``: the schedule has
    a Python ``if`` on the epoch (a negative epoch raises
    ``ValueError``), and a traced value cannot be tested that way.
    The loss is returned as a JAX scalar and read once per epoch by
    the caller.

    Args:
        model: the model, with ``forward``, ``backward`` and ``params``.
        loss: the loss, with ``forward`` and ``backward``.
        optimizer: the object form of an optimizer.
        schedule: the schedule.
        weight_decay: the weight decay coefficient.
        decay: ``"l2"`` or ``"decoupled"``.
        clip_norm: the global L2 threshold, or ``None``.
        compute_dtype: ``None``, ``"bfloat16"`` or ``"float16"``: the dtype
            of the forward and the backward; the weights are cast to it
            and the master copy stays in float32.
        loss_scale: ``None`` for the plain step. Any other value selects
            the mixed step; the factor itself is the ``scale`` argument
            of ``mixed``, so it can change between steps.

    Returns:
        A ``CompiledStep``. Called as ``step(values, opt_state, x, y,
        epoch_t)`` it returns ``(new_values, new_opt_state, loss_value)``.
        When ``compute_dtype`` or ``loss_scale`` is given, ``step.mixed(
        values, opt_state, x, y, epoch_t, scale)`` is the whole step in one
        compiled function: forward and backward in the compute dtype, the
        gradients divided by ``scale`` and clipped in float32, the
        optimizer update, and the skip of the step when a gradient is not
        finite. It returns ``(new_values, new_opt_state, loss, finite)``;
        when ``finite`` is false the values and the state are the old
        ones. ``step.grads(values, x, y, scale)`` returns ``(grads_fp16,
        grads_fp32, loss)``: the scaled gradients in the compute dtype and
        the unscaled, clipped gradients in float32.

    Raises:
        ValueError: on an unsupported optimizer, on a model with state
            that must cross the jit boundary, or on ``compute_dtype``
            outside ``None``, ``"bfloat16"``, ``"float16"``.
    """
    name = _optimizer_name(optimizer)
    blocker = _stateful_blocker(model)
    if blocker is not None:
        raise ValueError(
            f"compile=True does not support {blocker}: its state advances "
            "at every training forward and must cross the jit boundary; "
            "see the item 0 audit."
        )

    def _step(
        values: list, opt_state: Any, x: Any, y: Any, lr: Any, eta: Any
    ) -> tuple[list, Any, Any]:
        _write_values(model, values)
        for p in model.params():
            p.zero_grad()
        pred = model.forward(x, training=True)
        loss_value = loss.forward(pred, y)
        model.backward(loss.backward())
        grads = _read_grads(model)
        if clip_norm is not None:
            grads = _clip_grads(grads, clip_norm)
        new_values, new_opt_state = _apply_update(
            optimizer, values, grads, opt_state, lr, eta, weight_decay, decay
        )
        return new_values, new_opt_state, loss_value

    if name not in ("SGD", "Adam", "RMSProp", "AdaGrad"):
        raise ValueError(f"compile=True does not support optimizer {name}")

    if compute_dtype is None and loss_scale is None:
        jitted = jax.jit(_step)

        def plain_step(
            values: list, opt_state: Any, x: Any, y: Any, epoch_t: float
        ) -> tuple[list, Any, Any]:
            tt = float(epoch_t)
            lr = schedule(tt)
            eta = schedule.multiplier(tt)
            return jitted(values, opt_state, x, y, lr, eta)

        return CompiledStep(plain=plain_step, mixed=None)

    if compute_dtype is not None and compute_dtype not in _DTYPE_MAP:
        raise ValueError(
            f"compute_dtype must be None, 'bfloat16' or 'float16', "
            f"got {compute_dtype!r}"
        )
    dtype = _DTYPE_MAP[compute_dtype] if compute_dtype is not None else None

    def _grads_inner(values: list, x: Any, y: Any, scale: Any, axis_name):
        cast_values = _cast(values, dtype)
        _write_values(model, cast_values)
        for pp in model.params():
            pp.zero_grad()
        pred = model.forward(_cast(x, dtype), training=True)
        loss_value = loss.forward(pred, y)
        loss_value = jnp.asarray(loss_value, dtype=jnp.float32)
        grad = loss.backward()
        grad = grad * scale
        model.backward(grad)
        raw = _read_grads(model)
        stored = [jnp.asarray(g, dtype=dtype) for g in raw]
        finite_local = jnp.all(jnp.stack([jnp.all(jnp.isfinite(g)) for g in stored]))
        as32 = [jnp.asarray(g, dtype=jnp.float32) for g in stored]
        if axis_name is not None:
            as32 = [jax.lax.pmean(g, axis_name) for g in as32]
            finite = jax.lax.pmin(finite_local, axis_name)
            loss_value = jax.lax.pmean(loss_value, axis_name)
        else:
            finite = finite_local
        unscaled = [g / scale for g in as32]
        if clip_norm is not None:
            unscaled = _clip_grads(unscaled, clip_norm)
        return stored, unscaled, loss_value, finite

    def _grads(values: list, x: Any, y: Any, scale: Any):
        stored, unscaled, loss_value, _finite = _grads_inner(
            values, x, y, scale, axis_name=None
        )
        return stored, unscaled, loss_value

    def _mixed_step_inner(
        values: list,
        opt_state: Any,
        x: Any,
        y: Any,
        lr: Any,
        eta: Any,
        scale: Any,
        axis_name,
    ):
        _stored, grads_fp32, loss_value, finite = _grads_inner(
            values, x, y, scale, axis_name
        )
        new_values, new_state = _apply_update(
            optimizer, values, grads_fp32, opt_state, lr, eta, weight_decay, decay
        )
        kept_values = [jnp.where(finite, n, v) for n, v in zip(new_values, values)]
        kept_state = jax.tree_util.tree_map(
            lambda n, o: jnp.where(finite, n, o), new_state, opt_state
        )
        return kept_values, kept_state, loss_value, finite

    def _mixed_step(
        values: list, opt_state: Any, x: Any, y: Any, lr: Any, eta: Any, scale: Any
    ):
        return _mixed_step_inner(
            values, opt_state, x, y, lr, eta, scale, axis_name=None
        )

    grads_jitted = jax.jit(_grads)
    mixed_jitted = jax.jit(_mixed_step)

    def _grads_public(values: list, x: Any, y: Any, scale: float):
        return grads_jitted(values, x, y, jnp.float32(scale))

    def _mixed(
        values: list, opt_state: Any, x: Any, y: Any, epoch_t: float, scale: float
    ):
        tt = float(epoch_t)
        lr = schedule(tt)
        eta = schedule.multiplier(tt)
        return mixed_jitted(values, opt_state, x, y, lr, eta, jnp.float32(scale))

    def _mixed_devices_inner_bound(
        values: list, opt_state: Any, x: Any, y: Any, lr: Any, eta: Any, scale: Any
    ):
        return _mixed_step_inner(
            values, opt_state, x, y, lr, eta, scale, axis_name="devices"
        )

    return CompiledStep(
        plain=None,
        mixed=_mixed,
        grads=_grads_public,
        mixed_devices_inner=_mixed_devices_inner_bound,
    )


@dataclass
class AccumulatingStep:
    """The compiled pieces of a step that accumulates the gradient.

    Attributes:
        grads: ``(values, x, y, scale) -> (grads, loss, finite)``: the
            gradient of one micro-batch, already averaged over the devices
            and divided by ``scale`` when the step is mixed, with the loss
            and the flag that says every gradient is finite. With devices,
            ``values``, ``x`` and ``y`` carry the device axis.
        add: ``(a, b) -> a + b`` over two lists of gradients.
        update: ``(values, opt_state, grads_sum, count, finite, epoch_t) ->
            (values, opt_state)``: the mean of ``count`` accumulated
            gradients, clipped, then the optimizer update; when ``finite``
            is false the old values and the old state are kept.
    """

    grads: Callable
    add: Callable
    update: Callable


def make_accumulating_step(
    model: Any,
    loss: Any,
    optimizer: Any,
    schedule: Any,
    weight_decay: float,
    decay: str,
    clip_norm: float | None,
    n_devices: int = 1,
    compute_dtype: str | None = None,
    loss_scale: float | str | None = None,
    devices: list[int] | None = None,
) -> AccumulatingStep:
    """Return the compiled pieces of gradient accumulation.

    ``k`` micro-batches are run with ``grads`` and summed with ``add``;
    ``update`` divides the sum by ``k`` (the loss is divided by ``k``),
    clips the mean gradient, and applies the optimizer once. With equal
    micro-batches the result is the step of one batch of ``k`` times the
    size.

    With ``n_devices > 1`` each micro-batch is split over the devices and
    its gradients are averaged with ``pmean`` before they are summed. With
    ``compute_dtype`` or ``loss_scale`` the micro-batch is run in the
    mixed way: the gradients are stored in the compute dtype, cast to
    float32, averaged, divided by the scale of that step, and the flag
    ``finite`` is the ``pmin`` of the flags of the devices. The flags of
    the micro-batches of a group are combined with ``and``, so one
    overflow skips the whole group: a sum that holds an ``inf`` cannot be
    recovered.

    Args:
        model: the model, with ``forward``, ``backward`` and ``params``.
        loss: the loss, with ``forward`` and ``backward``.
        optimizer: the object form of an optimizer.
        schedule: the schedule.
        weight_decay: the weight decay coefficient.
        decay: ``"l2"`` or ``"decoupled"``.
        clip_norm: the global L2 threshold, or ``None``.
        n_devices: the number of devices; 1 for a single device.
        compute_dtype: ``None``, ``"bfloat16"`` or ``"float16"``.
        loss_scale: ``None``, a positive float or ``"dynamic"``; any value
            selects the mixed micro-batch.
        devices: ids of the devices (indices into ``jax.devices()``) when
            ``n_devices > 1``; ``None`` takes the first ones.

    Returns:
        An ``AccumulatingStep``.

    Raises:
        ValueError: on an unsupported optimizer, on a model with state
            that must cross the jit boundary, or on ``compute_dtype``
            outside ``None``, ``"bfloat16"``, ``"float16"``.
    """
    name = _optimizer_name(optimizer)
    if name not in ("SGD", "Adam", "RMSProp", "AdaGrad"):
        raise ValueError(f"compile=True does not support optimizer {name}")
    blocker = _stateful_blocker(model)
    if blocker is not None:
        raise ValueError(
            f"compile=True does not support {blocker}: its state advances "
            "at every training forward and must cross the jit boundary."
        )
    if compute_dtype is not None and compute_dtype not in _DTYPE_MAP:
        raise ValueError(
            f"compute_dtype must be None, 'bfloat16' or 'float16', got {compute_dtype!r}"
        )
    mixed = compute_dtype is not None or loss_scale is not None
    dtype = _DTYPE_MAP[compute_dtype] if compute_dtype is not None else None
    axis = "devices" if n_devices > 1 else None

    def _grads(values: list, x: Any, y: Any, scale: Any) -> tuple[list, Any, Any]:
        _write_values(model, _cast(values, dtype))
        for p in model.params():
            p.zero_grad()
        pred = model.forward(_cast(x, dtype), training=True)
        loss_value = loss.forward(pred, y)
        grad = loss.backward()
        if mixed:
            loss_value = jnp.asarray(loss_value, dtype=jnp.float32)
            grad = grad * scale
        model.backward(grad)
        raw = _read_grads(model)
        if mixed:
            stored = [jnp.asarray(g, dtype=dtype) for g in raw]
            finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(g)) for g in stored]))
            grads = [jnp.asarray(g, dtype=jnp.float32) for g in stored]
        else:
            finite = jnp.bool_(True)
            grads = raw
        if axis is not None:
            grads = [jax.lax.pmean(g, axis) for g in grads]
            finite = jax.lax.pmin(finite, axis)
            loss_value = jax.lax.pmean(loss_value, axis)
        if mixed:
            grads = [g / scale for g in grads]
        return grads, loss_value, finite

    def _add(a: list, b: list) -> list:
        return [u + v for u, v in zip(a, b)]

    def _update(
        values: list,
        opt_state: Any,
        grads_sum: list,
        count: Any,
        finite: Any,
        lr: Any,
        eta: Any,
    ) -> tuple[list, Any]:
        grads = [g / count for g in grads_sum]
        if clip_norm is not None:
            grads = _clip_grads(grads, clip_norm)
        new_values, new_state = _apply_update(
            optimizer, values, grads, opt_state, lr, eta, weight_decay, decay
        )
        kept_values = [jnp.where(finite, n, v) for n, v in zip(new_values, values)]
        kept_state = jax.tree_util.tree_map(
            lambda n, o: jnp.where(finite, n, o), new_state, opt_state
        )
        return kept_values, kept_state

    if axis is None:
        grads_fn: Callable = jax.jit(_grads)
        update_fn: Callable = jax.jit(_update)
    else:
        pmap_devices = None if devices is None else [jax.devices()[i] for i in devices]
        grads_fn = jax.pmap(
            _grads, axis_name=axis, in_axes=(0, 0, 0, None), devices=pmap_devices
        )
        update_fn = jax.pmap(
            _update,
            axis_name=axis,
            in_axes=(0, 0, 0, None, 0, None, None),
            devices=pmap_devices,
        )

    def grads(values: list, x: Any, y: Any, scale: float) -> tuple[list, Any, Any]:
        return grads_fn(values, x, y, jnp.float32(scale))

    def update(
        values: list,
        opt_state: Any,
        grads_sum: list,
        count: int,
        finite: Any,
        epoch_t: float,
    ) -> tuple[list, Any]:
        tt = float(epoch_t)
        return update_fn(
            values,
            opt_state,
            grads_sum,
            jnp.float32(count),
            finite,
            schedule(tt),
            schedule.multiplier(tt),
        )

    return AccumulatingStep(grads=grads, add=jax.jit(_add), update=update)


def make_step_parallel(
    model: Any,
    loss: Any,
    optimizer: Any,
    schedule: Any,
    weight_decay: float,
    decay: str,
    clip_norm: float | None,
    n_devices: int,
    compute_dtype: str | None = None,
    loss_scale: float | str | None = None,
    devices: list[int] | None = None,
) -> Callable:
    """Return the data-parallel compiled step over ``n_devices`` devices.

    The global batch is split evenly over the devices. When
    ``compute_dtype`` and ``loss_scale`` are both ``None`` the plain
    gradient path is used. Otherwise the mixed step of the single-device
    trainer runs with the per-device gradients averaged in float32 and the
    overflow flag reduced with ``pmin``, so that a step skipped on one
    device is skipped on all of them and the replicated weights stay
    equal.

    Args:
        model: the model, with ``forward``, ``backward`` and ``params``.
        loss: the loss, with ``forward`` and ``backward``.
        optimizer: the object form of the optimizer.
        schedule: the schedule.
        weight_decay: the weight decay coefficient.
        decay: ``"l2"`` or ``"decoupled"``.
        clip_norm: the global L2 threshold, or ``None``.
        n_devices: the number of devices to use.
        compute_dtype: ``None``, ``"bfloat16"`` or ``"float16"``.
        loss_scale: ``None``, a positive float, or ``"dynamic"``.
        devices: ids of the devices (indices into ``jax.devices()``) the
            step runs on, ``n_devices`` of them; ``None`` takes the first
            ``n_devices``.

    Returns:
        ``step_parallel(values, opt_state, x, y, epoch_t)`` in the plain
        case, or ``step_parallel(values, opt_state, x, y, epoch_t, scale)``
        in the mixed case. ``x`` and ``y`` are already split along axis 0
        into ``n_devices`` shards. The plain path returns
        ``(values, opt_state, loss)``; the mixed path returns
        ``(values, opt_state, loss, finite)``. Every array carries the
        leading device axis.

    Raises:
        ValueError: on an unsupported optimizer, on a stateful model, or
            on ``n_devices < 2``.

    References:
        Goyal et al. (2017), "Accurate, Large Minibatch SGD". Section 2.1,
        page 3: setting ``η̂ = kη`` makes one large step approximate ``k``
        small steps (after equation (4)). Section 2.3, page 4, equation
        (7): ``ŵ_{t+1} = w_t - η̂ (1/k) Σ_{j<k} ∇L(B_j, w_t)``, the
        average of the gradients over ``k`` workers; on the same page,
        after equation (7): "Following similar logic as in §2.1, we set
        η̂ = kη and we keep the per-worker sample size n constant when we
        change the number of workers k." For the mixed step,
        Micikevicius et al. (2017) §3.1 page 2 (FP32 master copy) and
        pages 4-5 (loss scaling).

    Choices:
        The average of the per-device gradients in float32 of the values
        stored at ``compute_dtype``, the ``pmin`` of the ``finite`` flag
        across devices, and the ``pmean`` of the loss, are our combination
        of the two papers: Goyal covers the averaging over workers but not
        the mixed step, Micikevicius covers the mixed step but not the
        multi-device reduction. No paper prescribes the order of the two.
    """
    name = _optimizer_name(optimizer)
    if name not in ("SGD", "Adam", "RMSProp", "AdaGrad"):
        raise ValueError(f"compile=True does not support optimizer {name}")
    blocker = _stateful_blocker(model)
    if blocker is not None:
        raise ValueError(
            f"devices cannot compile a model with {blocker}: its state "
            "must cross the device boundary"
        )
    if n_devices < 2:
        raise ValueError(f"n_devices must be >= 2, got {n_devices}")
    pmap_devices = None if devices is None else [jax.devices()[i] for i in devices]

    if compute_dtype is None and loss_scale is None:

        def _step_parallel(
            values: list, opt_state: Any, x: Any, y: Any, lr: Any, eta: Any
        ) -> tuple[list, Any, Any]:
            _write_values(model, values)
            for p in model.params():
                p.zero_grad()
            pred = model.forward(x, training=True)
            loss_value = loss.forward(pred, y)
            model.backward(loss.backward())
            grads = _read_grads(model)
            grads = jax.lax.pmean(grads, axis_name="devices")
            loss_value = jax.lax.pmean(loss_value, axis_name="devices")
            if clip_norm is not None:
                grads = _clip_grads(grads, clip_norm)
            new_values, new_opt_state = _apply_update(
                optimizer, values, grads, opt_state, lr, eta, weight_decay, decay
            )
            return new_values, new_opt_state, loss_value

        pmapped = jax.pmap(
            _step_parallel,
            axis_name="devices",
            in_axes=(0, 0, 0, 0, None, None),
            devices=pmap_devices,
        )

        def step_parallel_plain(
            values: list, opt_state: Any, x: Any, y: Any, epoch_t: float
        ):
            tt = float(epoch_t)
            lr = schedule(tt)
            eta = schedule.multiplier(tt)
            return pmapped(values, opt_state, x, y, lr, eta)

        return step_parallel_plain

    compiled = make_step(
        model,
        loss,
        optimizer,
        schedule,
        weight_decay,
        decay,
        clip_norm,
        compute_dtype=compute_dtype,
        loss_scale=loss_scale,
    )

    pmapped_mixed = jax.pmap(
        compiled.mixed_devices_inner,
        axis_name="devices",
        in_axes=(0, 0, 0, 0, None, None, None),
        devices=pmap_devices,
    )

    def step_parallel_mixed(
        values: list, opt_state: Any, x: Any, y: Any, epoch_t: float, scale: float
    ):
        tt = float(epoch_t)
        lr = schedule(tt)
        eta = schedule.multiplier(tt)
        return pmapped_mixed(values, opt_state, x, y, lr, eta, jnp.float32(scale))

    return step_parallel_mixed


__all__ = [
    "make_accumulating_step",
    "make_step",
    "make_step_parallel",
]
