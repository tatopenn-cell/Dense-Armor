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

from typing import Any

import jax
import jax.numpy as jnp


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


def _write_values(model: Any, values: list) -> None:
    """Set ``p.value`` of every ``Param`` from the flat list."""
    for p, v in zip(model.params(), values):
        p.value = jnp.asarray(v)


def _read_values(model: Any) -> list:
    """Return the current ``p.value`` of every ``Param`` as a flat list."""
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
) -> Any:
    """Return ``step(values, opt_state, x, y, epoch_t)``.

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

    Returns:
        A function ``step(values, opt_state, x, y, epoch_t)`` that
        returns ``(new_values, new_opt_state, loss_value)``.
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
    jitted = jax.jit(_step)

    def step(
        values: list, opt_state: Any, x: Any, y: Any, epoch_t: float
    ) -> tuple[list, Any, Any]:
        t = float(epoch_t)
        lr = schedule(t)
        eta = schedule.multiplier(t)
        return jitted(values, opt_state, x, y, lr, eta)

    return step


__all__ = [
    "make_step",
]
