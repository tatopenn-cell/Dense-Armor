"""PyTorch bridge for Dense-Armor.

Converts a torch tensor (or a pytree of tensors) to a jax array (or a
pytree of jax arrays) and back, keeping the dtype and, for contiguous
CPU tensors, the buffer (no copy).

Needs the optional dependency: ``pip install dense-armor[torch]``.
"""

from __future__ import annotations

from typing import Any

import copy

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import torch
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "dense_armor.bridges.torch needs torch and jax: pip install dense-armor[torch]"
    ) from exc

__all__ = ["TorchModule", "TorchAdapter", "from_torch", "to_jax", "to_torch"]


def _to_jax_leaf(t: Any) -> Any:
    if isinstance(t, torch.Tensor):
        if t.requires_grad:
            t = t.detach()
        if not t.is_contiguous():
            t = t.contiguous()
        return jax.dlpack.from_dlpack(t)
    return t


def _to_torch_leaf(a: Any) -> Any:
    if isinstance(a, jax.Array):
        return torch.from_dlpack(a)
    return a


def to_jax(x: Any) -> Any:
    """Convert a torch tensor, or a pytree of them, to jax.

    Args:
        x: a ``torch.Tensor``, or any pytree whose leaves may be
            ``torch.Tensor``.

    Returns:
        A ``jax.Array`` for a tensor input, or a pytree with the same
        structure and ``jax.Array`` leaves. The dtype is kept: a
        ``float32`` tensor becomes a ``float32`` array, no float64 is
        introduced. A tensor that requires grad is detached before the
        conversion; the caller re-attaches the gradient on the torch
        side if it needs it.

    Example:
        >>> import torch
        >>> from dense_armor.bridges.torch import to_jax
        >>> t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        >>> a = to_jax(t)
        >>> a.dtype, a.shape
        (dtype('float32'), (3,))
    """
    return jax.tree_util.tree_map(_to_jax_leaf, x)


def to_torch(a: Any) -> Any:
    """Convert a jax array, or a pytree of them, to torch.

    Args:
        a: a ``jax.Array``, or any pytree whose leaves may be
            ``jax.Array``.

    Returns:
        A ``torch.Tensor`` for an array input, or a pytree with the same
        structure and ``torch.Tensor`` leaves. The dtype is kept.

    Example:
        >>> import jax.numpy as jnp
        >>> from dense_armor.bridges.torch import to_torch
        >>> a = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        >>> t = to_torch(a)
        >>> t.dtype, tuple(t.shape)
        (torch.float32, (3,))
    """
    return jax.tree_util.tree_map(_to_torch_leaf, a)


class TorchModule(torch.nn.Module):
    """A Dense-Armor model wrapped as a ``torch.nn.Module``.

    Every Dense-Armor ``Param`` becomes a ``torch.nn.Parameter``, so an
    optimiser of torch sees them. ``forward`` calls the Dense-Armor
    model through a ``torch.autograd.Function`` whose backward is
    ``jax.vjp`` of the same call.
    """

    def __init__(self, model: Any) -> None:
        super().__init__()
        self._model = model
        self._torch_params = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.tensor(np.asarray(p.value)))
                for p in model.params()
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return _TorchModuleFn.apply(
            self._model, x, self.training, *tuple(self._torch_params)
        )


class _TorchModuleFn(torch.autograd.Function):
    """``torch.autograd.Function`` whose forward and backward are jax."""

    @staticmethod
    def forward(
        ctx: Any, model: Any, x: torch.Tensor, training: bool, *params: Any
    ) -> torch.Tensor:
        import copy

        base = model
        pp = [jnp.asarray(p.detach().numpy()) for p in params]
        xa = jnp.asarray(x.detach().numpy())

        def f(pp_, xx_):
            m = copy.deepcopy(base)
            for param, v in zip(m.params(), pp_):
                param.value = v
            return m.forward(xx_, training=training)

        out, vjp_fn = jax.vjp(f, pp, xa)
        ctx.save_for_backward(*params)
        ctx._vjp_fn = vjp_fn
        return torch.from_dlpack(out).clone()

    @staticmethod
    def backward(ctx: Any, g_out: torch.Tensor) -> tuple:
        g = jnp.asarray(g_out.detach().numpy())
        g_p, g_x = ctx._vjp_fn(g)
        out = [None, torch.from_dlpack(g_x).clone(), None]
        out.extend(torch.from_dlpack(gg).clone() for gg in g_p)
        return tuple(out)


class TorchAdapter:
    """A ``torch.nn.Module`` usable inside Dense-Armor.

    ``apply(x)`` runs the torch module on ``x`` (a jax array or a torch
    tensor). If ``x`` is a jax array, the gradient of the output with
    respect to ``x`` is computed with ``torch.autograd`` and exposed to
    jax through ``jax.custom_vjp``, so a Dense-Armor pipeline can back
    propagate through the adapter.

    The module's own parameters are updated by torch, not by the
    Dense-Armor estimator: the adapter is a computation node, not a
    container of params.

    Args:
        module: a ``torch.nn.Module``.

    Example:
        >>> import torch
        >>> from dense_armor.bridges.torch import from_torch
        >>> module = torch.nn.Linear(4, 2)
        >>> adapter = from_torch(module)
        >>> x = torch.zeros(3, 4)
        >>> adapter.apply(x).shape
        torch.Size([3, 2])
    """

    def __init__(self, module: Any) -> None:
        self.module = module
        self._apply_jax = _make_apply_jax(module)

    def apply(self, x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return self.module(x)
        return self._apply_jax(x)


def _make_apply_jax(module: Any) -> Any:
    out_dim = getattr(module, "out_features", None)
    if out_dim is None:
        out_dim = list(module.parameters())[-1].shape[0]

    @jax.custom_vjp
    def _apply(x: Any) -> Any:
        return _forward(x)

    def _forward(x: Any) -> Any:
        out_shape = jax.ShapeDtypeStruct((x.shape[0], out_dim), jnp.float32)

        def cb(x_np: Any) -> Any:
            with torch.no_grad():
                x_t = torch.from_numpy(np.array(x_np, copy=True))
                y = module(x_t)
            return y.numpy()

        return jax.pure_callback(cb, out_shape, x)

    def _fwd(x: Any) -> tuple:
        return _forward(x), (x,)

    def _bwd(res: tuple, g: Any) -> tuple:
        (x,) = res
        g_shape = jax.ShapeDtypeStruct(x.shape, jnp.float32)

        def cb(x_np: Any, g_np: Any) -> Any:
            x_t = torch.from_numpy(np.array(x_np, copy=True)).requires_grad_(True)
            y = module(x_t)
            y.backward(torch.from_numpy(np.array(g_np, copy=True)))
            return x_t.grad.numpy()

        g_x = jax.pure_callback(cb, g_shape, x, g)
        return (g_x,)

    _apply.defvjp(_fwd, _bwd)
    return _apply


def from_torch(module: Any) -> TorchAdapter:
    """Wrap a ``torch.nn.Module`` as a Dense-Armor adapter.

    Args:
        module: a ``torch.nn.Module``.

    Returns:
        A ``TorchAdapter`` whose ``apply`` runs the module on the input
        and, for a jax array input, exposes the gradient of the module
        through ``jax.custom_vjp``. A torch tensor input is passed to the
        module unchanged, and torch.autograd of the caller still works.
    """
    return TorchAdapter(module)
