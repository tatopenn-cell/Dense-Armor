"""Feed-forward layers with a gated linear unit: GLU, GEGLU, SwiGLU."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from dense_armor.utility.neural.activations import (
    sigmoid_backward,
    sigmoid_forward,
)
from dense_armor.utility.neural.activations_more import (
    gelu_backward,
    gelu_forward,
    silu_backward,
    silu_forward,
)
from dense_armor.utility.neural.init import xavier_normal
from dense_armor.utility.neural.params import Param

_ACTIVATIONS = {
    "sigmoid": (sigmoid_forward, sigmoid_backward),
    "gelu": (gelu_forward, gelu_backward),
    "silu": (silu_forward, silu_backward),
}


class GLUFeedForward:
    """The feed-forward layer ``(act(x W) * (x V)) W2`` of Shazeer 2020.

    Eq.(6), page 2 of Shazeer (2020): the first linear layer and the
    activation of the Transformer feed-forward network are replaced by a
    gated linear unit, ``FFN_GLU`` with the sigmoid, ``FFN_GEGLU`` with the
    GELU and ``FFN_SwiGLU`` with Swish_1 (the SiLU). The layer has three
    weight matrices: ``W`` and ``V`` of shape ``(d_model, d_ff)`` and ``W2``
    of shape ``(d_ff, d_model)``. The paper omits the biases (page 2); with
    ``bias=True`` the layer adds ``b`` to ``x W``, ``c`` to ``x V`` and
    ``b2`` to the output, the forms with biases of Eq.(4), page 1, and Eq.(5), page 2 (our choice, off by default).
    The paper reduces ``d_ff`` by a factor ``2/3`` to keep the number of
    parameters of a two-matrix layer (page 2); this class does not.

    Args:
        d_model: the input and output dimension.
        d_ff: the hidden dimension.
        activation: ``"sigmoid"`` (GLU), ``"gelu"`` (GEGLU) or ``"silu"``
            (SwiGLU).
        key: a ``jax.random`` key.
        bias: add the three bias vectors. Default ``False``.

    Raises:
        ValueError: if ``activation`` is not one of the three names.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.glu import GLUFeedForward
        >>> f = GLUFeedForward(4, 8, "silu", jax.random.PRNGKey(0))
        >>> f.forward(jnp.ones((3, 4))).shape
        (3, 4)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: str,
        key: Any,
        bias: bool = False,
    ) -> None:
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {sorted(_ACTIVATIONS)}, got {activation!r}"
            )
        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.activation = activation
        self.bias = bool(bias)
        keys = jax.random.split(key, 3)
        self.W = Param(xavier_normal(keys[0], (d_model, d_ff)))
        self.V = Param(xavier_normal(keys[1], (d_model, d_ff)))
        self.W2 = Param(xavier_normal(keys[2], (d_ff, d_model)))
        if self.bias:
            self.b = Param(jnp.zeros(d_ff))
            self.c = Param(jnp.zeros(d_ff))
            self.b2 = Param(jnp.zeros(d_model))
        self._cache: dict[str, jax.Array] | None = None

    def __repr__(self) -> str:
        return (
            f"GLUFeedForward({self.d_model}, {self.d_ff}, "
            f"{self.activation!r}, bias={self.bias})"
        )

    def params(self) -> list[Param]:
        """``W``, ``V``, ``W2``, then ``b``, ``c``, ``b2`` if ``bias``."""
        out = [self.W, self.V, self.W2]
        if self.bias:
            out += [self.b, self.c, self.b2]
        return out

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Return ``(act(x W + b) * (x V + c)) W2 + b2`` on the last axis."""
        del training
        x = jnp.asarray(x)
        a = x @ self.W.value
        c = x @ self.V.value
        if self.bias:
            a = a + self.b.value
            c = c + self.c.value
        act = _ACTIVATIONS[self.activation][0](a)
        h = act * c
        y = h @ self.W2.value
        if self.bias:
            y = y + self.b2.value
        self._cache = dict(x=x, a=a, c=c, act=act, h=h)
        return y

    def backward(self, g: jax.Array) -> jax.Array:
        """Return ``dx`` and add to the gradients of every ``Param``."""
        if self._cache is None:
            raise RuntimeError("call forward() before backward()")
        g = jnp.asarray(g)
        x, a, c = self._cache["x"], self._cache["a"], self._cache["c"]
        act, h = self._cache["act"], self._cache["h"]
        d_in = x.shape[-1]
        d_ff = h.shape[-1]
        dh = g @ self.W2.value.T
        d_act = dh * c
        d_c = dh * act
        d_a = _ACTIVATIONS[self.activation][1](a, act, d_act)
        x_flat = x.reshape(-1, d_in)
        self.W2.grad = self.W2.grad + h.reshape(-1, d_ff).T @ g.reshape(-1, d_in)
        self.W.grad = self.W.grad + x_flat.T @ d_a.reshape(-1, d_ff)
        self.V.grad = self.V.grad + x_flat.T @ d_c.reshape(-1, d_ff)
        if self.bias:
            batch_axes = tuple(range(g.ndim - 1))
            self.b.grad = self.b.grad + jnp.sum(d_a, axis=batch_axes)
            self.c.grad = self.c.grad + jnp.sum(d_c, axis=batch_axes)
            self.b2.grad = self.b2.grad + jnp.sum(g, axis=batch_axes)
        return d_a @ self.W.value.T + d_c @ self.V.value.T
