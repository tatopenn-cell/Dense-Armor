"""Activations with a forward and a backward, on a batch or one sample.

Each activation has two pure functions, ``<name>_forward(x)`` and
``<name>_backward(x, y, g)``, that work under ``jax.jit``, and a class
whose ``forward(x)`` caches ``x`` and ``y`` and whose ``backward(g)``
returns the gradient of the input.

References:
    He, K., Zhang, X., Ren, S., Sun, J. (2015). Delving Deep into
        Rectifiers. arXiv:1502.01852. Eqn.(1), page 2: the parametric
        rectifier ``f(y) = y`` if ``y > 0``, ``a*y`` otherwise.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def relu_forward(x: jax.Array) -> jax.Array:
    """``max(x, 0)``. The derivative at 0 is 0 (choice of this code)."""
    return jnp.maximum(x, 0.0)


def relu_backward(x: jax.Array, y: jax.Array, g: jax.Array) -> jax.Array:
    """Derivative of ReLU: ``g * (x > 0)``. At ``x = 0`` it returns 0."""
    del y
    return jnp.where(x > 0.0, g, jnp.zeros_like(g))


def prelu_forward(x: jax.Array, a: float = 0.25) -> jax.Array:
    """``x`` where ``x > 0``, ``a*x`` otherwise (Eqn.(1), page 2)."""
    return jnp.where(x > 0.0, x, a * x)


def prelu_backward(x: jax.Array, y: jax.Array, g: jax.Array,
                   a: float = 0.25) -> jax.Array:
    """Derivative of PReLU: ``g`` where ``x > 0``, ``a*g`` otherwise."""
    del y
    return jnp.where(x > 0.0, g, a * g)


def sigmoid_forward(x: jax.Array) -> jax.Array:
    """``1 / (1 + exp(-x))``, stable for large negative ``x``."""
    return jax.nn.sigmoid(x)


def sigmoid_backward(x: jax.Array, y: jax.Array, g: jax.Array) -> jax.Array:
    """``g * y * (1 - y)``."""
    del x
    return g * y * (1.0 - y)


def tanh_forward(x: jax.Array) -> jax.Array:
    """``tanh(x)``."""
    return jnp.tanh(x)


def tanh_backward(x: jax.Array, y: jax.Array, g: jax.Array) -> jax.Array:
    """``g * (1 - y**2)``."""
    del x
    return g * (1.0 - y * y)


def softmax_forward(x: jax.Array) -> jax.Array:
    """Softmax over the last axis, stable with the subtraction of the max."""
    return jax.nn.softmax(x, axis=-1)


def softmax_backward(x: jax.Array, y: jax.Array, g: jax.Array) -> jax.Array:
    """Jacobian-vector product of softmax: ``y * (g - sum(g*y))``."""
    del x
    s = jnp.sum(g * y, axis=-1, keepdims=True)
    return y * (g - s)


class Relu:
    """ReLU: ``forward`` caches ``x`` and ``y``; ``backward`` returns ``dx``."""

    def __init__(self) -> None:
        self.x: jax.Array | None = None
        self.y: jax.Array | None = None

    def forward(self, x: jax.Array) -> jax.Array:
        y = relu_forward(x)
        self.x = x
        self.y = y
        return y

    def backward(self, g: jax.Array) -> jax.Array:
        if self.x is None or self.y is None:
            raise RuntimeError("call forward() before backward()")
        return relu_backward(self.x, self.y, g)


class PRelu:
    """PReLU with ``a`` as a ``Param``; ``backward`` fills ``a.grad``."""

    def __init__(self, a: float = 0.25) -> None:
        from dense_armor.utility.neural.params import Param
        self.a = Param(np.asarray([a], dtype=np.float32))
        self.x: jax.Array | None = None
        self.y: jax.Array | None = None

    def forward(self, x: jax.Array) -> jax.Array:
        a = float(self.a.value[0])
        y = prelu_forward(x, a)
        self.x = x
        self.y = y
        return y

    def backward(self, g: jax.Array) -> jax.Array:
        if self.x is None or self.y is None:
            raise RuntimeError("call forward() before backward()")
        a = float(self.a.value[0])
        dx = prelu_backward(self.x, self.y, g, a)
        da = jnp.sum(jnp.where(self.x > 0.0, 0.0, self.x * g))
        self.a.grad = jnp.asarray([da], dtype=self.a.value.dtype)
        return dx


class Sigmoid:
    def __init__(self) -> None:
        self.x: jax.Array | None = None
        self.y: jax.Array | None = None

    def forward(self, x: jax.Array) -> jax.Array:
        y = sigmoid_forward(x)
        self.x = x
        self.y = y
        return y

    def backward(self, g: jax.Array) -> jax.Array:
        if self.x is None or self.y is None:
            raise RuntimeError("call forward() before backward()")
        return sigmoid_backward(self.x, self.y, g)


class Tanh:
    def __init__(self) -> None:
        self.x: jax.Array | None = None
        self.y: jax.Array | None = None

    def forward(self, x: jax.Array) -> jax.Array:
        y = tanh_forward(x)
        self.x = x
        self.y = y
        return y

    def backward(self, g: jax.Array) -> jax.Array:
        if self.x is None or self.y is None:
            raise RuntimeError("call forward() before backward()")
        return tanh_backward(self.x, self.y, g)


class Softmax:
    def __init__(self) -> None:
        self.x: jax.Array | None = None
        self.y: jax.Array | None = None

    def forward(self, x: jax.Array) -> jax.Array:
        y = softmax_forward(x)
        self.x = x
        self.y = y
        return y

    def backward(self, g: jax.Array) -> jax.Array:
        if self.x is None or self.y is None:
            raise RuntimeError("call forward() before backward()")
        return softmax_backward(self.x, self.y, g)
