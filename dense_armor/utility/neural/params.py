"""A parameter that holds its gradient, and the update through an optimizer."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


class Param:
    """A jax array with a gradient of the same shape and dtype.

    Args:
        value: the array held by the parameter. Converted to ``jax.numpy``.

    Raises:
        ValueError: on construction if ``value`` is not an array.
    """

    def __init__(self, value: Any) -> None:
        v = jnp.asarray(value)
        if v.shape == ():
            raise ValueError("value must be an array, got a scalar")
        self.value = v
        self.grad = jnp.zeros_like(v)

    def zero_grad(self) -> Param:
        """Set the gradient to zero."""
        self.grad = jnp.zeros_like(self.value)
        return self

    def update(self, optimizer: Any) -> Param:
        """Replace ``value`` with ``optimizer.step(value, grad)``.

        The optimizer is the object form of ``dense_armor.utility.optim``:
        its ``step(weights, gradient)`` returns the new weights.

        Raises:
            ValueError: if ``optimizer.step`` returns a different shape.
        """
        new_value = optimizer.step(self.value, self.grad)
        nv = jnp.asarray(new_value)
        if nv.shape != self.value.shape:
            raise ValueError(
                f"optimizer.step returned shape {nv.shape}, "
                f"expected {self.value.shape}"
            )
        self.value = nv
        return self
