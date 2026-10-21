"""Chains and branches of layers.

``Sequential`` passes an input through its layers in order, and
``Parallel`` sends the same input to several branches and merges the
outputs. Both are layers: ``forward``, ``backward`` and ``params()``,
so a chain goes inside another chain, inside ``Residual`` and inside
``AsTransformer``.

References:
    He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep Residual Learning
        for Image Recognition. arXiv:1512.03385. Page 2, Figure 2: the
        building block. Page 3, Eqn.(1): ``y = F(x, {W_i}) + x``.
"""

from __future__ import annotations

import inspect
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from dense_armor.utility.neural.params import Param


def _accepts_training(layer: Any) -> bool:
    """True if ``layer.forward`` accepts a keyword argument ``training``."""
    try:
        sig = inspect.signature(layer.forward)
    except (TypeError, ValueError):
        return False
    if "training" in sig.parameters:
        return True
    return any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values())


class Sequential:
    """A chain of layers, applied in order.

    ``forward(x, training=True)`` passes ``x`` through the layers from
    the first to the last; a layer whose ``forward`` accepts ``training``
    receives it, any other layer is called as ``forward(x)``. The test of
    the argument is made once, when the chain is built. ``backward(g)``
    passes ``g`` through the layers in reverse order and returns the
    gradient of the input.

    ``layers`` is a non-empty list of objects with ``forward`` and
    ``backward``; ``params()`` is optional on a single layer.

    Args:
        layers: the layers, in the order they are applied.

    Raises:
        ValueError: if ``layers`` is empty.

    Examples:
        >>> import jax
        >>> from dense_armor.utility.neural.chain import Sequential
        >>> from dense_armor.utility.neural.layers import Dense
        >>> from dense_armor.utility.neural.activations import Relu
        >>> s = Sequential([Dense(3, 4, jax.random.PRNGKey(0)), Relu(),
        ...                 Dense(4, 2, jax.random.PRNGKey(1))])
        >>> len(s)
        3
    """

    def __init__(self, layers: list[Any]) -> None:
        if not layers:
            raise ValueError("Sequential needs at least one layer")
        self.layers = list(layers)
        self._accepts = [_accepts_training(layer) for layer in self.layers]
        self.x_: jax.Array | None = None

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, i: int) -> Any:
        return self.layers[i]

    def __repr__(self) -> str:
        return f"Sequential([{', '.join(repr(x) for x in self.layers)}])"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sequential):
            return NotImplemented
        if len(self.layers) != len(other.layers):
            return False
        for a, b in zip(self.layers, other.layers):
            if type(a) is not type(b):
                return False
            if a != b:
                return False
        return True

    __hash__ = None  # type: ignore[assignment]

    def add(self, layer: Any) -> Sequential:
        """Append a layer and return ``self``."""
        self.layers.append(layer)
        self._accepts.append(_accepts_training(layer))
        return self

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Pass ``x`` through the layers from the first to the last."""
        self.x_ = x
        y = x
        for layer, accepts in zip(self.layers, self._accepts):
            if accepts:
                y = layer.forward(y, training=training)
            else:
                y = layer.forward(y)
        return y

    def backward(self, g: jax.Array) -> jax.Array:
        """Pass ``g`` through the layers in reverse order."""
        if self.x_ is None:
            raise RuntimeError("call forward() before backward()")
        for i in range(len(self.layers) - 1, -1, -1):
            g = self.layers[i].backward(g)
        return g

    def params(self) -> list[Param]:
        """The parameters of every layer, in order."""
        out: list[Param] = []
        for layer in self.layers:
            if hasattr(layer, "params"):
                out.extend(layer.params())
        return out

    def zero_grad(self) -> None:
        """Set the gradient of every parameter to zero."""
        for p in self.params():
            p.zero_grad()

    def state_dict(self) -> list[np.ndarray]:
        """The values of every parameter, as numpy arrays, in order."""
        return [np.asarray(p.value) for p in self.params()]

    def load_state_dict(self, state: list[np.ndarray]) -> None:
        """Set the values from a list of the same length.

        Raises:
            ValueError: if the length or the shape of an array differs.
        """
        params = self.params()
        if len(state) != len(params):
            raise ValueError(f"state has {len(state)} arrays, expected {len(params)}")
        for i, (p, v) in enumerate(zip(params, state)):
            arr = np.asarray(v)
            if arr.shape != p.value.shape:
                raise ValueError(
                    f"state[{i}] has shape {arr.shape}, expected {p.value.shape}"
                )
            p.value = jnp.asarray(arr, dtype=p.value.dtype)


class Parallel:
    """Several branches on the same input, merged on the output.

    ``merge="add"`` sums the outputs of the branches; the shapes must
    be equal. ``merge="concat"`` concatenates them on the last axis.
    Any other value, fewer than two branches, or shapes that do not
    match with ``"add"`` raise ``ValueError``.

    ``backward(g)`` splits ``g`` for ``"concat"`` (one slice per branch,
    by the width of each output) and passes the same ``g`` to every
    branch for ``"add"``, then returns the sum of the gradients that the
    branches return.

    Args:
        branches: the layers, at least two.
        merge: ``"add"`` (default) or ``"concat"``.

    Raises:
        ValueError: on a bad ``merge``, fewer than two branches, or
            shapes that do not match with ``"add"``.

    Examples:
        >>> import jax
        >>> from dense_armor.utility.neural.chain import Parallel
        >>> from dense_armor.utility.neural.layers import Dense
        >>> p = Parallel([Dense(3, 4, jax.random.PRNGKey(0)),
        ...               Dense(3, 4, jax.random.PRNGKey(1))], "add")
        >>> len(p)
        2
    """

    def __init__(self, branches: list[Any], merge: str = "add") -> None:
        if merge not in ("add", "concat"):
            raise ValueError(f"merge must be 'add' or 'concat', got {merge!r}")
        if len(branches) < 2:
            raise ValueError(
                f"Parallel needs at least two branches, got {len(branches)}"
            )
        self.branches = list(branches)
        self.merge = merge
        self._accepts = [_accepts_training(b) for b in self.branches]
        self.x_: jax.Array | None = None
        self._ys: list[jax.Array] = []

    def __len__(self) -> int:
        return len(self.branches)

    def __getitem__(self, i: int) -> Any:
        return self.branches[i]

    def __repr__(self) -> str:
        return (
            f"Parallel([{', '.join(repr(b) for b in self.branches)}], "
            f"merge={self.merge!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Parallel):
            return NotImplemented
        if self.merge != other.merge:
            return False
        if len(self.branches) != len(other.branches):
            return False
        for a, b in zip(self.branches, other.branches):
            if type(a) is not type(b):
                return False
            if a != b:
                return False
        return True

    __hash__ = None  # type: ignore[assignment]

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Send ``x`` to every branch, then merge the outputs."""
        self.x_ = x
        self._ys = []
        for branch, accepts in zip(self.branches, self._accepts):
            if accepts:
                y = branch.forward(x, training=training)
            else:
                y = branch.forward(x)
            self._ys.append(y)
        if self.merge == "add":
            shape0 = self._ys[0].shape
            for i, y in enumerate(self._ys[1:], 1):
                if y.shape != shape0:
                    raise ValueError(
                        f"branch {i} has output shape {y.shape}, "
                        f"expected the shape of branch 0, {shape0}"
                    )
            return sum(self._ys)
        return jnp.concatenate(self._ys, axis=-1)

    def backward(self, g: jax.Array) -> jax.Array:
        """Split or copy ``g``, then sum the input gradients."""
        if self.x_ is None:
            raise RuntimeError("call forward() before backward()")
        if self.merge == "concat":
            widths = [y.shape[-1] for y in self._ys]
            pieces = []
            start = 0
            for w in widths:
                pieces.append(jnp.take(g, jnp.arange(start, start + w), axis=-1))
                start += w
        else:
            pieces = [g] * len(self.branches)
        dxs = [self.branches[i].backward(pieces[i]) for i in range(len(self.branches))]
        return sum(dxs)

    def params(self) -> list[Param]:
        """The parameters of every branch, in order."""
        out: list[Param] = []
        for branch in self.branches:
            if hasattr(branch, "params"):
                out.extend(branch.params())
        return out

    def zero_grad(self) -> None:
        """Set the gradient of every parameter to zero."""
        for p in self.params():
            p.zero_grad()
