"""Two classic architectures: the MLP and the residual MLP.

``mlp`` builds a chain of dense layers with an activation between them.
``resnet_mlp`` builds a chain of residual blocks. Both are
``Sequential``, so the same training loop of the layers works on them.

References:
    He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep Residual Learning
        for Image Recognition. arXiv:1512.03385. Page 2, Figure 2: the
        building block. Page 3, Eqn.(1): ``y = F(x, {W_i}) + x``.
    He, K., Zhang, X., Ren, S., Sun, J. (2015). Delving Deep into
        Rectifiers. arXiv:1502.01852. Page 5, Eqn.(15): the variance of
        the initialisation, with ``a = 0`` the rectifier case and
        ``a = 1`` the linear case named Xavier.
"""

from __future__ import annotations

from typing import Any

import jax

from dense_armor.utility.neural.activations import Relu, Sigmoid, Tanh
from dense_armor.utility.neural.chain import Sequential
from dense_armor.utility.neural.layers import Dense, Residual


def _make_activation(name: str) -> Any:
    if name == "relu":
        return Relu()
    if name == "tanh":
        return Tanh()
    if name == "sigmoid":
        return Sigmoid()
    raise ValueError(f"activation must be 'relu', 'tanh' or 'sigmoid', got {name!r}")


def mlp(sizes: list[int], key: Any, activation: str = "relu") -> Sequential:
    """A chain of dense layers with an activation between them.

    ``sizes`` is the list of the widths, from the input to the output;
    the chain has ``len(sizes) - 1`` dense layers and ``len(sizes) - 2``
    activations, with no activation after the last dense layer.

    The dense layers use ``init="he"`` when ``activation="relu"`` and
    ``init="xavier"`` when the activation is ``"tanh"`` or
    ``"sigmoid"``. This is our choice: He et al. 2015, Eqn.(15) page 5,
    derive the variance for the rectifier and name the linear case as
    Xavier; the rectifier is the activation the He initialisation is
    derived for, and the linear case is the one Xavier is derived for.

    Args:
        sizes: the widths, at least two.
        key: a ``jax.random`` key, split once per dense layer.
        activation: ``"relu"`` (default), ``"tanh"`` or ``"sigmoid"``.

    Raises:
        ValueError: if ``sizes`` has fewer than two entries, or if
            ``activation`` is none of the three names.

    Examples:
        >>> import jax
        >>> from dense_armor.utility.neural.models import mlp
        >>> m = mlp([2, 8, 1], jax.random.PRNGKey(0))
        >>> [p.value.shape for p in m.params()]
        [(2, 8), (8,), (8, 1), (1,)]
    """
    if len(sizes) < 2:
        raise ValueError(f"sizes needs at least two entries, got {len(sizes)}")
    _make_activation(activation)
    init = "he" if activation == "relu" else "xavier"
    n_layers = len(sizes) - 1
    keys = jax.random.split(key, n_layers)
    layers: list[Any] = []
    for i in range(n_layers):
        layers.append(Dense(sizes[i], sizes[i + 1], keys[i], init=init))
        if i < n_layers - 1:
            layers.append(_make_activation(activation))
    return Sequential(layers)


def resnet_mlp(
    width: int,
    depth: int,
    n_in: int,
    n_out: int,
    key: Any,
) -> Sequential:
    """A chain of residual blocks, the dense form of He et al. 2015.

    The chain is: a ``Dense(n_in, width)`` and a ``Relu``, then
    ``depth`` blocks, each ``Sequential([Residual(Sequential([
    Dense(width, width), Relu(), Dense(width, width)])), Relu()])``,
    then a ``Dense(width, n_out)``. The block is Figure 2, page 2 of
    He et al. 2016 with the addition ``y = F(x) + x`` of Eqn.(1),
    page 3, followed by the rectifier.

    The blocks have no normalisation layer: the block of the figure is
    written for convolutions, and this is the dense form. This is our
    choice, not a claim of the paper.

    Args:
        width: the width of every hidden layer.
        depth: the number of residual blocks, at least one.
        n_in: the input width.
        n_out: the output width.
        key: a ``jax.random`` key, split once per dense layer.

    Raises:
        ValueError: if ``depth`` is below 1.

    Examples:
        >>> import jax
        >>> from dense_armor.utility.neural.models import resnet_mlp
        >>> r = resnet_mlp(8, 2, 3, 2, jax.random.PRNGKey(0))
        >>> len(r.params())
        12
    """
    if depth < 1:
        raise ValueError(f"depth must be at least 1, got {depth}")
    keys = jax.random.split(key, 2 + 2 * depth)
    layers: list[Any] = [
        Dense(n_in, width, keys[0]),
        Relu(),
    ]
    k = 1
    for _ in range(depth):
        inner = Sequential(
            [
                Dense(width, width, keys[k]),
                Relu(),
                Dense(width, width, keys[k + 1]),
            ]
        )
        block = Sequential([Residual(inner), Relu()])
        layers.append(block)
        k += 2
    layers.append(Dense(width, n_out, keys[k]))
    return Sequential(layers)
