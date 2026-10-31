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
import jax.numpy as jnp

from dense_armor.utility.neural.activations import Relu, Sigmoid, Tanh
from dense_armor.utility.neural.attention import (
    TransformerBlock,
    sinusoidal_positions,
)
from dense_armor.utility.neural.chain import Sequential
from dense_armor.utility.neural.conv import Conv2D, Flatten, MaxPool2D
from dense_armor.utility.neural.decoder import TransformerDecoderBlock
from dense_armor.utility.neural.layers import Dense, Embedding, Residual
from dense_armor.utility.neural.norm import LayerNorm
from dense_armor.utility.neural.params import Param
from dense_armor.utility.neural.recurrent import GRU, LSTM, RNN


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


class LastStep:
    """Take the last step of a sequence ``(B, T, n)``, giving ``(B, n)``.

    The ``backward`` puts the whole ``g`` at the last step and zeros
    elsewhere.

    Examples:
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.models import LastStep
        >>> l = LastStep()
        >>> l.forward(jnp.ones((2, 3, 4))).shape
        (2, 4)
    """

    def __init__(self) -> None:
        self.x_: jax.Array | None = None

    def __repr__(self) -> str:
        return "LastStep()"

    def params(self) -> list[Param]:
        """Empty: the layer has no parameters."""
        return []

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Return the last step of every sequence."""
        del training
        x = jnp.asarray(x)
        self.x_ = x
        if x.ndim == 2:
            return x[-1]
        return x[:, -1, :]

    def backward(self, g: jax.Array) -> jax.Array:
        """Put ``g`` at the last step, zeros elsewhere."""
        if self.x_ is None:
            raise RuntimeError("call forward() before backward()")
        g = jnp.asarray(g)
        dx = jnp.zeros_like(self.x_)
        if self.x_.ndim == 2:
            return dx.at[-1].set(g)
        return dx.at[:, -1, :].set(g)


class MeanOverTime:
    """Mean over the time axis of ``(B, T, n)`` or ``(T, n)``.

    The ``backward`` spreads ``g / T`` over every step.

    Examples:
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.models import MeanOverTime
        >>> m = MeanOverTime()
        >>> m.forward(jnp.ones((2, 3, 4))).shape
        (2, 4)
    """

    def __init__(self) -> None:
        self.x_: jax.Array | None = None

    def __repr__(self) -> str:
        return "MeanOverTime()"

    def params(self) -> list[Param]:
        """Empty: the layer has no parameters."""
        return []

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Return the mean over the time axis."""
        del training
        x = jnp.asarray(x)
        self.x_ = x
        if x.ndim == 2:
            return x.mean(axis=0)
        return x.mean(axis=1)

    def backward(self, g: jax.Array) -> jax.Array:
        """Spread ``g / T`` over every step."""
        if self.x_ is None:
            raise RuntimeError("call forward() before backward()")
        g = jnp.asarray(g)
        T = self.x_.shape[-2]
        if self.x_.ndim == 2:
            return jnp.broadcast_to(g / T, self.x_.shape)
        return jnp.broadcast_to((g / T)[:, None, :], self.x_.shape)


class AddPositions:
    """Add the sinusoidal positions of page 6 of Vaswani et al. 2017.

    Args:
        max_len: the maximum sequence length.

    Raises:
        ValueError: if the sequence is longer than ``max_len``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.models import AddPositions
        >>> a = AddPositions(8)
        >>> a.forward(jnp.zeros((2, 4, 8))).shape
        (2, 4, 8)
    """

    def __init__(self, max_len: int) -> None:
        self.max_len = max_len
        self.x_: jax.Array | None = None

    def __repr__(self) -> str:
        return f"AddPositions({self.max_len})"

    def params(self) -> list[Param]:
        """Empty: the layer has no parameters."""
        return []

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Add the positions to the last axis of ``x``."""
        del training
        x = jnp.asarray(x)
        T = x.shape[-2]
        if T > self.max_len:
            raise ValueError(
                f"sequence length {T} is longer than max_len {self.max_len}"
            )
        self.x_ = x
        pos = sinusoidal_positions(T, x.shape[-1])
        if x.ndim == 2:
            return x + pos
        return x + pos[None]

    def backward(self, g: jax.Array) -> jax.Array:
        """Return ``g`` unchanged, since the positions have no gradient."""
        if self.x_ is None:
            raise RuntimeError("call forward() before backward()")
        return jnp.asarray(g)


class _CnnSequential(Sequential):
    """Sequential that tolerates no-batch inputs to the cnn.

    The internal chain always sees ``(B, H, W, C)``: a ``(H, W, C)``
    input gets a batch axis added, and the output's batch axis is
    squeezed back. This keeps ``Flatten`` (which assumes the batch
    axis when the input has more than one axis) working for both
    cases.
    """

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        x = jnp.asarray(x)
        squeeze = x.ndim == 3
        if squeeze:
            x = x[None]
        y = super().forward(x, training=training)
        return y[0] if squeeze else y

    def backward(self, g: jax.Array) -> jax.Array:
        g = jnp.asarray(g)
        squeeze = g.ndim == 1
        if squeeze:
            g = g[None]
        dx = super().backward(g)
        return dx[0] if squeeze else dx


def cnn(
    input_shape: tuple[int, int, int],
    channels: list[int],
    n_out: int,
    key: Any,
    kernel: int = 3,
    activation: str = "relu",
) -> Sequential:
    """A chain of ``Conv2D`` + activation + ``MaxPool2D(2)``, then a head.

    Each convolutional layer uses ``padding=kernel // 2`` so that the
    pooling is what reduces the spatial size. A side that the pooling
    would reduce below 1 raises ``ValueError`` (the ``MaxPool2D`` forward
    raises). ``activation`` is ``"relu"``, ``"tanh"`` or ``"sigmoid"``.

    Args:
        input_shape: ``(H, W, C)`` of a single sample.
        channels: the output channels of each convolutional layer.
        n_out: the width of the output of the final dense layer.
        key: a ``jax.random`` key, split once per layer with weights.
        kernel: the kernel side of every ``Conv2D``.
        activation: ``"relu"`` (default), ``"tanh"`` or ``"sigmoid"``.

    Raises:
        ValueError: if ``activation`` is not one of the three names, or if
            a pooling reduces a side below 1.

    Examples:
        >>> import jax
        >>> from dense_armor.utility.neural.models import cnn
        >>> m = cnn((8, 8, 1), [2], 3, jax.random.PRNGKey(0))
        >>> len(m.params())
        4
    """
    _make_activation(activation)
    n_layers = len(channels)
    keys = jax.random.split(key, n_layers + 1)
    H, W, C = input_shape
    layers: list[Any] = []
    c_prev = C
    for i, c in enumerate(channels):
        layers.append(Conv2D(c_prev, c, kernel, keys[i], padding=kernel // 2))
        layers.append(_make_activation(activation))
        layers.append(MaxPool2D(2))
        H = H // 2
        W = W // 2
        if H < 1 or W < 1:
            raise ValueError(
                f"pooling reduces the side below 1: {H} x {W} after "
                f"layer {i} of {n_layers}"
            )
        c_prev = c
    layers.append(Flatten())
    layers.append(Dense(H * W * c_prev, n_out, keys[n_layers]))
    return _CnnSequential(layers)


def recurrent_net(
    n_in: int,
    n_hidden: int,
    n_out: int,
    key: Any,
    cell: str = "lstm",
    n_layers: int = 1,
) -> Sequential:
    """A chain of recurrent layers, then ``LastStep`` and a dense head.

    Args:
        n_in: the input width of the first layer.
        n_hidden: the hidden width of every layer.
        n_out: the width of the output of the dense head.
        key: a ``jax.random`` key, split once per layer with weights.
        cell: ``"lstm"`` (default), ``"gru"`` or ``"rnn"``.
        n_layers: the number of recurrent layers, at least one.

    Raises:
        ValueError: if ``cell`` is not one of the three names, or if
            ``n_layers`` is below 1.

    Examples:
        >>> import jax
        >>> from dense_armor.utility.neural.models import recurrent_net
        >>> m = recurrent_net(3, 4, 2, jax.random.PRNGKey(0))
        >>> len(m.params())
        17
    """
    if cell not in ("lstm", "gru", "rnn"):
        raise ValueError(f"cell must be 'lstm', 'gru' or 'rnn', got {cell!r}")
    if n_layers < 1:
        raise ValueError(f"n_layers must be at least 1, got {n_layers}")
    keys = jax.random.split(key, n_layers + 1)
    layers: list[Any] = []
    in_prev = n_in
    for i in range(n_layers):
        if cell == "lstm":
            layers.append(LSTM(in_prev, n_hidden, keys[i]))
        elif cell == "gru":
            layers.append(GRU(in_prev, n_hidden, keys[i]))
        else:
            layers.append(RNN(in_prev, n_hidden, keys[i]))
        in_prev = n_hidden
    layers.append(LastStep())
    layers.append(Dense(n_hidden, n_out, keys[n_layers]))
    return Sequential(layers)


def transformer_encoder(
    n_in: int,
    d_model: int,
    n_heads: int,
    d_ff: int,
    depth: int,
    n_out: int,
    key: Any,
    max_len: int,
    causal: bool = False,
) -> Sequential:
    """The encoder of Vaswani et al. 2017, as a chain.

    The chain is ``Dense(n_in, d_model)``, ``AddPositions(max_len)``,
    ``depth`` ``TransformerBlock``, ``MeanOverTime``, ``Dense(d_model,
    n_out)``.

    Args:
        n_in: the input width.
        d_model: the representation width.
        n_heads: the number of heads of every block.
        d_ff: the width of the feed-forward of every block.
        depth: the number of blocks, at least one.
        n_out: the width of the output of the final dense layer.
        key: a ``jax.random`` key, split once per layer with weights.
        max_len: the maximum sequence length.
        causal: if True, every block is causal.

    Raises:
        ValueError: if ``depth`` is below 1.

    Examples:
        >>> import jax
        >>> from dense_armor.utility.neural.models import transformer_encoder
        >>> m = transformer_encoder(4, 8, 2, 16, 1, 3, jax.random.PRNGKey(0), 16)
        >>> m.forward(jax.numpy.ones((5, 4))).shape
        (3,)
    """
    if depth < 1:
        raise ValueError(f"depth must be at least 1, got {depth}")
    keys = jax.random.split(key, depth + 2)
    layers: list[Any] = [
        Dense(n_in, d_model, keys[0]),
        AddPositions(max_len),
    ]
    for i in range(depth):
        layers.append(
            TransformerBlock(d_model, n_heads, d_ff, keys[i + 1], causal=causal)
        )
    layers.append(MeanOverTime())
    layers.append(Dense(d_model, n_out, keys[depth + 1]))
    return Sequential(layers)


class _TiedHead:
    """Logits ``x @ E.T``, reusing an embedding matrix as output projection.

    Our choice, not in Vaswani et al. 2017: the paper does not tie the
    output projection to the input embedding. Weight tying is a common
    practice in language-model implementations.

    Args:
        embedding: the ``Embedding`` whose matrix ``E`` is reused.
    """

    def __init__(self, embedding: Embedding) -> None:
        self.embedding = embedding
        self.x_: jax.Array | None = None

    def __repr__(self) -> str:
        return (
            f"_TiedHead(n_tokens={self.embedding.n_tokens}, dim={self.embedding.dim})"
        )

    def params(self) -> list[Param]:
        """Empty: the matrix is the one of the wrapped ``Embedding``."""
        return []

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Return ``x @ E.T``, the logits over the vocabulary."""
        del training
        x = jnp.asarray(x)
        self.x_ = x
        return x @ self.embedding.E.value.T

    def backward(self, g: jax.Array) -> jax.Array:
        """Gradients: adds ``dE`` to ``E.grad`` and returns ``dx``."""
        if self.x_ is None:
            raise RuntimeError("call forward() before backward()")
        x = self.x_
        g = jnp.asarray(g)
        x_flat = x.reshape(-1, x.shape[-1])
        g_flat = g.reshape(-1, g.shape[-1])
        dE = g_flat.T @ x_flat
        self.embedding.E.grad = self.embedding.E.grad + dE
        return g @ self.embedding.E.value


def transformer_decoder(
    vocab: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    max_len: int,
    *,
    key: Any,
    d_ff: int | None = None,
    tie: bool = False,
) -> Sequential:
    """The decoder of Vaswani et al. 2017, as a chain to a causal LM.

    ``forward(tokens)`` takes integer token ids of shape ``(T,)`` or
    ``(B, T)`` and returns logits of shape ``(T, vocab)`` or
    ``(B, T, vocab)``.

    Args:
        vocab: the vocabulary size.
        d_model: the representation width.
        n_heads: the number of heads of every block.
        n_layers: the number of decoder blocks, at least one.
        max_len: the maximum sequence length.
        key: a ``jax.random`` key, split once per layer with weights.
        d_ff: the width of the feed-forward. ``None`` uses ``4 * d_model``.
        tie: if True, the output head reuses the embedding matrix.

    Raises:
        ValueError: if ``n_layers`` is below 1, or if ``d_model`` is not
            divisible by ``n_heads``.

    References:
        Vaswani et al. 2017, Section 3.1 page 3 (decoder stacks, and
        ``d_model = 512`` as the width of every sub-layer), Table 3 page
        9 (``d_ff = 2048`` for the base model).

    Examples:
        >>> import jax
        >>> from dense_armor.utility.neural.models import transformer_decoder
        >>> m = transformer_decoder(8, 8, 2, 1, 4, key=jax.random.PRNGKey(0))
        >>> m.forward(jax.numpy.array([0, 1, 2, 3])).shape
        (4, 8)
    """
    if n_layers < 1:
        raise ValueError(f"n_layers must be at least 1, got {n_layers}")
    if d_model % n_heads != 0:
        raise ValueError(f"d_model {d_model} is not divisible by n_heads {n_heads}")
    if d_ff is None:
        d_ff = 4 * d_model
    keys = jax.random.split(key, n_layers + 2)
    emb = Embedding(vocab, d_model, keys[0])
    layers: list[Any] = [emb, AddPositions(max_len)]
    for i in range(n_layers):
        layers.append(TransformerDecoderBlock(d_model, n_heads, d_ff, keys[i + 1]))
    layers.append(LayerNorm(d_model))
    if tie:
        layers.append(_TiedHead(emb))
    else:
        layers.append(Dense(d_model, vocab, keys[n_layers + 1]))
    return Sequential(layers)
