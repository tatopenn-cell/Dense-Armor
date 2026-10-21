"""Dense, Embedding, Dropout, Residual: the layers with weights.

Every layer has ``forward(x, training=True)``, ``backward(g)`` and
``params()``. ``backward`` returns the gradient of the input and
**adds** to the gradients of the parameters, so the caller must call
``Param.zero_grad()`` before a fresh pass; the contract is the same as
``Param`` in N01.

References:
    He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep Residual Learning
        for Image Recognition. arXiv:1512.03385. Page 3, Eqn.(1):
        ``y = F(x, {W_i}) + x``.
    Hinton, G. E., Srivastava, N., Krizhevsky, A., Sutskever, I.,
        Salakhutdinov, R. R. (2012). Improving neural networks by
        preventing co-adaptation of feature detectors.
        arXiv:1207.0580. Page 1: each hidden unit omitted with
        probability 0.5. Page 2: the mean network with halved
        outgoing weights.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from dense_armor.utility.neural.init import (
    he_normal,
    xavier_normal,
    zeros_bias,
)
from dense_armor.utility.neural.params import Param


def dense_forward(w: jax.Array, b: jax.Array, x: jax.Array) -> jax.Array:
    """``x @ w + b``.

    ``x`` has shape ``(..., n_in)`` and ``w`` has shape
    ``(n_in, n_out)``, so a 1-D ``x`` of length ``n_in`` gives a 1-D
    output of length ``n_out`` and a batch of shape ``(m, n_in)`` gives
    a batch of shape ``(m, n_out)``.
    """
    return x @ w + b


def dense_backward(
    w: jax.Array, x: jax.Array, g: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Gradients of ``dense_forward``: ``(dx, dw, db)``.

    With ``x`` of shape ``(..., n_in)`` and ``g`` of shape
    ``(..., n_out)``, ``dx = g @ w.T`` has the shape of ``x``,
    ``dw = x.reshape(-1, n_in).T @ g.reshape(-1, n_out)`` sums the
    batch axis, and ``db = sum over the batch of g``.
    """
    n_in = w.shape[0]
    n_out = w.shape[1]
    x2 = x.reshape(-1, n_in)
    g2 = g.reshape(-1, n_out)
    dw = x2.T @ g2
    db = g2.sum(axis=0)
    dx = g @ jnp.swapaxes(w, -1, -2)
    return dx, dw, db


class Dense:
    """A dense layer ``y = x @ W + b``.

    Args:
        n_in: number of input features.
        n_out: number of output features.
        key: a ``jax.random`` key. No default.
        init: ``"he"`` (default) for a He normal initialisation,
            ``"xavier"`` for a Xavier normal one.
        a: negative slope of the PReLU for the He case. Default 0.0.

    Raises:
        ValueError: if ``init`` is neither ``"he"`` nor ``"xavier"``.

    Examples:
        >>> import jax
        >>> from dense_armor.utility.neural.layers import Dense
        >>> d = Dense(3, 2, jax.random.PRNGKey(0))
        >>> d.W.value.shape, d.b.value.shape
        ((3, 2), (2,))
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        key: Any,
        init: str = "he",
        a: float = 0.0,
    ) -> None:
        if init == "he":
            w0 = he_normal(key, (n_in, n_out), a)
        elif init == "xavier":
            w0 = xavier_normal(key, (n_in, n_out))
        else:
            raise ValueError(f"init must be 'he' or 'xavier', got {init!r}")
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.init = init
        self.a = float(a)
        self.W = Param(w0)
        self.b = Param(zeros_bias(n_out))
        self.x_: jax.Array | None = None

    def __repr__(self) -> str:
        return (
            f"Dense(n_in={self.n_in}, n_out={self.n_out}, "
            f"init={self.init!r}, a={self.a})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dense):
            return NotImplemented
        return (
            self.n_in == other.n_in
            and self.n_out == other.n_out
            and self.init == other.init
            and self.a == other.a
            and jnp.array_equal(self.W.value, other.W.value)
            and jnp.array_equal(self.b.value, other.b.value)
        )

    __hash__ = None  # type: ignore[assignment]

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Return ``x @ W + b``; ``training`` is accepted and ignored."""
        del training
        self.x_ = x
        return dense_forward(self.W.value, self.b.value, x)

    def backward(self, g: jax.Array) -> jax.Array:
        """Return ``dx`` and add to ``W.grad`` and ``b.grad``."""
        if self.x_ is None:
            raise RuntimeError("call forward() before backward()")
        dx, dw, db = dense_backward(self.W.value, self.x_, g)
        self.W.grad = self.W.grad + dw
        self.b.grad = self.b.grad + db
        return dx

    def params(self) -> list[Param]:
        """The two ``Param`` objects, ``W`` then ``b``."""
        return [self.W, self.b]


class Embedding:
    """A lookup table ``E[ids]``.

    The entries start from a zero-mean Gaussian of standard deviation
    ``1/sqrt(dim)``. This is our choice: the N02a sources do not state
    an initial scale for an embedding table, and ``1/sqrt(dim)`` keeps
    the variance of a lookup roughly independent of the embedding
    dimension when the ids are drawn uniformly.

    Args:
        n_tokens: number of rows of the table.
        dim: dimension of each row.
        key: a ``jax.random`` key. No default.

    Raises:
        ValueError: on ``forward`` if an id is outside ``[0, n_tokens)``.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.layers import Embedding
        >>> e = Embedding(10, 4, jax.random.PRNGKey(0))
        >>> e.forward(jnp.array([1, 1, 3])).shape
        (3, 4)
    """

    def __init__(self, n_tokens: int, dim: int, key: Any) -> None:
        self.n_tokens = int(n_tokens)
        self.dim = int(dim)
        std = 1.0 / float(dim) ** 0.5
        e0 = std * jax.random.normal(key, (n_tokens, dim))
        self.E = Param(e0)
        self.ids_: jax.Array | None = None

    def __repr__(self) -> str:
        return f"Embedding(n_tokens={self.n_tokens}, dim={self.dim})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Embedding):
            return NotImplemented
        return (
            self.n_tokens == other.n_tokens
            and self.dim == other.dim
            and jnp.array_equal(self.E.value, other.E.value)
        )

    __hash__ = None  # type: ignore[assignment]

    def forward(self, ids: jax.Array, training: bool = True) -> jax.Array:
        """Return ``E[ids]`` for an integer array of any shape."""
        del training
        ids = jnp.asarray(ids, dtype=jnp.int32)
        if bool(jnp.any(ids < 0)) or bool(jnp.any(ids >= self.n_tokens)):
            raise ValueError(
                f"ids must lie in [0, {self.n_tokens}), "
                f"got min={int(ids.min())} max={int(ids.max())}"
            )
        self.ids_ = ids
        return self.E.value[ids]

    def backward(self, g: jax.Array) -> None:
        """Add ``g`` into the rows of ``E.grad``; repeated ids add up."""
        if self.ids_ is None:
            raise RuntimeError("call forward() before backward()")
        rows = np.asarray(self.ids_).reshape(-1)
        g2 = np.asarray(g).reshape(len(rows), self.dim)
        add = np.zeros_like(np.asarray(self.E.grad))
        for i, r in enumerate(rows):
            add[r] += g2[i]
        self.E.grad = self.E.grad + jnp.asarray(add)
        return None

    def params(self) -> list[Param]:
        """The one ``Param``, the table ``E``."""
        return [self.E]


class Dropout:
    """Random dropout with the scaling of Hinton et al. 2012.

    In training mode, each element of ``x`` is kept with probability
    ``1 - p`` and the kept elements are multiplied by ``1/(1 - p)``.
    In evaluation mode the output is the input. The scaling at training
    time is our choice: the paper (page 2) halves the outgoing weights
    of the mean network at test time, and multiplying the kept
    activations by ``1/(1 - p)`` gives the same expected value and
    leaves the evaluation mode as the identity.

    Args:
        p: probability of dropping an element, in ``[0, 1)``.
        key: a ``jax.random`` key. No default.

    Raises:
        ValueError: if ``p < 0`` or ``p >= 1``.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.layers import Dropout
        >>> d = Dropout(0.5, jax.random.PRNGKey(0))
        >>> bool((d.forward(jnp.ones(10), training=False) == 1.0).all())
        True
    """

    def __init__(self, p: float, key: Any) -> None:
        if not (0.0 <= p < 1.0):
            raise ValueError(f"p must be in [0, 1), got {p}")
        self.p = float(p)
        self.key = key
        self.mask_: jax.Array | None = None

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dropout):
            return NotImplemented
        return self.p == other.p

    __hash__ = None  # type: ignore[assignment]

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Random dropout in training mode, identity in evaluation."""
        if not training:
            return x
        self.key, sub = jax.random.split(self.key)
        keep = 1.0 - self.p
        mask = (jax.random.uniform(sub, x.shape) < keep).astype(x.dtype)
        self.mask_ = mask
        return x * mask / keep

    def backward(self, g: jax.Array) -> jax.Array:
        """``g * mask / (1 - p)`` with the mask of the last training call."""
        if self.mask_ is None:
            raise RuntimeError("call forward() in training mode before backward()")
        return g * self.mask_ / (1.0 - self.p)

    def params(self) -> list[Param]:
        """No parameters."""
        return []


class Residual:
    """``y = x + inner.forward(x)`` (He et al. 2015, Eqn.(1) page 3).

    ``inner`` is any object with ``forward(x, training=True)``,
    ``backward(g)`` and ``params()``.

    Args:
        inner: the wrapped layer.

    Raises:
        ValueError: if ``inner.forward(x)`` and ``x`` have different
            shapes.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.layers import Dense, Residual
        >>> r = Residual(Dense(4, 4, jax.random.PRNGKey(0)))
        >>> r.forward(jnp.zeros(4)).shape
        (4,)
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.x_: jax.Array | None = None

    def __repr__(self) -> str:
        return f"Residual(inner={self.inner!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Residual):
            return NotImplemented
        return self.inner == other.inner

    __hash__ = None  # type: ignore[assignment]

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """``x + inner.forward(x)``."""
        y = self.inner.forward(x, training=training)
        if y.shape != x.shape:
            raise ValueError(
                f"inner.forward returned {y.shape}, expected the input shape {x.shape}"
            )
        self.x_ = x
        return x + y

    def backward(self, g: jax.Array) -> jax.Array:
        """``g + inner.backward(g)``."""
        if self.x_ is None:
            raise RuntimeError("call forward() before backward()")
        return g + self.inner.backward(g)

    def params(self) -> list[Param]:
        """The parameters of ``inner``."""
        return list(self.inner.params())
