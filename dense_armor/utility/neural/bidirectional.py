"""Bidirectional recurrent layer.

``Bidirectional(forward_layer, backward_layer)`` runs a forward layer on
the sequence and a backward layer on the reversed sequence, then
concatenates the two outputs on the last axis (Eq.(8) and Eq.(9), page
2 of Graves, Mohamed and Hinton 2013).

The output layer of Eq.(10) of the same page is not part of the class:
a :class:`~dense_armor.utility.neural.layers.Dense` after it gives the
same sum of two linear projections.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from dense_armor.utility.neural.params import Param


class Bidirectional:
    """A forward and a backward recurrent layer, concatenated.

    Args:
        forward_layer: a recurrent layer (``RNN``, ``GRU`` or ``LSTM``)
            with ``forward``, ``backward`` and ``params``.
        backward_layer: the same, read from the last step to the first.

    Raises:
        ValueError: if either layer has no ``forward`` or ``backward``
            method, or no ``params`` method.

    References:
        Graves, A., Mohamed, A.-R., Hinton, G. (2013). Speech
        recognition with deep recurrent neural networks.
        arXiv:1303.5778. Page 2, Eq.(8) and Eq.(9): the forward and the
        backward hidden states, concatenated. Eq.(10): the output
        layer.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.recurrent import RNN
        >>> from dense_armor.utility.neural.bidirectional import Bidirectional
        >>> f = RNN(1, 1, jax.random.PRNGKey(0))
        >>> b = RNN(1, 1, jax.random.PRNGKey(1))
        >>> f.W.value = jnp.array([[1.0]])
        >>> f.U.value = jnp.array([[0.5]])
        >>> f.b.value = jnp.array([0.0])
        >>> b.W.value = jnp.array([[1.0]])
        >>> b.U.value = jnp.array([[0.5]])
        >>> b.b.value = jnp.array([0.0])
        >>> m = Bidirectional(f, b)
        >>> m.forward(jnp.array([[1.0], [0.0], [-1.0]])).shape
        (3, 2)
    """

    def __init__(self, forward_layer: Any, backward_layer: Any) -> None:
        for name, layer in [
            ("forward_layer", forward_layer),
            ("backward_layer", backward_layer),
        ]:
            for method in ("forward", "backward", "params"):
                if not hasattr(layer, method):
                    raise ValueError(
                        f"{name} has no {method}() method; it must be a "
                        f"recurrent layer of dense_armor.utility.neural.recurrent"
                    )
        self.forward_layer = forward_layer
        self.backward_layer = backward_layer
        self._cache: dict[str, Any] | None = None

    def params(self) -> list[Param]:
        """The parameters of the forward layer and then of the backward layer."""
        return list(self.forward_layer.params()) + list(self.backward_layer.params())

    def __repr__(self) -> str:
        return f"Bidirectional({self.forward_layer!r}, {self.backward_layer!r})"

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Forward on ``x``, backward on ``flip(x)``, output concatenated."""
        x = jnp.asarray(x)
        squeeze = x.ndim == 2
        x_b = x[None] if squeeze else x
        x_rev = jnp.flip(x_b, axis=1)
        y_fwd = self.forward_layer.forward(x, training=training)
        y_bwd_rev = self.backward_layer.forward(x_rev, training=training)
        y_fwd_b = y_fwd[None] if squeeze else y_fwd
        y_bwd = jnp.flip(y_bwd_rev, axis=1)
        out = jnp.concatenate([y_fwd_b, y_bwd], axis=-1)
        self._cache = dict(squeeze=squeeze)
        return out[0] if squeeze else out

    def backward(self, g: jax.Array) -> jax.Array:
        """Return ``dx`` and add to the grads of the two layers."""
        if self._cache is None:
            raise RuntimeError("call forward() before backward()")
        c = self._cache
        g = jnp.asarray(g)
        if c["squeeze"]:
            g = g[None]
        h1 = self.forward_layer.n_hidden
        g_fwd = g[..., :h1]
        g_bwd = g[..., h1:]
        dx_fwd = self.forward_layer.backward(g_fwd)
        g_bwd_rev = jnp.flip(g_bwd, axis=1)
        dx_bwd_rev = self.backward_layer.backward(g_bwd_rev)
        dx_bwd = jnp.flip(dx_bwd_rev, axis=1)
        dx = dx_fwd + dx_bwd
        return dx[0] if c["squeeze"] else dx
