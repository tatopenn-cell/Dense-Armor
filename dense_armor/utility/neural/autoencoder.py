"""An autoencoder: an encoder, a decoder, and the reconstruction error.

The class joins two models, the ``encoder`` and the ``decoder``, into
one model with ``forward``, ``backward`` and ``params()``. The ``Trainer``
trains it with ``y = X`` and ``MSE``. The reconstruction error is the mean squared difference between
the input and the output of the autoencoder, one value per sample.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from dense_armor.utility.neural.params import Param


class Autoencoder:
    """An encoder and a decoder, joined end to end.

    ``forward(x, training=True)`` is ``decoder.forward(encoder.forward(x,
    training), training)``; ``backward(g)`` is ``encoder.backward(
    decoder.backward(g))``; ``params()`` lists the parameters of the
    encoder and then of the decoder.

    ``encode(x)`` returns the code, ``decode(z)`` the reconstruction,
    and ``reconstruction_error(x)`` the mean over the last axis of
    ``(x - forward(x, training=False)) ** 2``: an array with one value
    per sample for a batch, a scalar for one sample.

    Args:
        encoder: a model with ``forward(x, training=True)``,
            ``backward(g)`` and ``params()``.
        decoder: the same, with the input size of the code and the
            output size of the input.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.layers import Dense
        >>> from dense_armor.utility.neural.chain import Sequential
        >>> from dense_armor.utility.neural.autoencoder import Autoencoder
        >>> enc = Sequential([Dense(2, 1, jax.random.PRNGKey(0))])
        >>> dec = Sequential([Dense(1, 2, jax.random.PRNGKey(1))])
        >>> a = Autoencoder(enc, dec)
        >>> a.encode(jnp.ones(2)).shape
        (1,)
        >>> a.reconstruction_error(jnp.ones(2)).shape
        ()
    """

    def __init__(self, encoder: Any, decoder: Any) -> None:
        self.encoder = encoder
        self.decoder = decoder

    def __repr__(self) -> str:
        return f"Autoencoder({self.encoder!r}, {self.decoder!r})"

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Return the reconstruction of ``x``."""
        z = self.encoder.forward(x, training=training)
        return self.decoder.forward(z, training=training)

    def backward(self, g: jax.Array) -> jax.Array:
        """Return the gradient of the input."""
        gz = self.decoder.backward(g)
        return self.encoder.backward(gz)

    def params(self) -> list[Param]:
        """The parameters of the encoder, then of the decoder."""
        return list(self.encoder.params()) + list(self.decoder.params())

    def encode(self, x: jax.Array) -> jax.Array:
        """Return the code of ``x``."""
        return self.encoder.forward(x, training=False)

    def decode(self, z: jax.Array) -> jax.Array:
        """Return the reconstruction of the code ``z``."""
        return self.decoder.forward(z, training=False)

    def reconstruction_error(self, x: jax.Array) -> jax.Array:
        """Mean over the last axis of ``(x - forward(x, training=False)) ** 2``."""
        x = jnp.asarray(x)
        rec = self.forward(x, training=False)
        return jnp.mean((x - rec) ** 2, axis=-1)
