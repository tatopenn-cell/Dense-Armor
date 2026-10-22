"""A fixed-horizon sequence encoder and decoder.

The encoder is a recurrent layer (``LSTM`` of N03), the summary is its
last output, the decoder reads the summary repeated ``horizon`` times,
and a ``Dense`` head maps every decoder output to the target. The
source is read in reverse (page 3 of Sutskever et al. 2014).

This is a fixed-horizon forecaster with the summary ``v`` as input of
every decoder step, not the initial state of the decoder of Eq.(1),
page 3 of the paper: the ``LSTM`` of N03 starts from the zero state
and exposes no gradient on its initial state, and ``Trainer`` feeds a
model one array. Our choice, stated in the docstring.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from dense_armor.utility.neural.params import Param


class Seq2Seq:
    """A recurrent encoder, a recurrent decoder, a dense head.

    ``forward(x, training=True)`` takes a source sequence of shape
    ``(T, n_in)`` or ``(B, T, n_in)``. With ``reverse_source=True`` the
    time axis is reversed before the encoder. The summary ``v`` is the
    last output of the encoder. The decoder reads ``v`` repeated
    ``horizon`` times, its outputs go through the head, so the result
    has shape ``(horizon, n_out)`` or ``(B, horizon, n_out)``.

    ``backward(g)`` returns the gradient of ``x``: the sum of the
    decoder gradients on ``v`` enters the last step of the encoder,
    then the time axis is reversed back.

    Args:
        encoder: a recurrent layer with ``forward``, ``backward``,
            ``params()``.
        decoder: the same.
        head: a ``Dense`` layer, applied to the last axis.
        horizon: the number of forecast steps, at least 1.
        reverse_source: if True, read the source in reverse (default
            True, page 3 of Sutskever et al. 2014).

    Raises:
        ValueError: if ``horizon < 1``.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.seq2seq import Seq2Seq
        >>> from dense_armor.utility.neural.recurrent import LSTM
        >>> from dense_armor.utility.neural.layers import Dense
        >>> s = Seq2Seq(LSTM(2, 3, jax.random.PRNGKey(0)),
        ...             LSTM(3, 3, jax.random.PRNGKey(1)),
        ...             Dense(3, 1, jax.random.PRNGKey(2)), horizon=2)
        >>> s.forward(jnp.ones((4, 2))).shape
        (2, 1)
    """

    def __init__(
        self,
        encoder: Any,
        decoder: Any,
        head: Any,
        horizon: int,
        reverse_source: bool = True,
    ) -> None:
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}")
        self.encoder = encoder
        self.decoder = decoder
        self.head = head
        self.horizon = horizon
        self.reverse_source = reverse_source
        self._cache: dict[str, Any] | None = None

    def __repr__(self) -> str:
        return f"Seq2Seq(horizon={self.horizon}, reverse_source={self.reverse_source})"

    def params(self) -> list[Param]:
        """The encoder, the decoder, the head, in this order."""
        return (
            list(self.encoder.params())
            + list(self.decoder.params())
            + list(self.head.params())
        )

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Encode the source, decode the summary, apply the head."""
        x = jnp.asarray(x)
        squeeze = x.ndim == 2
        x_batch = x[None] if squeeze else x
        if self.reverse_source:
            x_in = jnp.flip(x_batch, axis=1)
        else:
            x_in = x_batch
        enc_out = self.encoder.forward(x_in, training=training)
        v = enc_out[:, -1, :]
        v_rep = jnp.repeat(v[:, None, :], self.horizon, axis=1)
        dec_out = self.decoder.forward(v_rep, training=training)
        out = self.head.forward(dec_out, training=training)
        self._cache = dict(
            x=x,
            squeeze=squeeze,
            x_in=x_in,
            v=v,
            v_rep=v_rep,
            dec_out=dec_out,
        )
        return out[0] if squeeze else out

    def backward(self, g: jax.Array) -> jax.Array:
        """Return the gradient of ``x`` and add to the grads of the three parts."""
        if self._cache is None:
            raise RuntimeError("call forward() before backward()")
        c = self._cache
        g = jnp.asarray(g)
        g_batch = g[None] if c["squeeze"] else g
        d_dec_out = self.head.backward(g_batch)
        d_v_rep = self.decoder.backward(d_dec_out)
        d_v = d_v_rep.sum(axis=1)
        enc_out = self.encoder.forward(c["x_in"], training=False)
        d_enc_out = jnp.zeros_like(enc_out).at[:, -1, :].set(d_v)
        d_x_in = self.encoder.backward(d_enc_out)
        if self.reverse_source:
            d_x = jnp.flip(d_x_in, axis=1)
        else:
            d_x = d_x_in
        return d_x[0] if c["squeeze"] else d_x
