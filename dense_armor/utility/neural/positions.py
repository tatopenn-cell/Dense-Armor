"""The learned positional embedding.

``LearnedPositions(max_len, d_model, key)`` holds a ``Param`` ``P`` of
shape ``(max_len, d_model)`` drawn from a normal distribution with
standard deviation ``0.02`` (our choice: the paper says that learned
positional embeddings give nearly identical results to the sinusoidal
ones, and gives no scale), and adds ``P[:T]`` to the first ``T``
positions of its input.

References:
    Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
        Gomez, A. N., Kaiser, L., Polosukhin, I. (2017). Attention Is
        All You Need. arXiv:1706.03762. Page 6: "we also experimented
        with using learned positional embeddings [9] instead, and
        found that the two versions produced nearly identical results
        (see Table 3 row (E))". Page 9, Table 3 row (E): the result.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from dense_armor.utility.neural.params import Param


class LearnedPositions:
    """Add a learned vector to each position of a sequence.

    Input ``(T, d_model)`` or ``(B, T, d_model)``; output the same shape.
    The parameter ``P`` has shape ``(max_len, d_model)``; only its first
    ``T`` rows are used.

    Args:
        max_len: the longest sequence the layer accepts.
        d_model: the size of the model dimension.
        key: a ``jax.random`` key.
        std: the standard deviation of the initial ``P``, default 0.02.

    Raises:
        ValueError: on a non-positive ``max_len`` or ``d_model``, or on
            ``T > max_len`` in ``forward``.

    References:
        Vaswani et al. (2017). arXiv:1706.03762. Page 6 and page 9,
        Table 3 row (E).

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.positions import LearnedPositions
        >>> m = LearnedPositions(4, 3, jax.random.PRNGKey(0))
        >>> m.forward(jnp.zeros((4, 3))).shape
        (4, 3)
    """

    def __init__(
        self,
        max_len: int,
        d_model: int,
        key: Any,
        std: float = 0.02,
    ) -> None:
        if max_len < 1:
            raise ValueError(f"max_len must be at least 1, got {max_len}")
        if d_model < 1:
            raise ValueError(f"d_model must be at least 1, got {d_model}")
        if std <= 0.0:
            raise ValueError(f"std must be > 0, got {std}")
        self.max_len = int(max_len)
        self.d_model = int(d_model)
        self.std = float(std)
        p = std * jax.random.normal(key, (max_len, d_model))
        self.P = Param(p)
        self._cache: dict[str, Any] | None = None

    def params(self) -> list[Param]:
        """The positional table."""
        return [self.P]

    def __repr__(self) -> str:
        return f"LearnedPositions({self.max_len}, {self.d_model}, std={self.std})"

    def forward(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Return ``x + P[:T]``."""
        del training
        x = jnp.asarray(x)
        squeeze = x.ndim == 2
        x_b = x[None] if squeeze else x
        T = x_b.shape[1]
        if T > self.max_len:
            raise ValueError(
                f"sequence of length {T} is longer than max_len {self.max_len}"
            )
        y = x_b + self.P.value[:T][None]
        self._cache = dict(squeeze=squeeze, T=T)
        return y[0] if squeeze else y

    def backward(self, g: jax.Array) -> jax.Array:
        """Return ``g`` and add to ``P.grad[:T]`` the sum over the batch."""
        if self._cache is None:
            raise RuntimeError("call forward() before backward()")
        c = self._cache
        g = jnp.asarray(g)
        g_b = g[None] if c["squeeze"] else g
        T = c["T"]
        dP = g_b.sum(axis=0)
        n_new = self.max_len - T
        if n_new > 0:
            pad = jnp.zeros((n_new, self.d_model), dtype=dP.dtype)
            dP = jnp.concatenate([dP, pad], axis=0)
        self.P.grad = self.P.grad + dP
        return g
