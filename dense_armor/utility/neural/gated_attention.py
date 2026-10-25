"""A gate on a signal: look at the other stream only when it matters.

``GatedCrossAttention`` wraps a :class:`MultiHeadCrossAttention` and
multiplies each row of its output by ``1.0`` where the signal is
strictly greater than a fixed threshold and by ``0.0`` elsewhere. The
threshold is a fixed rule stated by whoever assembles the system: it
is not learned, and it is stored unchanged.

A closed step produces the zero vector and sends no gradient to the
queries, to the context or to any parameter: the gate is applied to
the output and then again to the incoming gradient before the inner
attention's ``backward``.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from dense_armor.utility.neural.cross_attention import MultiHeadCrossAttention
from dense_armor.utility.neural.params import Param


class GatedCrossAttention:
    """Cross-attention gated by a per-query signal.

    Args:
        attention: a :class:`MultiHeadCrossAttention` to gate.
        threshold: the threshold of the gate; a fixed float, not learned.

    Raises:
        ValueError: on a forward whose ``signal`` shape does not match
            the queries, or on a ``backward`` before ``forward``.
    """

    def __init__(self, attention: MultiHeadCrossAttention, threshold: float) -> None:
        self.attention = attention
        self.threshold = float(threshold)
        self.gate_: jnp.ndarray | None = None
        self._signal_shape: tuple[int, ...] | None = None
        self._squeeze: bool = False

    def __repr__(self) -> str:
        return (
            f"GatedCrossAttention(threshold={self.threshold}, "
            f"attention={self.attention!r})"
        )

    def params(self) -> list[Param]:
        """The four parameters of the wrapped attention."""
        return self.attention.params()

    def forward(
        self,
        xq: Any,
        xc: Any,
        signal: Any,
        training: bool = True,
    ):
        """Cross-attention with each query row multiplied by its gate.

        Args:
            xq: queries, shape ``(Tq, d_model)`` or ``(B, Tq, d_model)``.
            xc: context, shape ``(Tc, d_context)`` or ``(B, Tc, d_context)``.
            signal: gate signal, one number per query position, shape
                ``(Tq,)`` or ``(B, Tq)``.
            training: unused, forwarded to the inner attention.

        Returns:
            The gated output, of the shape of ``xq``.

        Raises:
            ValueError: if ``signal`` does not match the queries.
        """
        xq = jnp.asarray(xq)
        signal = jnp.asarray(signal)
        self._squeeze = xq.ndim == 2
        Tq = xq.shape[0] if self._squeeze else xq.shape[1]
        expected_2d = (Tq,)
        expected_3d = (xq.shape[0], Tq)
        if signal.shape not in (expected_2d, expected_3d):
            raise ValueError(
                f"signal shape {signal.shape}, expected {expected_2d} or {expected_3d}"
            )
        gate = jnp.where(signal > self.threshold, 1.0, 0.0)
        gate = jnp.where(jnp.isnan(signal), 0.0, gate)
        y = self.attention.forward(xq, xc, training=training)
        if self._squeeze:
            g = gate[:, None]
        else:
            g = gate[:, :, None]
        out = y * g
        self.gate_ = gate
        self._signal_shape = signal.shape
        return out

    def backward(self, g_out: Any):
        """Gradients of the queries and of the context.

        The gate is multiplied into the incoming gradient, so a closed
        step sends no gradient to ``xq``, to ``xc`` or to a parameter.

        Args:
            g_out: gradient of the output, of the shape of the output.

        Returns:
            The pair ``(dxq, dxc)``.

        Raises:
            RuntimeError: if ``forward`` has not been called.
        """
        if self.gate_ is None:
            raise RuntimeError("call forward() before backward()")
        g_out = jnp.asarray(g_out)
        if self._squeeze:
            g = self.gate_[:, None]
        else:
            g = self.gate_[:, :, None]
        return self.attention.backward(g_out * g)
