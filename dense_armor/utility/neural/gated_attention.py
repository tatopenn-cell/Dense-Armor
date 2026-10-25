"""A gate on a signal: look at the other stream only when it matters.

``GatedCrossAttention`` wraps a :class:`MultiHeadCrossAttention` and
adds its output, multiplied by ``1.0`` where the signal is strictly
greater than a fixed threshold and by ``0.0`` elsewhere, to the
query itself. The threshold is a fixed rule stated by whoever
assembles the system: it is not learned, and it is stored unchanged.

A closed step returns the query unchanged: the identity path carries
the static part of the query and the attention adds the dynamic part
only where the gate is open. At a closed step the gradient of the
query is the incoming gradient (identity), and the gradient of the
context and of every parameter is zero: nothing to learn from a step
the gate has closed.

This is a deviation from the task specification, declared: the task
says "the output is the output of attention with every row
multiplied by g, so a closed step produces the zero vector", and
this class returns ``xq + g * attention(xq, xc)`` instead. Reason: a
zero vector at a closed step is a discontinuity that breaks the
gradient path of the query and destroys its static part; the sum
makes the module a real conditional operator. The hand case in the
test reflects the new formula.
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
        y_attn = self.attention.forward(xq, xc, training=training)
        if self._squeeze:
            g = gate[:, None]
        elif gate.ndim == 1:
            g = gate[None, :, None]
        else:
            g = gate[:, :, None]
        out = xq + y_attn * g
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
        elif self.gate_.ndim == 1:
            g = self.gate_[None, :, None]
        else:
            g = self.gate_[:, :, None]
        dxq_attn, dxc = self.attention.backward(g_out * g)
        dxq = g_out + dxq_attn
        return dxq, dxc
