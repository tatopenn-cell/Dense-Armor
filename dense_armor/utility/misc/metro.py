"""Logarithmic scaling of a tiny value into a visible range.

A gradient, a residual or a rate can shrink to a magnitude so small that
the float64 representation loses most of its digits, or flushes to zero.
``Metro`` multiplies the value by a power of ten that brings it into a
visible range (``enc``), and divides it back (``dec``) once the caller is
done with it. The sign is preserved: a negative value is scaled by the
factor of its absolute value. Zero stays zero with factor 1.
"""

import numpy as np


class Metro:
    """Logarithmic scaler with an anti-underflow floor.

    Args:
        val_e: target order of magnitude, in base 10. A value whose order
            of magnitude is smaller than ``val_e`` is multiplied by a
            power of ten to reach ``val_e``. Default ``-4.0``.
    """

    def __init__(self, val_e: float = -4.0) -> None:
        self.val_e = float(val_e)
        self.eps64 = np.finfo(np.float64).eps

    def enc(self, raw_v: float) -> tuple:
        """Bring ``raw_v`` up to the target order of magnitude.

        Returns ``(v_new, fact)``: ``v_new`` is ``raw_v * fact`` and
        ``fact`` is the power of ten applied. ``dec(v_new, fact)``
        restores the original value.

        Zero returns ``(0.0, 1.0)``.
        """
        v64 = np.float64(raw_v)
        if v64 == 0.0:
            return np.float64(0.0), np.float64(1.0)

        exp10 = np.log10(abs(v64))
        shift = self.val_e - exp10
        fact = np.float64(10**shift)
        v_new = v64 * fact
        return v_new, fact

    def dec(self, v_new: float, fact: float) -> float:
        """Restore the original value from ``enc``."""
        return np.float64(v_new) / np.float64(fact)
