"""Frequency, cents and note coordinates in equal temperament.

Source: A. Falkenberg (2026), "Real-Time Arbitrary Resampling for Phase-Continuous
Microtuning in Audio Streams" (preprint by one author), Algorithm 1, PDF page 3:
``C = 1200 log2(f / fref)``, ``a = round(C / 100)``, ``delta = C - 100 a``,
``o = floor(a / 12)``, ``j = a - 12 o``; back, ``f = fref 2^(o + j/12 + delta/1200)``.
The reference frequency is an argument with no default; 440 Hz is the usual concert pitch
(Berendes, Maman, Mueller 2025, PDF page 2, ISO 16).
"""

import numpy as np


def freq_to_cents(f, fref: float):
    """Cents of ``f`` above ``fref``: ``1200 log2(f / fref)``.

    Raises:
        ValueError: if ``f`` or ``fref`` is not positive.

    Example:
        >>> float(freq_to_cents(880.0, 440.0))
        1200.0
    """
    f = np.asarray(f, dtype=float)
    if np.any(f <= 0) or fref <= 0:
        raise ValueError("frequencies must be positive")
    return 1200.0 * np.log2(f / fref)


def cents_to_freq(c, fref: float):
    """Frequency ``fref 2^(c / 1200)``."""
    if fref <= 0:
        raise ValueError("fref must be positive")
    return fref * 2.0 ** (np.asarray(c, dtype=float) / 1200.0)


def freq_to_note(f: float, fref: float, names: tuple[str, ...] | None = None):
    """Note coordinates ``(j, o, delta)`` of a frequency, as in Algorithm 1.

    ``j`` is the semitone in ``0..11`` above the note at ``fref`` in its octave ``o`` (an
    octave number relative to ``fref``), ``delta`` the residual in cents, within half a
    semitone. Tie-breaking, which the algorithm leaves to the caller: round half to even
    (``numpy.round``).

    Args:
        f: frequency, positive.
        fref: reference frequency, positive.
        names: optional twelve names; the name of ``j`` is appended to the result.

    Returns:
        ``(j, o, delta)`` or ``(j, o, delta, name)`` when ``names`` is given.

    Raises:
        ValueError: if ``names`` does not have twelve entries.
    """
    if names is not None and len(names) != 12:
        raise ValueError("names must have twelve entries")
    c = float(freq_to_cents(f, fref))
    a = int(np.round(c / 100.0))
    delta = c - 100.0 * a
    o = a // 12
    j = a - 12 * o
    return (j, o, delta) if names is None else (j, o, delta, names[j])


def note_to_freq(j: int, o: int, delta: float, fref: float) -> float:
    """Frequency ``fref 2^(o + j/12 + delta/1200)`` of note coordinates.

    Raises:
        ValueError: if ``j`` is outside ``0..11`` or ``fref`` is not positive.
    """
    if not 0 <= j <= 11:
        raise ValueError(f"j must be in 0..11, got {j}")
    if fref <= 0:
        raise ValueError("fref must be positive")
    return float(fref * 2.0 ** (o + j / 12.0 + delta / 1200.0))
