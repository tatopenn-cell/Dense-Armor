"""Four-point polyBLAMP: an alias-reduced triangle wave and hard clipping.

Source: Esqueda, Valimaki and Bilbao (2016), "Rounding corners with BLAMP", DAFx-16.
The residual is the cubic B-spline basis function of Table 1 (PDF page 3), integrated twice
(the polyBLAMP) minus the ramp ``max(t, 0)``, as the paper describes; it has the peak 7/30
at the corner (Fig. 3d) and is zero for ``|t| >= 2`` samples. Three of the four bottom rows
of Table 1 are reproduced exactly by these polynomials; the fourth row (the span ``[0, T]``)
is misread in the text layer of the PDF, so the polynomials here are derived, not copied.

* Triangle (Sec. 4.1): a trivial triangle is built from a modulo counter; at every corner the
  residual, scaled by the slope jump ``2 mu`` with ``mu = 4 f0 / fs``, is added with a
  positive sign at a valley and a negative sign at a peak; the fractional position of the
  corner follows Eq. (13).
* Hard clipping (Sec. 4.2): a cubic is fitted to the four samples around each crossing of the
  clipping level (Eq. 8), the crossing ``D`` is found by Newton-Raphson on Eq. (9) from
  ``D0 = 1.5`` (Eq. 10 and 11) and the slope there by Eq. (12). Choice stated: the correction
  at a corner is the slope jump of the clipped signal times the residual, which for a valley of
  the triangle is ``+2 mu``, the sign and size the paper gives.
"""

import numpy as np


def blamp_residual(t) -> np.ndarray:
    """Four-point polyBLAMP residual at offsets ``t`` (in samples) from the corner.

    Args:
        t: offsets; the result is 0 for ``|t| >= 2``.

    Example:
        >>> round(float(blamp_residual(0.0)), 6)
        0.233333
    """
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    a = (t >= -2) & (t < -1)
    b = (t >= -1) & (t < 0)
    c = (t >= 0) & (t < 1)
    e = (t >= 1) & (t < 2)
    u = t[a]
    out[a] = u**5 / 120 + u**4 / 12 + u**3 / 3 + 2 * u**2 / 3 + 2 * u / 3 + 4 / 15
    u = t[b]
    out[b] = -(u**5) / 40 - u**4 / 12 + u**2 / 3 + u / 2 + 7 / 30
    u = t[c]
    out[c] = u**5 / 40 - u**4 / 12 + u**2 / 3 - u / 2 + 7 / 30
    u = t[e]
    out[e] = -(u**5) / 120 + u**4 / 12 - u**3 / 3 + 2 * u**2 / 3 - 2 * u / 3 + 4 / 15
    return out


def _r(t: float) -> float:
    if t <= -2.0 or t >= 2.0:
        return 0.0
    if t < -1.0:
        return t**5 / 120 + t**4 / 12 + t**3 / 3 + 2 * t**2 / 3 + 2 * t / 3 + 4 / 15
    if t < 0.0:
        return -(t**5) / 40 - t**4 / 12 + t**2 / 3 + t / 2 + 7 / 30
    if t < 1.0:
        return t**5 / 40 - t**4 / 12 + t**2 / 3 - t / 2 + 7 / 30
    return -(t**5) / 120 + t**4 / 12 - t**3 / 3 + 2 * t**2 / 3 - 2 * t / 3 + 4 / 15


def triangle_blamp(f0: float, sr: float, n: int, phase0: float = 0.0) -> np.ndarray:
    """``n`` samples of a triangle wave in ``[-1, 1]`` with polyBLAMP corrected corners.

    The wave starts at a valley when ``phase0 = 0``; that corner is corrected too. Corners
    before the first sample are not, so the first samples are exact only up to that cut.

    Args:
        f0: fundamental in Hz, with ``0 < f0 < sr / 4``.
        sr: sample rate.
        n: number of samples.
        phase0: starting phase in cycles.

    Raises:
        ValueError: if ``f0`` is outside ``(0, sr / 4)``.
    """
    if not 0.0 < f0 < sr / 4.0:
        raise ValueError(f"f0 must be in (0, sr/4), got {f0}")
    step = f0 / sr
    p = (phase0 + np.arange(n) * step) % 1.0
    y = 1.0 - 4.0 * np.abs(p - 0.5)
    x = 2.0 * (phase0 + np.arange(-1, n) * step)
    q = np.floor(x)
    mu = 4.0 * step
    idx = np.nonzero(q[1:] > q[:-1])[0]
    for nb in idx:
        delta = (x[nb + 1] - q[nb + 1]) / (2.0 * step)
        tc = nb - delta
        sign = 1.0 if int(q[nb + 1]) % 2 == 0 else -1.0
        for s in range(nb - 2, nb + 2):
            if 0 <= s < n:
                y[s] += sign * 2.0 * mu * _r(s - tc)
    return y


def _fit(s: np.ndarray, na: int) -> tuple[float, float, float, float]:
    s0, s1, s2, s3 = s[na - 1], s[na], s[na + 1], s[na + 2]
    a = -s0 / 6 + s1 / 2 - s2 / 2 + s3 / 6
    b = s0 - 2.5 * s1 + 2 * s2 - s3 / 2
    c = -11 * s0 / 6 + 3 * s1 - 1.5 * s2 + s3 / 3
    return float(a), float(b), float(c), float(s0)


def hard_clip_blamp(x: np.ndarray, level: float) -> np.ndarray:
    """Hard clipping (Eq. 14 of the source) with the four-point polyBLAMP correction.

    At every crossing of ``level`` or ``-level`` between two samples with a sample before and
    a sample after available, the crossing position and slope are found as in the module
    docstring and the residual times the slope jump of the clipped signal is added to the four
    samples around it. Crossings at the first or last samples are left uncorrected.

    Args:
        x: samples, shape ``(n,)``.
        level: the threshold ``L`` in ``(0, 1]``.

    Raises:
        ValueError: if ``level`` is outside ``(0, 1]``.
    """
    if not 0.0 < level <= 1.0:
        raise ValueError(f"level must be in (0, 1], got {level}")
    x = np.asarray(x, dtype=float)
    y = np.sign(x) * np.minimum(np.abs(x), level)
    clipped = np.abs(x) >= level
    events: list[int] = (np.nonzero(clipped[1:-2] != clipped[2:-1])[0] + 1).tolist()
    for na in events:
        if np.sign(x[na]) * np.sign(x[na + 1]) < 0 and clipped[na] and clipped[na + 1]:
            continue
        a, b, c, e = _fit(x, na)
        rho = level if (x[na] + x[na + 1]) >= 0 else -level
        D = 1.5
        for _ in range(30):
            fD = a * D**3 + b * D**2 + c * D + e - rho
            dfD = 3 * a * D**2 + 2 * b * D + c
            if dfD == 0.0:
                break
            step = fD / dfD
            D -= step
            if abs(step) < 1e-12:
                break
        D = min(max(D, 1.0), 2.0)
        slope = 3 * a * D**2 + 2 * b * D + c
        jump = -slope if clipped[na + 1] else slope
        tc = na + (D - 1.0)
        for s in range(na - 1, na + 3):
            y[s] += jump * _r(s - tc)
    return y
