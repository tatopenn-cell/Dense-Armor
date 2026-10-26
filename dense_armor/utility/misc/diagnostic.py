"""Differential analysis of two audio signals.

``diag`` compares an original signal and a filtered version of it, both
as file paths (WAV) or as NumPy arrays, and returns a small dictionary
of statistics:

- ``fidelity``: percentage of the original energy that survives in the
  filtered signal, ``(1 - sum(diff^2) / sum(orig^2)) * 100``. A filter
  that only scales the signal by 0.98 gives 99.96.
- ``removed_energy``: variance of the difference, computed with
  :class:`dense_armor.utility.stats.RunningMoments` as if it were a
  stream.
- ``peak_distortion_db``: the peak of the removed component in dBFS,
  ``20 * log10(max|diff|)``. ``-99.0`` when the difference is zero.
- ``modulation_rate``: the fraction of samples where ``|diff|`` exceeds
  ``threshold``, in percent.
- ``verdict``: a one-word label for the fidelity. ``"micro_edits"``
  above 99.5, ``"balanced"`` between 95 and 99.5, ``"aggressive"``
  below 95.

The output has no side effect: the earlier version printed the result
to stdout and this version returns it.
"""

import os

import numpy as np
from scipy.io import wavfile

from dense_armor.utility.stats import RunningMoments


def _unit(a: np.ndarray) -> np.ndarray:
    """Samples as float32 in [-1, 1]: floats as they are, integers by their range."""
    if np.issubdtype(a.dtype, np.floating):
        return a.astype(np.float32)
    info = np.iinfo(a.dtype)
    half = (float(info.max) + 1.0) / (2.0 if info.min == 0 else 1.0)
    offset = half if info.min == 0 else 0.0
    return (a.astype(np.float32) - offset) / half


def diag(
    iorig: str | np.ndarray,
    ifilt: str | np.ndarray,
    threshold: float = 0.05,
) -> dict[str, float | str] | None:
    """Compare two signals and return the differential statistics.

    Args:
        iorig: the original signal, as a WAV path or a NumPy array.
        ifilt: the filtered signal, same kind as ``iorig``.
        threshold: absolute difference above which a sample counts as
            modified. Default ``0.05``.

    Returns:
        A dict with keys ``fidelity``, ``removed_energy``,
        ``peak_distortion_db``, ``modulation_rate``, ``verdict``. The
        first four are floats, the fifth is a string. ``None`` if one
        of the two paths does not exist.

    Raises:
        TypeError: if one argument is a path and the other an array.
        ValueError: if the two WAV files have different sample rates.
    """
    if isinstance(iorig, str) != isinstance(ifilt, str):
        raise TypeError("diag takes two file paths or two arrays, not one of each")
    if isinstance(iorig, str) and isinstance(ifilt, str):
        if not os.path.exists(iorig) or not os.path.exists(ifilt):
            return None
        sr1, d_ori = wavfile.read(iorig)
        sr2, d_flt = wavfile.read(ifilt)
        if sr1 != sr2:
            raise ValueError(f"the sample rates differ: {sr1} and {sr2}")
        v_ori = _unit(d_ori)
        v_flt = _unit(d_flt)
    else:
        v_ori = _unit(np.asarray(iorig))
        v_flt = _unit(np.asarray(ifilt))

    if len(v_ori.shape) > 1:
        v_ori = np.mean(v_ori, axis=1)
    if len(v_flt.shape) > 1:
        v_flt = np.mean(v_flt, axis=1)

    mlen = min(v_ori.shape[0], v_flt.shape[0])
    v_ori = v_ori[:mlen]
    v_flt = v_flt[:mlen]

    v_dff = v_ori - v_flt
    rm = RunningMoments()
    rm.fit([{"x": float(d)} for d in v_dff])
    v_rem = float(rm.var) if rm.count > 1 else 0.0
    fdel = (
        (1.0 - (float(np.sum(v_dff**2)) / float(np.sum(v_ori**2)))) * 100.0
        if np.sum(v_ori**2) > 0
        else 0.0
    )
    pk_df = float(np.max(np.abs(v_dff))) if v_dff.size > 0 else 0.0
    pk_db = 20.0 * float(np.log10(pk_df)) if pk_df > 0 else -99.0
    alter = float(np.mean(np.abs(v_dff) > threshold) * 100.0) if v_dff.size > 0 else 0.0

    if fdel > 99.5:
        verdict = "micro_edits"
    elif fdel >= 95.0:
        verdict = "balanced"
    else:
        verdict = "aggressive"

    return {
        "fidelity": float(fdel),
        "removed_energy": v_rem,
        "peak_distortion_db": float(pk_db),
        "modulation_rate": float(alter),
        "verdict": verdict,
    }
