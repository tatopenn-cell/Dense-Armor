# -*- coding: utf-8 -*-
"""
utility/stable_frame_filter.py
================================
`velocity_gated_stable_mask` -- restricts analysis of a signal to
points where a COMPANION REFERENCE signal indicates a stable/slow
state, so a detector isn't judging a signal at moments when a known,
independent confound (the reference moving fast) would otherwise
produce a misleading transient. Promoted from Dense-Evolution-
Discovery after validation on two independent real physical domains
(see VALIDATION below) -- this project's own rule (mirroring how
`one_sided_upper_filter` was promoted only after a real ablation
across two independent real cases) is that a filter earns library-code
status after a SECOND real case shows it still helps, not from one
experiment alone.

DESIGN CHOICE, made explicit rather than left implicit: this is
"Variant A" -- stability is judged from a REFERENCE signal, separate
from the signal actually being analyzed for anomalies (e.g. a
teleoperated robot arm's commanded leader position, separate from the
leader-follower tracking-offset signal being analyzed). This matters:
gating on the analyzed signal's OWN volatility ("Variant B") would be
circular for a signal that spikes BECAUSE of the very confound
"stable" is supposed to remove -- found and rejected exactly this trap
on a real tracking-offset signal (see VALIDATION case 1). Variant A
avoids that by keying stability off an independent reference the
analyst already has a physical reason to trust.

already_rate PARAMETER -- exists because of a real, honest failure on
a second real case: a first version always differentiated `reference`
(treating it as a position-like signal). Applied naively to gating
accelerometer-magnitude analysis by gyroscope magnitude (VALIDATION
case 2), that was WRONG -- gyroscope magnitude is ALREADY a rate
(angular velocity), so differentiating it again doesn't mean the same
thing as differentiating a position. `already_rate=True` skips the
differentiation and thresholds `reference` directly instead.

VALIDATION, two real, independent physical domains (Dense-Evolution-
Discovery, "Predicting a robot teleoperation calibration-offset regime"
/ IMU experiments):
  1. already_rate=False (position-like reference): a real teleoperated
     robot arm's leader-position channel gating analysis of its
     leader-follower tracking offset.
  2. already_rate=True (rate-like reference): real human IMU data --
     gyroscope magnitude gating analysis of accelerometer magnitude
     during a real active (walking) recording. Real, modest, honest
     effect: accelerometer-magnitude std on gated-stable frames dropped
     19% versus the full raw signal -- a real, physically sensible but
     not dramatic noise reduction (less real device rotation correlates
     with less incidental translational-acceleration variance).
"""
from typing import Optional

import numpy as np


def velocity_gated_stable_mask(
    reference: np.ndarray, vel_threshold: float, axis: Optional[int] = None,
    already_rate: bool = False,
) -> np.ndarray:
    """Boolean mask, True where `reference` indicates a "slow" or
    "stable" state at that index, trustworthy for judging a companion
    analyzed signal at the same index.

    Parameters
    ----------
    reference : np.ndarray
        The signal stability is judged FROM -- not necessarily the
        same signal being analyzed for anomalies (see module docstring's
        "Variant A" note). 1D (shape (n,)) or 2D (shape (n, d), e.g. one
        column per joint/channel).
    vel_threshold : float
        A frame is "stable" only if EVERY channel's value (already_rate=
        True) or per-step absolute change (already_rate=False) is below
        this threshold.
    axis : int, optional
        Unused for 1D input. For 2D input, the channel axis (default:
        last axis, matching an (n_frames, n_channels) layout).
    already_rate : bool, default False
        False (position/command-like reference): velocity is computed
        as `reference`'s own step-to-step absolute difference before
        thresholding.
        True (rate/velocity-like reference, e.g. angular velocity
        magnitude): `reference` is thresholded directly, no further
        differentiation.

    Returns
    -------
    np.ndarray of bool, shape (n,)
        True at indices where `reference` indicates a stable/slow state.

    Examples
    --------
    Gate a leader-follower tracking-offset signal by the leader's own
    commanded-position velocity (position-like reference, the default):

    >>> import numpy as np
    >>> leader_position = np.array([0.0, 0.0, 5.0, 5.1, 5.1])
    >>> velocity_gated_stable_mask(leader_position, vel_threshold=1.0)
    array([ True,  True, False,  True,  True])

    Gate accelerometer-magnitude analysis by real gyroscope magnitude
    (already a rate -- thresholded directly):

    >>> gyro_magnitude = np.array([0.05, 0.06, 0.9, 0.08])
    >>> velocity_gated_stable_mask(gyro_magnitude, vel_threshold=0.5, already_rate=True)
    array([ True,  True, False,  True])
    """
    reference = np.asarray(reference, dtype=np.float64)
    if reference.ndim not in (1, 2):
        raise ValueError(f"reference must be 1D or 2D, got shape {reference.shape}")

    if already_rate:
        rate = np.abs(reference)
    elif reference.ndim == 1:
        rate = np.abs(np.diff(reference, prepend=reference[:1]))
    else:
        rate = np.abs(np.diff(reference, axis=0, prepend=reference[:1]))

    if rate.ndim == 1:
        return rate < vel_threshold
    return np.all(rate < vel_threshold, axis=1)
