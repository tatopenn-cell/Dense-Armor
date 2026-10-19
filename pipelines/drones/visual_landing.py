"""Visual landing: is the pad where I think it is, and is the camera moving?

A drone descends toward a landing pad. A downward camera captures
640x360 frames at 10 Hz. The pad is a bright rectangle on the ground.
Between two frames the camera moves, so the pad shifts by a few pixels.
The pipeline estimates the shift from the frames and compares it to the
truth.

The estimator is :class:`GlobalMotion` of ``dense_armor.utility.vision``,
which uses **phase correlation**: the Fourier transform of the two
frames, the cross power spectrum, and the shift from the peak of its
inverse transform, sub-pixel accuracy included. The theory is the one
of Ziani et al. 2025 (arXiv:2511.16535, "Investigating Optical Flow
Computation: From Local to Global Methods", section 3.1, pages 3-4),
which covers Horn-Schunck as the dense optical-flow method and phase
correlation as one way to read a global shift. The library's own
implementation uses phase correlation, not Horn-Schunck; the page
declares this in Details.

The stream is synthetic: a black frame, a bright pad, and a known
shift per frame. The pad moves with constant velocity, so the shift
is the same every frame; the estimator should recover it exactly for
a pure translation.

Motion energy, from :class:`MotionEnergy`, is also reported: it is the
number a landing controller uses to know whether the pad is moving in
the frame (camera motion) or the pad itself is moving (target motion),
but here both are the same because the pad is the only moving object.

Example:
    >>> from pipelines.drones.visual_landing import synthetic_frames
    >>> frames, truth = synthetic_frames(n=3, shift=(4, -2), seed=0)
    >>> len(frames)
    3
    >>> truth[0]
    (0, 0)
    >>> truth[1]
    (4, -2)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dense_armor.utility.vision.motion import GlobalMotion, MotionEnergy

FRAME_H = 360
FRAME_W = 640
PAD_H = 60
PAD_W = 80
PAD_X0 = 100
PAD_Y0 = 150


@dataclass
class VisualLandingResult:
    """Result of one run of the visual landing pipeline.

    Attributes:
        n_frames: number of frames.
        true_dx: the planted horizontal shift per frame, in pixels.
        true_dy: the planted vertical shift per frame, in pixels.
        est_dx: estimated horizontal shift per frame, per frame (n,).
        est_dy: estimated vertical shift per frame, per frame (n,).
        error_px: Euclidean error between the estimate and the truth,
            per frame, in pixels (n,).
        mean_error_px: mean of ``error_px`` over the frames that have
            an estimate.
        max_error_px: maximum of ``error_px``.
        motion_sum: per-frame motion sum from MotionEnergy.
        time_per_frame_ms: median time per frame, in milliseconds,
            for GlobalMotion.transform_one plus learn_one.
    """

    n_frames: int
    true_dx: int
    true_dy: int
    est_dx: np.ndarray
    est_dy: np.ndarray
    error_px: np.ndarray
    mean_error_px: float
    max_error_px: float
    motion_sum: np.ndarray
    time_per_frame_ms: float


def synthetic_frames(
    n: int, shift: tuple[int, int], seed: int,
) -> tuple[list[dict], list[tuple[int, int]]]:
    """Return ``n`` synthetic frames and the true shift of each one.

    Args:
        n: number of frames. ``n`` must be at least 1.
        shift: ``(dx, dy)`` shift per frame, in pixels, from one frame
            to the next. Positive ``dx`` moves right, positive ``dy``
            moves down.
        seed: RNG seed (unused: the frames are deterministic).

    Returns:
        A pair ``(frames, truth)``: a list of ``{"array": img}`` dicts,
        and a list of ``(dx, dy)`` true shifts per frame. The first
        shift is ``(0, 0)``.

    Raises:
        ValueError: if ``n < 1``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    dx, dy = int(shift[0]), int(shift[1])
    frames: list[dict] = []
    truth: list[tuple[int, int]] = []
    for k in range(n):
        img: np.ndarray = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        x = PAD_X0 + k * dx
        y = PAD_Y0 + k * dy
        if 0 <= x <= FRAME_W - PAD_W and 0 <= y <= FRAME_H - PAD_H:
            img[y:y+PAD_H, x:x+PAD_W] = 255
        frames.append({"array": img})
        truth.append((k * dx, k * dy))
    return frames, truth


def run(
    frames: list[dict],
    truth: list[tuple[int, int]] | None = None,
) -> VisualLandingResult:
    """Run the visual landing pipeline on a list of frames.

    Args:
        frames: list of ``{"array": img}`` dicts, one per frame, from
            :func:`synthetic_frames` or a real camera.
        truth: list of ``(dx, dy)`` planted shifts, one per frame,
            from :func:`synthetic_frames`. If ``None``, the pipeline
            still estimates the shifts but reports ``nan`` errors.

    Returns:
        :class:`VisualLandingResult`.
    """
    import time

    gm = GlobalMotion()
    me = MotionEnergy()
    n = len(frames)
    est_dx = np.full(n, np.nan)
    est_dy = np.full(n, np.nan)
    error_px = np.full(n, np.nan)
    motion_sum = np.full(n, np.nan)
    latencies: list[float] = []

    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        out_gm = gm.transform_one(frame)
        gm.learn_one(frame)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        if isinstance(out_gm, dict) and "dx" in out_gm and "dy" in out_gm:
            est_dx[i] = float(out_gm["dx"])
            est_dy[i] = float(out_gm["dy"])
        out_me = me.transform_one(frame)
        me.learn_one(frame)
        if isinstance(out_me, dict) and "motion_sum" in out_me:
            motion_sum[i] = float(out_me["motion_sum"])

    if truth is not None and len(truth) == n:
        for i in range(n):
            # the truth is the cumulative shift; the estimate is the
            # per-frame shift (between frame i-1 and frame i), so
            # compare it with the per-frame change of the truth
            if i == 0:
                continue
            tdx = truth[i][0] - truth[i - 1][0]
            tdy = truth[i][1] - truth[i - 1][1]
            if np.isfinite(est_dx[i]) and np.isfinite(est_dy[i]):
                error_px[i] = float(np.hypot(est_dx[i] - tdx, est_dy[i] - tdy))

    valid_err = error_px[np.isfinite(error_px)]
    mean_err = float(valid_err.mean()) if len(valid_err) else float("nan")
    max_err = float(valid_err.max()) if len(valid_err) else float("nan")
    median_lat = float(np.median(latencies)) if latencies else float("nan")

    true_dx = truth[1][0] - truth[0][0] if truth and len(truth) > 1 else 0
    true_dy = truth[1][1] - truth[0][1] if truth and len(truth) > 1 else 0

    return VisualLandingResult(
        n_frames=n,
        true_dx=int(true_dx),
        true_dy=int(true_dy),
        est_dx=est_dx,
        est_dy=est_dy,
        error_px=error_px,
        mean_error_px=mean_err,
        max_error_px=max_err,
        motion_sum=motion_sum,
        time_per_frame_ms=median_lat,
    )
