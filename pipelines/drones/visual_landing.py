"""Visual landing: is the pad where I think it is, and is the camera moving?

Three primitives of the library run on the same frames, each with its
own question:

1. :class:`GlobalMotion` (``motion.py``) estimates the global shift of
   the scene, in pixels, by phase correlation. It answers "is the
   camera moving, and by how much".
2. :class:`DenseFlow` (``dense.py``) computes the dense optical flow by
   Horn-Schunck. :class:`MotionMask` (``dense.py``) takes the flow
   field, removes the global component, and flags the pixels whose
   residual flow is above ``n_sigmas`` robust deviations. Together they
   answer "is something moving on the pad, on top of the camera
   motion".
3. :class:`FrameAnomaly` (``scene.py``) scores the patches of a frame
   against the patch memory built from the normal frames (PatchCore
   equation 6, Roth et al. 2022, quoted in the ``PatchMemory``
   docstring). It answers "does this frame contain something the
   memory has never seen".

The input protocol is the one of the library:

- ``GlobalMotion`` and ``DenseFlow`` take ``{"array": img}``, a float32
  image in ``[0, 1]``;
- ``MotionMask`` takes the flow field ``{"u": ..., "v": ...}``;
- ``FrameAnomaly`` takes ``{"frame": img}`` (via
  ``PatchFeatures._as_array``).

The synthetic stream has:

- a textured background (fine noise, ``0.5 +/- 0.01``), so phase
  correlation has a signal;
- a bright pad rectangle (``0.7``) on the background, moving with it;
- a dark "person" square (``0.0``) on the pad, that walks on the pad at
  ``PERSON_REL_VX`` pixels per frame relative to the pad, from sample
  ``person_at`` on. The person is not in the training frames, so the
  patch memory has never seen a dark square on the pad.

On this stream:

- ``GlobalMotion`` recovers the global shift exactly, sub-pixel;
- ``MotionMask``, after removing the global shift, flags the pixels of
  the person;
- ``FrameAnomaly`` scores high after the person appears.

Example:
    >>> from pipelines.drones.visual_landing import synthetic_frames
    >>> frames, truth, person_at = synthetic_frames(
    ...     n=5, shift=(1, 0), seed=0, person_at=3)
    >>> len(frames)
    5
    >>> truth[0]
    (0, 0)
    >>> truth[1]
    (1, 0)
    >>> person_at
    3
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dense_armor.utility.vision.dense import DenseFlow, MotionMask
from dense_armor.utility.vision.motion import GlobalMotion
from dense_armor.utility.vision.scene import FrameAnomaly

FRAME_H = 360
FRAME_W = 640
PAD_H = 80
PAD_W = 100
PAD_X0 = 250
PAD_Y0 = 130
PAD_LEVEL = 0.7
PERSON_H = 30
PERSON_W = 30
PERSON_REL_VX = 2
PERSON_LEVEL = 0.0


@dataclass
class VisualLandingResult:
    """Result of one run of the visual landing pipeline.

    Attributes:
        n_frames: number of frames.
        true_dx: horizontal shift per frame, in pixels.
        true_dy: vertical shift per frame, in pixels.
        person_at: sample where the person appears, or None.
        est_dx: estimated shift per frame (n,).
        est_dy: estimated shift per frame (n,).
        error_px: Euclidean error of the estimate, per frame (n,).
        mean_error_px: mean of ``error_px``.
        max_error_px: maximum of ``error_px``.
        mask_pixels: number of pixels flagged by MotionMask, per frame (n,).
        mask_delta_mean: mean of ``mask_pixels`` after ``person_at``
            minus before.
        anomaly: FrameAnomaly score, per frame (n,).
        anomaly_before_mean: mean of ``anomaly`` before ``person_at``.
        anomaly_after_mean: mean of ``anomaly`` after ``person_at``.
        time_global_ms: median time per frame of GlobalMotion, in ms.
        time_dense_ms: median time per frame of DenseFlow + MotionMask, in ms.
        time_anomaly_ms: median time per frame of FrameAnomaly, in ms.
    """

    n_frames: int
    true_dx: int
    true_dy: int
    person_at: int | None
    est_dx: np.ndarray
    est_dy: np.ndarray
    error_px: np.ndarray
    mean_error_px: float
    max_error_px: float
    mask_pixels: np.ndarray
    mask_delta_mean: float
    anomaly: np.ndarray
    anomaly_before_mean: float
    anomaly_after_mean: float
    time_global_ms: float
    time_dense_ms: float
    time_anomaly_ms: float


def _textured_background(seed: int) -> np.ndarray:
    """A textured background, fine noise around 0.5."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.5, 0.01, (FRAME_H, FRAME_W)).astype(np.float32)


def _shift_image(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    out = np.zeros_like(img)
    ys_src = slice(max(0, -dy), min(img.shape[0], img.shape[0] - dy))
    xs_src = slice(max(0, -dx), min(img.shape[1], img.shape[1] - dx))
    ys_dst = slice(max(0, dy), min(img.shape[0], img.shape[0] + dy))
    xs_dst = slice(max(0, dx), min(img.shape[1], img.shape[1] + dx))
    out[ys_dst, xs_dst] = img[ys_src, xs_src]
    return out


def synthetic_frames(
    n: int,
    shift: tuple[int, int],
    seed: int,
    person_at: int | None = None,
) -> tuple[list[dict], list[tuple[int, int]], int | None]:
    """Return ``n`` synthetic frames, the truth shift, and person_at.

    Args:
        n: number of frames. ``n`` must be >= 1.
        shift: ``(dx, dy)`` global shift per frame, in pixels. Small
            shifts (1-2 pixels) are the regime of Horn-Schunck.
        seed: RNG seed for the background texture.
        person_at: sample where the person appears, or None for no
            person.

    Returns:
        A triple ``(frames, truth, person_at)``: list of
        ``{"array": img}`` dicts (the input protocol of GlobalMotion and
        DenseFlow), list of cumulative true shifts, and the sample of
        the person (or None).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if person_at is not None and (person_at < 0 or person_at >= n):
        raise ValueError(f"person_at must be in [0, {n}), got {person_at}")
    dx, dy = int(shift[0]), int(shift[1])
    bg = _textured_background(seed)
    frames: list[dict] = []
    truth: list[tuple[int, int]] = []
    for k in range(n):
        img = _shift_image(bg, k * dx, k * dy)
        px = PAD_X0 + k * dx
        py = PAD_Y0 + k * dy
        if 0 <= px <= FRAME_W - PAD_W and 0 <= py <= FRAME_H - PAD_H:
            img[py:py+PAD_H, px:px+PAD_W] = np.maximum(
                img[py:py+PAD_H, px:px+PAD_W], PAD_LEVEL
            )
        if person_at is not None and k >= person_at:
            person_local_x = 10 + (k - person_at) * PERSON_REL_VX
            person_local_y = 25
            qx = px + person_local_x
            qy = py + person_local_y
            if (0 <= qx <= FRAME_W - PERSON_W
                    and 0 <= qy <= FRAME_H - PERSON_H):
                img[qy:qy+PERSON_H, qx:qx+PERSON_W] = PERSON_LEVEL
        frames.append({"array": img})
        truth.append((k * dx, k * dy))
    return frames, truth, person_at


def run(
    frames: list[dict],
    truth: list[tuple[int, int]] | None = None,
    person_at: int | None = None,
    anomaly_warmup: int = 20,
    anomaly_stride: int = 1,
) -> VisualLandingResult:
    """Run the visual landing pipeline on a list of frames.

    Frames are ``{"array": img}`` dicts. ``GlobalMotion`` and
    ``DenseFlow`` take the dict directly. ``FrameAnomaly`` is fed a
    ``{"frame": img}`` dict, per ``PatchFeatures``. ``MotionMask`` takes
    the flow field from ``DenseFlow``.

    ``FrameAnomaly`` is run on every ``anomaly_stride`` frame (default
    1: every frame), with a warm-up of ``anomaly_warmup`` frames (the
    first frames are learned, not scored).

    Returns:
        :class:`VisualLandingResult`.
    """
    import time

    gm = GlobalMotion()
    df = DenseFlow(use_jax=False)
    mm = MotionMask()
    fa = FrameAnomaly()
    n = len(frames)
    est_dx = np.full(n, np.nan)
    est_dy = np.full(n, np.nan)
    error_px = np.full(n, np.nan)
    mask_pixels: np.ndarray = np.zeros(n, dtype=int)
    anomaly = np.full(n, np.nan)
    t_gm: list[float] = []
    t_df: list[float] = []
    t_fa: list[float] = []

    for i, frame in enumerate(frames):
        img = frame["array"]
        # 1. GlobalMotion: {"array": img}
        t0 = time.perf_counter()
        out_gm = gm.transform_one({"array": img})
        gm.learn_one({"array": img})
        t_gm.append((time.perf_counter() - t0) * 1000.0)
        if isinstance(out_gm, dict) and "dx" in out_gm:
            est_dx[i] = float(out_gm["dx"])
            est_dy[i] = float(out_gm["dy"])

        # 2. DenseFlow + MotionMask: {"array": img} -> {"u", "v"} -> mask
        t0 = time.perf_counter()
        flow = df.transform_one({"array": img})
        df.learn_one({"array": img})
        if isinstance(flow, dict) and flow.get("u") is not None:
            m = mm.transform_one({"u": flow["u"], "v": flow["v"]})
            if isinstance(m, dict) and "mask" in m:
                mask_pixels[i] = int(np.asarray(m["mask"]).sum())
        t_df.append((time.perf_counter() - t0) * 1000.0)

        # 3. FrameAnomaly: {"frame": img}
        if i % anomaly_stride == 0:
            frame_in = {"frame": img}
            t0 = time.perf_counter()
            a = fa.transform_one(frame_in) if i >= anomaly_warmup else None
            fa.learn_one(frame_in)
            t_fa.append((time.perf_counter() - t0) * 1000.0)
            if a is not None and isinstance(a, dict) and "anomaly" in a:
                anomaly[i] = float(a["anomaly"])

    if truth is not None and len(truth) == n:
        for i in range(1, n):
            tdx = truth[i][0] - truth[i - 1][0]
            tdy = truth[i][1] - truth[i - 1][1]
            if np.isfinite(est_dx[i]) and np.isfinite(est_dy[i]):
                error_px[i] = float(
                    np.hypot(est_dx[i] - tdx, est_dy[i] - tdy)
                )

    valid = error_px[np.isfinite(error_px)]
    true_dx = truth[1][0] - truth[0][0] if truth and len(truth) > 1 else 0
    true_dy = truth[1][1] - truth[0][1] if truth and len(truth) > 1 else 0

    if person_at is not None:
        before = mask_pixels[:person_at]
        after = mask_pixels[person_at:]
        mask_delta = float(after.mean() - before.mean()) if len(before) and len(after) else 0.0
        anom_before = anomaly[:person_at]
        anom_after = anomaly[person_at:]
        ab = float(np.nanmean(anom_before)) if np.any(np.isfinite(anom_before)) else float("nan")
        aa = float(np.nanmean(anom_after)) if np.any(np.isfinite(anom_after)) else float("nan")
    else:
        mask_delta = 0.0
        ab = aa = float("nan")

    return VisualLandingResult(
        n_frames=n,
        true_dx=int(true_dx),
        true_dy=int(true_dy),
        person_at=person_at,
        est_dx=est_dx,
        est_dy=est_dy,
        error_px=error_px,
        mean_error_px=float(valid.mean()) if len(valid) else float("nan"),
        max_error_px=float(valid.max()) if len(valid) else float("nan"),
        mask_pixels=mask_pixels,
        mask_delta_mean=mask_delta,
        anomaly=anomaly,
        anomaly_before_mean=ab,
        anomaly_after_mean=aa,
        time_global_ms=float(np.median(t_gm)) if t_gm else float("nan"),
        time_dense_ms=float(np.median(t_df)) if t_df else float("nan"),
        time_anomaly_ms=float(np.median(t_fa)) if t_fa else float("nan"),
    )
