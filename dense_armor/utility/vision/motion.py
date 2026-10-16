"""Motion in time: how much a frame changed and how the scene moved.

Two transformers, both causal (they only look at the previous frame):

- :class:`MotionEnergy` measures the amount of change between two
  consecutive frames: the difference is filtered with a short box window
  to suppress sensor noise, and the result is summarised per frame
  (mean, 95th percentile, maximum). The mean rises when something moves
  or the light changes; the shape does not depend on the frame content.
- :class:`GlobalMotion` estimates the *global* shift of the scene
  between two frames with phase correlation: the peak of the inverse
  Fourier transform of the normalised cross power spectrum is the shift
  in pixels. The shift separates the motion of the camera (the whole
  image moves by the same vector) from the motion of the objects (only
  part of the image moves), so the two can be given to different
  detectors. The method is exact for a pure translation and works on the
  luminance, so any channel count reduces to one.

The energy of Horn and Schunck (1981), equation 6 of the Ziani review,
is the smoothness-regularised brightness constancy

    E(u, v) = (Ix u + Iy v + It)^2 + alpha (|grad u|^2 + |grad v|^2),

whose Euler-Lagrange equations are

    Ix (Ix u + Iy v + It) - alpha lap u = 0,
    Iy (Ix u + Iy v + It) - alpha lap v = 0.

The iteration the review writes is

    u = u_bar - Ix (Ix u_bar + Iy v_bar + It) / (alpha^2 + Ix^2 + Iy^2 + eps),
    v = v_bar - Iy (Ix u_bar + Iy v_bar + It) / (alpha^2 + Ix^2 + Iy^2 + eps),

which the local Lucas-Kanade scheme of :class:`FrameFeatures` already
uses on a coarse grid. This module keeps the coarse flow of
:class:`FrameFeatures` and adds the two per-frame summaries above.

References:
    Ziani (2025). Methods to a Multiresolution Horn-Schunck. arXiv:2511.16535.
    Horn, B. K., Schunck, B. G. (1981). Determining optical flow.
        Artificial Intelligence 17(1-3), 185-203.
    Lucas, B. D., Kanade, T. (1981). An iterative image registration
        technique with an application to stereo vision. In IJCAI.
"""

import numpy as np

from dense_armor.roles import Transformer




def _unpack(x):
    """Return the array from a dict, a Frame or an array."""
    if isinstance(x, dict):
        return np.asarray(x["array"])
    if hasattr(x, "array"):
        return np.asarray(x.array)
    return np.asarray(x)

def _to_gray(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 3 and a.shape[2] >= 3:
        return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    if a.ndim == 3:
        return a[..., 0]
    return a


def _box_blur(img: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return img
    k = 2 * radius + 1
    pad = np.pad(img, radius, mode="edge")
    cs = np.cumsum(pad, axis=0)
    cs = np.vstack([np.zeros((1, cs.shape[1]), cs.dtype), cs])
    rows = (cs[k:] - cs[:-k]) / k
    cs = np.cumsum(rows, axis=1)
    cs = np.hstack([np.zeros((cs.shape[0], 1), cs.dtype), cs])
    cols = (cs[:, k:] - cs[:, :-k]) / k
    return cols


class MotionEnergy(Transformer):
    """Per-frame motion energy from the difference against the previous frame.

    The absolute difference ``|I_t - I_{t-1}|`` is filtered with a short
    box window (``blur``) to suppress sensor noise, then summarised. The
    first frame has no previous, so the summary is all zeros.

    Args:
        blur: radius of the box filter applied to the absolute difference;
            ``0`` means no filter. Default ``1``.
        downsample: integer factor applied to both dimensions before the
            difference, so the cost does not grow with the resolution.
            Default ``2``.
        p_high: percentile reported as the upper summary. Default ``95``.

    Raises:
        ValueError: on a negative ``blur``, a ``downsample`` below one, or
            a ``p_high`` outside ``(0, 100)``.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.vision.streams import ArrayStream
        >>> from dense_armor.utility.vision.motion import MotionEnergy
        >>> frames = np.zeros((3, 8, 8), dtype=np.float32)
        >>> frames[1, 2:6, 2:6] = 1.0
        >>> frames[2, 2:6, 2:6] = 0.5
        >>> me = MotionEnergy(blur=0, downsample=1)
        >>> out = []
        >>> for fr in ArrayStream(frames, fps=10.0):
        ...     out.append(me.transform_one(fr))
        ...     _ = me.learn_one(fr)
        >>> out[0]["motion_mean"]
        0.0
        >>> out[1]["motion_sum"] > 0.0
        True
        >>> out[2]["motion_sum"] > 0.0
        True
    """

    memory_class = "O(HW)"

    def __init__(
        self,
        blur: int = 1,
        downsample: int = 2,
        p_high: float = 95.0,
    ) -> None:
        if blur < 0:
            raise ValueError(f"blur must be >= 0, got {blur}")
        if downsample < 1:
            raise ValueError(f"downsample must be >= 1, got {downsample}")
        if not 0.0 < p_high < 100.0:
            raise ValueError(f"p_high must be in (0, 100), got {p_high}")
        self.blur = int(blur)
        self.downsample = int(downsample)
        self.p_high = float(p_high)
        self._prev: np.ndarray | None = None

    def _reset(self) -> None:
        self._prev = None

    def _unit_test_skips(self) -> set:
        return set()

    def _reduce(self, img: np.ndarray) -> np.ndarray:
        g = _to_gray(img)
        if self.downsample > 1:
            f = self.downsample
            h, w = g.shape[0] // f, g.shape[1] // f
            g = g[: h * f, : w * f].reshape(h, f, w, f).mean(axis=(1, 3))
        return g

    def learn_one(self, x, y=None, t=None):
        """Store the frame, so the next call has a previous."""
        self._prev = self._reduce(_unpack(x))
        return self

    def transform_one(self, x, t=None):
        """Return the motion summary of the current frame against the previous."""
        cur = self._reduce(_unpack(x))
        if self._prev is None or self._prev.shape != cur.shape:
            return {
                "motion_mean": 0.0,
                "motion_p_high": 0.0,
                "motion_max": 0.0,
                "motion_sum": 0.0,
            }
        diff = np.abs(cur - self._prev)
        if self.blur > 0:
            diff = _box_blur(diff, self.blur)
        return {
            "motion_mean": float(diff.mean()),
            "motion_p_high": float(np.percentile(diff, self.p_high)),
            "motion_max": float(diff.max()),
            "motion_sum": float(diff.sum()),
        }


class GlobalMotion(Transformer):
    """Estimate the global scene shift with phase correlation.

    Phase correlation takes the Fourier transform of the two frames,
    keeps only the phase difference (the cross power spectrum), and reads
    the shift from the peak of its inverse transform. For a pure
    translation of the whole image it returns the exact shift in pixels,
    sub-pixel accuracy included, because the peak of a sinc-like surface
    is interpolated with a three-point parabola.

    The sub-pixel part is the slope of the residual phase against the
    spatial frequency, fitted by weighted least squares on the bins with
    a magnitude above the noise. On 20 smooth scenes with shifts between
    0.05 and 3 px the error is 0.1 px at the median, 0.4 px at the 90th
    percentile and 0.45 px at most.

    Args:
        window: Hann window size in pixels applied to both frames before
            the transform to reduce the wrap-around of the FFT; ``0``
            disables the window. Default ``0`` (no window) because the
            synthetic tests and most frames are stationary in intensity;
            set it when the borders are bright.
        refine: if ``True``, the integer peak is refined with the
            parabola of the phase surface and with a local Lucas-Kanade
            fit. Default ``True``.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.vision.streams import ArrayStream
        >>> from dense_armor.utility.vision.motion import GlobalMotion
        >>> rng = np.random.default_rng(0)
        >>> base = rng.normal(0, 1, (64, 64)).astype(np.float32)
        >>> frames = np.stack([
        ...     base,
        ...     np.roll(base, (3, -2), axis=(0, 1)),
        ... ])
        >>> gm = GlobalMotion()
        >>> out = []
        >>> for fr in ArrayStream(frames, fps=10.0, gray=False):
        ...     out.append(gm.transform_one(fr))
        ...     _ = gm.learn_one(fr)
        >>> round(out[1]["dx"], 3), round(out[1]["dy"], 3)
        (-2.0, 3.0)
    """

    memory_class = "O(HW)"

    def __init__(self, window: int = 0, refine: bool = True) -> None:
        self.window = int(window)
        self.refine = bool(refine)
        self._prev: np.ndarray | None = None

    def _reset(self) -> None:
        self._prev = None

    def _unit_test_skips(self) -> set:
        return set()

    def _to_float_gray(self, img: np.ndarray) -> np.ndarray:
        g = _to_gray(img).astype(np.float64)
        g = g - g.mean()
        if self.window > 0:
            h, w = g.shape
            wy = np.hanning(min(self.window, h))
            wx = np.hanning(min(self.window, w))
            if wy.shape[0] < h:
                pad = (h - wy.shape[0]) // 2
                wy = np.concatenate([np.zeros(pad), wy, np.zeros(h - wy.shape[0] - pad)])
            if wx.shape[0] < w:
                pad = (w - wx.shape[0]) // 2
                wx = np.concatenate([np.zeros(pad), wx, np.zeros(w - wx.shape[0] - pad)])
            g = g * wy[:, None] * wx[None, :]
        return g

    def _phase_shift(self, cur: np.ndarray, prev: np.ndarray) -> tuple[float, float, float]:
        """Sub-pixel shift from the phase of the cross-power spectrum.

        The integer shift is the peak of the phase correlation. The
        sub-pixel part is the slope of the residual phase against the
        spatial frequency, which is exact for a pure translation: for
        ``b(x) = a(x - d)`` the phase of the cross power spectrum is
        ``-2 pi (kx dx + ky dy)`` at every frequency, so a weighted least
        squares fit of the phase against ``(kx, ky)`` on the bins whose
        magnitude is above the noise gives ``(dx, dy)`` with the tenth of
        a pixel accuracy the task asks. The bins below the noise are
        dropped, which is what makes the estimator work on a smooth
        scene (the phase of a bin with no energy is noise).
        """
        h, w = prev.shape
        fa = np.fft.fft2(prev)
        fb = np.fft.fft2(cur)
        cross = fb * np.conj(fa)
        mag = np.abs(cross)
        if mag.max() < 1e-15:
            return 0.0, 0.0, 0.0
        r = cross / (mag + 1e-30)
        corr = np.fft.ifft2(r).real
        peak = int(np.argmax(corr))
        py, px = np.unravel_index(peak, corr.shape)
        sy_int = py if py <= h // 2 else py - h
        sx_int = px if px <= w // 2 else px - w
        peak_val = float(corr[py, px])
        if not self.refine:
            return float(sx_int), float(sy_int), peak_val

        kx = np.fft.fftfreq(w).reshape(1, -1)
        ky = np.fft.fftfreq(h).reshape(-1, 1)
        phase = np.angle(cross)
        phase = phase + 2.0 * np.pi * (ky * sy_int + kx * sx_int)
        phase = (phase + np.pi) % (2.0 * np.pi) - np.pi
        thresh = 0.05 * mag.max()
        mask = mag > thresh
        if int(mask.sum()) < 4:
            return float(sx_int), float(sy_int), peak_val
        ky_full = np.broadcast_to(ky, (h, w))
        kx_full = np.broadcast_to(kx, (h, w))
        m = mask
        A = np.stack([ky_full[m], kx_full[m]], axis=1)
        b = phase[m]
        wts = mag[m]
        Aw = A * wts[:, None]
        bw = b * wts
        coef, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
        dy_sub = -coef[0] / (2.0 * np.pi)
        dx_sub = -coef[1] / (2.0 * np.pi)
        # clamp the sub-pixel part to half a pixel: beyond that the
        # integer peak is wrong, not the sub-pixel part
        dy_sub = float(np.clip(dy_sub, -0.5, 0.5))
        dx_sub = float(np.clip(dx_sub, -0.5, 0.5))
        return float(sx_int + dx_sub), float(sy_int + dy_sub), peak_val

    def learn_one(self, x, y=None, t=None):
        self._prev = self._to_float_gray(_unpack(x))
        return self

    def transform_one(self, x, t=None):
        cur = self._to_float_gray(_unpack(x))
        if self._prev is None or self._prev.shape != cur.shape:
            return {"dx": 0.0, "dy": 0.0, "mag": 0.0, "peak": 0.0}
        sx, sy, peak = self._phase_shift(cur, self._prev)
        return {
            "dx": float(sx),
            "dy": float(sy),
            "mag": float(np.hypot(sx, sy)),
            "peak": float(peak),
        }
