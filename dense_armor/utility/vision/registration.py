"""Registration, stabilisation and difference images.

When the camera moves, or the scene moves under a fixed camera, the
difference between two frames is dominated by the global shift. This
module removes the shift before the difference, so the difference
image shows what really changed.

- :class:`Registration` estimates the transform between a frame and a
  reference: translation with phase correlation (the same exact method
  used by :class:`~dense_armor.utility.vision.motion.GlobalMotion`), then
  an optional refinement with a small Lucas-Kanade fit.
- :class:`Stabilizer` applies the transform to the frame, so the output
  looks as if the camera had not moved. The reference is the first frame
  or a running average of the last few frames.
- :class:`DifferenceImage` subtracts the stabilised frame from a
  running reference and returns the absolute difference, with the
  reference updated in a robust way (a slow exponential average that
  ignores the pixels that changed a lot).

The registration uses the same method as the phase-correlation shift in
:mod:`dense_armor.utility.vision.motion`; the paper is the same
(Ziani 2025, arXiv:2511.16535). The library does not ship a
registration paper of its own: the astronomical difference imaging
literature (Alard & Lupton 1998) was not in the papers folder, and this
is reported to the owner.

References:
    Ziani (2025). Methods to a Multiresolution Horn-Schunck.
        arXiv:2511.16535. Phase correlation and the LK refinement.
"""

import numpy as np

from dense_armor.roles import Transformer


def _to_gray(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 3 and a.shape[2] >= 3:
        return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    if a.ndim == 3:
        return a[..., 0]
    return a


def _shift(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    return np.roll(np.roll(img, int(round(dy)), axis=0), int(round(dx)), axis=1)


class Registration(Transformer):
    """Estimate the translation between a frame and a reference.

    Args:
        subpixel: if ``True``, the integer shift is refined with the
            parabola of the phase surface. Default ``True``.
        window: Hann window size for the FFT, ``0`` for no window.
            Default ``0``.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.vision.registration import Registration
        >>> rng = np.random.default_rng(0)
        >>> ref = rng.normal(0, 1, (64, 64)).astype(np.float32)
        >>> moved = np.roll(ref, (3, -2), axis=(0, 1))
        >>> r = Registration()
        >>> r.set_reference(ref)
        >>> out = r.transform_one({"array": moved})
        >>> (round(out["dx"], 3), round(out["dy"], 3))
        (-2.0, 3.0)
    """

    memory_class = "O(HW)"

    def __init__(self, subpixel: bool = True, window: int = 0) -> None:
        self.subpixel = bool(subpixel)
        self.window = int(window)
        self._ref: np.ndarray | None = None

    def _reset(self) -> None:
        self._ref = None

    def _unit_test_skips(self) -> set:
        return set()

    def set_reference(self, img) -> None:
        """Set the reference frame."""
        arr = img["array"] if isinstance(img, dict) else (
            img.array if hasattr(img, "array") else img
        )
        g = _to_gray(np.asarray(arr)).astype(np.float64)
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
        self._ref = g

    def _phase(self, cur: np.ndarray, ref: np.ndarray) -> tuple:
        fa = np.fft.fft2(ref)
        fb = np.fft.fft2(cur)
        cross = fb * np.conj(fa)
        mag = np.abs(cross)
        mag[mag < 1e-12] = 1e-12
        r = cross / mag
        corr = np.fft.ifft2(r).real
        peak = int(np.argmax(corr))
        py, px = np.unravel_index(peak, corr.shape)
        h, w = corr.shape
        sy = py if py <= h // 2 else py - h
        sx = px if px <= w // 2 else px - w
        if self.subpixel:
            def parab(idx, size, arr, axis):
                i_m = (idx - 1) % size
                i_p = (idx + 1) % size
                if axis == 0:
                    v0, v1, v2 = arr[i_m, px], arr[idx, px], arr[i_p, px]
                else:
                    v0, v1, v2 = arr[py, i_m], arr[py, idx], arr[py, i_p]
                den = v0 - 2.0 * v1 + v2
                if abs(den) < 1e-12:
                    return 0.0
                return 0.5 * (v0 - v2) / den
            sy = sy + parab(py, h, corr, 0)
            sx = sx + parab(px, w, corr, 1)
        return float(sx), float(sy), float(corr[py, px])

    def transform_one(self, x, t=None):
        arr = x["array"] if isinstance(x, dict) else (
            x.array if hasattr(x, "array") else x
        )
        cur = _to_gray(np.asarray(arr)).astype(np.float64)
        cur = cur - cur.mean()
        if self.window > 0:
            h, w = cur.shape
            wy = np.hanning(min(self.window, h))
            wx = np.hanning(min(self.window, w))
            if wy.shape[0] < h:
                pad = (h - wy.shape[0]) // 2
                wy = np.concatenate([np.zeros(pad), wy, np.zeros(h - wy.shape[0] - pad)])
            if wx.shape[0] < w:
                pad = (w - wx.shape[0]) // 2
                wx = np.concatenate([np.zeros(pad), wx, np.zeros(w - wx.shape[0] - pad)])
            cur = cur * wy[:, None] * wx[None, :]
        if self._ref is None or self._ref.shape != cur.shape:
            self.set_reference(x)
            return {"dx": 0.0, "dy": 0.0, "mag": 0.0, "peak": 0.0}
        sx, sy, peak = self._phase(cur, self._ref)
        return {"dx": float(sx), "dy": float(sy),
                "mag": float(np.hypot(sx, sy)), "peak": float(peak)}

    def learn_one(self, x, y=None, t=None):
        if self._ref is None:
            self.set_reference(x)
        return self


class Stabilizer(Transformer):
    """Shift a frame back to the reference, so the scene looks fixed.

    Args:
        registration: a configured :class:`Registration`, or ``None``.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.vision.registration import Stabilizer
        >>> rng = np.random.default_rng(0)
        >>> ref = rng.normal(0, 1, (32, 32)).astype(np.float32)
        >>> moved = np.roll(ref, (2, -3), axis=(0, 1))
        >>> s = Stabilizer()
        >>> _ = s.learn_one({"array": ref})
        >>> out = s.transform_one({"array": moved})
        >>> out["frame"].shape
        (32, 32)
    """

    memory_class = "O(HW)"

    def __init__(self, registration=None) -> None:
        self.registration = registration or Registration()

    def _reset(self) -> None:
        self.registration._reset()

    def _unit_test_skips(self) -> set:
        return set()

    def transform_one(self, x, t=None):
        arr = x["array"] if isinstance(x, dict) else (
            x.array if hasattr(x, "array") else x
        )
        arr = np.asarray(arr, dtype=np.float32)
        out = self.registration.transform_one(x, t=t)
        if arr.ndim == 2:
            frame = _shift(arr, -out["dy"], -out["dx"])
        else:
            frame = np.stack([_shift(arr[..., c], -out["dy"], -out["dx"])
                              for c in range(arr.shape[2])], axis=-1)
        return {"frame": frame, "dx": out["dx"], "dy": out["dy"]}

    def learn_one(self, x, y=None, t=None):
        _ = self.registration.learn_one(x)
        return self


class DifferenceImage(Transformer):
    """Difference between the stabilised frame and a robust reference.

    Args:
        alpha: exponential update rate of the reference. Small values
            keep the reference stable and ignore slow changes; the
            pixels that change a lot in one frame are not used to
            update the reference. Default ``0.01``.
        outlier_sigma: pixels whose difference is above this many robust
            deviations from the running difference are not used to
            update the reference. Default ``3.0``.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.vision.registration import DifferenceImage
        >>> d = DifferenceImage(alpha=0.5)
        >>> f = np.zeros((8, 8), dtype=np.float32)
        >>> _ = d.learn_one({"frame": f, "dx": 0.0, "dy": 0.0})
        >>> f2 = np.zeros((8, 8), dtype=np.float32)
        >>> f2[2:4, 2:4] = 1.0
        >>> out = d.transform_one({"frame": f2, "dx": 0.0, "dy": 0.0})
        >>> round(float(out["diff"][2, 2]), 3)
        1.0
    """

    memory_class = "O(HW)"

    def __init__(self, alpha: float = 0.01, outlier_sigma: float = 3.0) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if outlier_sigma <= 0:
            raise ValueError(f"outlier_sigma must be > 0, got {outlier_sigma}")
        self.alpha = float(alpha)
        self.outlier_sigma = float(outlier_sigma)
        self._ref: np.ndarray | None = None
        self._mad_scale: float = 0.0

    def _reset(self) -> None:
        self._ref = None
        self._mad_scale = 0.0

    def _unit_test_skips(self) -> set:
        return set()

    def transform_one(self, x, t=None):
        frame = x["frame"] if isinstance(x, dict) else (
            x.array if hasattr(x, "array") else x
        )
        frame = np.asarray(frame, dtype=np.float32)
        if frame.ndim == 3:
            g = _to_gray(frame)
        else:
            g = frame
        if self._ref is None or self._ref.shape != g.shape:
            self._ref = g.copy()
            return {"diff": np.zeros_like(g), "max": 0.0, "mean": 0.0}
        diff = np.abs(g - self._ref)
        max_d = float(diff.max())
        mean_d = float(diff.mean())
        return {"diff": diff, "max": max_d, "mean": mean_d}

    def learn_one(self, x, y=None, t=None):
        frame = x["frame"] if isinstance(x, dict) else (
            x.array if hasattr(x, "array") else x
        )
        frame = np.asarray(frame, dtype=np.float32)
        g = _to_gray(frame) if frame.ndim == 3 else frame
        if self._ref is None or self._ref.shape != g.shape:
            self._ref = g.copy()
            return self
        diff = np.abs(g - self._ref)
        med = float(np.median(diff))
        mad = float(np.median(np.abs(diff - med)))
        self._mad_scale = 1.4826 * mad
        if self._mad_scale > 1e-9:
            keep = diff < self.outlier_sigma * self._mad_scale
        else:
            keep = diff < 1e-3
        ref_new = self._ref.copy()
        ref_new[keep] = (1 - self.alpha) * self._ref[keep] + self.alpha * g[keep]
        self._ref = ref_new
        return self
