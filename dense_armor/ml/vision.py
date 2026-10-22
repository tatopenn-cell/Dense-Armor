"""Vision: the flow between two frames, the global shift, and the reduction of a table of features.

References:
    Horn, B. K., Schunck, B. G. (1981). Determining optical flow. Artificial
        Intelligence 17(1-3), 185-203: the energy and its Euler-Lagrange
        equations (not in the folder, bibliographic only).
    Ziani (2025). Methods to a Multiresolution Horn-Schunck. arXiv:2511.16535,
        equations 6, 7, 8: the Jacobi iteration (not in the folder,
        bibliographic only).
    Johnson, W. B., Lindenstrauss, J. (1984) and Ghojogh et al. (2021), section
        2.2: the Gaussian matrix with variance 1/k (not in the folder,
        bibliographic only).
    Balsubramani, A., Dasgupta, S., Freund, Y. (2013). The fast convergence of
        incremental PCA, NIPS: the target of the rule is the top eigenvector of
        the covariance (not in the folder, bibliographic only).
    ``GlobalMotion``: the algorithm of the online class ``GlobalMotion`` of
        ``dense_armor.utility.vision.motion``, on a pair of frames.
    ``MotionMask``: the algorithm of the online class ``MotionMask`` of
        ``dense_armor.utility.vision.dense``, on a flow field.
"""

from __future__ import annotations

import numpy as np

from dense_armor.ml.base import MLBase

__all__ = ["DenseFlow", "GlobalMotion", "MotionMask", "PCA", "RandomProjection"]

_FRAME = "it computes one record (a frame, a box, a window of recent frames) from the state kept between calls, and the steps of the docstring act on that state"
_NOREF = "the docstring gives a missing reference and the algorithm is outside the textbooks: it is stopped and reported"
STOPPED = {
    "TrackToSignal": _FRAME,
    "FeatureToSignal": _FRAME,
    "FrameFeatures": _FRAME,
    "FrameJointFusion": _NOREF,
    "LandmarkCleaner": "it is a causal Hampel filter with a gap rule on a landmark stream, defined on the values seen before",
    "Kinematics": _FRAME,
    "MotionEnergy": _FRAME,
    "PatchFeatures": _FRAME,
    "PatchMemory": "the reservoir mode bounds the bank on a stream with Algorithm R; the coreset reduction of PatchCore is in a paper outside the folder",
    "FrameQuality": _NOREF,
    "Sync": _NOREF,
    "EventFrame": _NOREF,
    "Registration": _FRAME,
    "Stabilizer": _FRAME,
    "DifferenceImage": "the reference is an exponential update of the frames that came before",
    "FrameNormalizer": _NOREF,
    "SceneChange": "it is a CUSUM or ADWIN test on a stream of frame statistics",
    "FrameAnomaly": "it is a CUSUM or ADWIN test on a stream of frame statistics",
    "MovingObjectTracker": "the association and the Kalman prediction use the tracks of the previous frames",
    "FalseColor": _NOREF,
    "Overlay": _NOREF,
    "BagOfVisualWords": _FRAME,
    "FrameWindow": _FRAME,
}


def _gray(img) -> np.ndarray:
    a = np.asarray(img, dtype=float)
    if a.ndim == 3 and a.shape[2] >= 3:
        return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    if a.ndim == 3:
        return a[..., 0]
    return a


def _average(a: np.ndarray) -> np.ndarray:
    return 0.25 * (
        np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1)
    )


class DenseFlow(MLBase):
    """Horn-Schunck optical flow between two frames.

    The energy ``E(u, v) = (Ix u + Iy v + It)^2 + alpha (|grad u|^2 + |grad v|^2)``
    has the Euler-Lagrange fixed point

    ``u = u_bar - Ix (Ix u_bar + Iy v_bar + It) / (alpha^2 + Ix^2 + Iy^2 + eps)``,
    ``v = v_bar - Iy (Ix u_bar + Iy v_bar + It) / (alpha^2 + Ix^2 + Iy^2 + eps)``,

    where ``u_bar`` and ``v_bar`` are the means of the four neighbours (wrap-around
    at the borders). ``Ix`` and ``Iy`` are the forward differences of the first
    frame (0 on the last column and row), ``It`` is the second frame minus the
    first, ``eps = 1e-9``, and the iteration starts from zero. The flow depends
    on the intensity scale because ``alpha`` weighs the smoothness against the
    gradient of the image; with ``normalize_intensity`` both frames are divided
    by the largest absolute intensity of the pair first.

    Args:
        alpha: smoothness weight.
        n_iter: number of Jacobi iterations.
        normalize_intensity: divide both frames by their largest absolute value.

    Example:
        >>> a = np.tile(np.sin(2 * np.pi * np.arange(16) / 16), (16, 1))
        >>> m = DenseFlow(alpha=0.1, n_iter=100).fit(a, np.roll(a, 1, axis=1))
        >>> round(float(m.u_[8, 0]), 2), float(abs(m.v_).max())
        (1.0, 0.0)
    """

    def __init__(
        self, alpha: float = 1.0, n_iter: int = 30, normalize_intensity: bool = False
    ) -> None:
        self.alpha = alpha
        self.n_iter = n_iter
        self.normalize_intensity = normalize_intensity

    def fit(self, X, y=None):  # type: ignore[override]
        g1, g2 = _gray(X), _gray(y)
        if self.normalize_intensity:
            m = max(float(np.max(np.abs(g1))), float(np.max(np.abs(g2))), 1e-12)
            g1, g2 = g1 / m, g2 / m
        Ix = np.zeros_like(g1)
        Iy = np.zeros_like(g1)
        Ix[:, :-1] = g1[:, 1:] - g1[:, :-1]
        Iy[:-1, :] = g1[1:, :] - g1[:-1, :]
        It = g2 - g1
        den = self.alpha**2 + Ix * Ix + Iy * Iy + 1e-9
        u = np.zeros_like(g1)
        v = np.zeros_like(g1)
        for _ in range(self.n_iter):
            ub, vb = _average(u), _average(v)
            p = Ix * ub + Iy * vb + It
            u = ub - Ix * p / den
            v = vb - Iy * p / den
        self.u_, self.v_, self.mag_ = u, v, np.hypot(u, v)
        return self


class MotionMask(MLBase):
    """Mask of the pixels whose flow magnitude is large, from the two components of a flow field.

    With ``mag = hypot(u, v)``, ``med`` its median and ``scale = 1.4826 * MAD``,
    a pixel is moving when ``mag > med + n_sigmas * scale`` (the mask is empty when
    ``scale <= 1e-9``); the mask is then dilated ``dilation`` times with the four
    neighbours.

    Args:
        n_sigmas: threshold in robust deviations.
        dilation: number of dilations.

    Example:
        >>> u = np.tile([0.1, 0.0], (8, 4)); u[3:5, 3:5] = 2.0
        >>> MotionMask(dilation=0).fit(u, np.zeros((8, 8))).mask_.sum()
        np.int64(4)
    """

    def __init__(self, n_sigmas: float = 3.0, dilation: int = 1) -> None:
        self.n_sigmas = n_sigmas
        self.dilation = dilation

    def fit(self, X, y=None):  # type: ignore[override]
        mag = np.hypot(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        med = float(np.median(mag))
        scale = 1.4826 * float(np.median(np.abs(mag - med)))
        mask = (
            np.zeros_like(mag, dtype=bool)
            if scale <= 1e-9
            else mag > med + self.n_sigmas * scale
        )
        for _ in range(self.dilation):
            out = mask.copy()
            out[1:, :] |= mask[:-1, :]
            out[:-1, :] |= mask[1:, :]
            out[:, 1:] |= mask[:, :-1]
            out[:, :-1] |= mask[:, 1:]
            mask = out
        self.mask_ = mask
        return self


class GlobalMotion(MLBase):
    """Global shift between two frames by phase correlation.

    Both frames are reduced to the luminance and to zero mean (and multiplied by a
    Hann window of ``window`` pixels when ``window > 0``). With ``F`` the Fourier
    transforms, the cross power spectrum is ``C = F(cur) conj(F(prev))``; the
    integer shift ``(sx, sy)`` is the position of the peak of the inverse
    transform of ``C / |C|`` (wrapped to half the frame). With ``refine``, the
    phase of ``C`` after removing the integer shift is, for a pure translation,
    ``-2 pi (kx dx + ky dy)`` at every frequency; ``(dx, dy)`` is fitted by least
    squares weighted by ``|C|`` on the bins with ``|C| > 0.05 max |C|`` and each
    part is clipped to half a pixel. The shift is the integer part plus this
    part.

    Args:
        window: Hann window size in pixels, or 0.
        refine: add the sub-pixel part.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> a = rng.normal(size=(32, 32))
        >>> m = GlobalMotion().fit(a, np.roll(a, (3, -2), axis=(0, 1)))
        >>> round(m.dx_, 3), round(m.dy_, 3)
        (-2.0, 3.0)
    """

    def __init__(self, window: int = 0, refine: bool = True) -> None:
        self.window = window
        self.refine = refine

    def _prep(self, img) -> np.ndarray:
        g = _gray(img)
        g = g - g.mean()
        if self.window > 0:
            h, w = g.shape
            wy = np.hanning(min(self.window, h))
            wx = np.hanning(min(self.window, w))
            if wy.shape[0] < h:
                p = (h - wy.shape[0]) // 2
                wy = np.concatenate([np.zeros(p), wy, np.zeros(h - wy.shape[0] - p)])
            if wx.shape[0] < w:
                p = (w - wx.shape[0]) // 2
                wx = np.concatenate([np.zeros(p), wx, np.zeros(w - wx.shape[0] - p)])
            g = g * wy[:, None] * wx[None, :]
        return g

    def fit(self, X, y=None):  # type: ignore[override]
        prev, cur = self._prep(X), self._prep(y)
        h, w = prev.shape
        self.dx_ = self.dy_ = self.peak_ = 0.0
        cross = np.fft.fft2(cur) * np.conj(np.fft.fft2(prev))
        mag = np.abs(cross)
        if mag.max() < 1e-15:
            return self
        corr = np.fft.ifft2(cross / (mag + 1e-30)).real
        py, px = np.unravel_index(int(np.argmax(corr)), corr.shape)
        sy = int(py) if py <= h // 2 else int(py) - h
        sx = int(px) if px <= w // 2 else int(px) - w
        self.peak_ = float(corr[py, px])
        self.dx_, self.dy_ = float(sx), float(sy)
        if not self.refine:
            return self
        kx = np.fft.fftfreq(w).reshape(1, -1)
        ky = np.fft.fftfreq(h).reshape(-1, 1)
        phase = np.angle(cross) + 2.0 * np.pi * (ky * sy + kx * sx)
        phase = (phase + np.pi) % (2.0 * np.pi) - np.pi
        m = mag > 0.05 * mag.max()
        if int(m.sum()) < 4:
            return self
        A = np.stack(
            [np.broadcast_to(ky, (h, w))[m], np.broadcast_to(kx, (h, w))[m]], axis=1
        )
        wts = mag[m]
        coef, *_ = np.linalg.lstsq(A * wts[:, None], phase[m] * wts, rcond=None)
        self.dy_ = float(sy + np.clip(-coef[0] / (2.0 * np.pi), -0.5, 0.5))
        self.dx_ = float(sx + np.clip(-coef[1] / (2.0 * np.pi), -0.5, 0.5))
        return self


class RandomProjection(MLBase):
    """Gaussian random projection to ``k`` dimensions.

    The matrix has shape ``(d, k)`` with i.i.d. Gaussian entries of mean zero and
    variance ``1 / k``, so that the expected squared norm of a projected row equals
    the squared norm of the row (Ghojogh et al. 2021, section 2.2); ``fit`` draws
    it with ``numpy.random.default_rng(seed).standard_normal((d, k)) / sqrt(k)``.

    Args:
        k: output dimension.
        seed: seed of the generator.

    Example:
        >>> X = np.arange(12.0).reshape(2, 6)
        >>> RandomProjection(k=3, seed=0).fit(X).transform(X).shape
        (2, 3)
    """

    def __init__(self, k: int, seed: int | None = None) -> None:
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        self.k = k
        self.seed = seed

    def fit(self, X, y=None):  # type: ignore[override]
        d = np.asarray(X, dtype=float).shape[1]
        self.W_ = np.random.default_rng(self.seed).standard_normal(
            (d, self.k)
        ) / np.sqrt(self.k)
        return self

    def transform(self, X) -> np.ndarray:
        """The rows times the matrix."""
        return np.asarray(X, dtype=float) @ self.W_


class PCA(MLBase):
    """Principal components of a table: the top ``k`` eigenvectors of its covariance.

    The rows are centred on their mean; the components are the eigenvectors of
    ``Xc^T Xc / n`` with the ``k`` largest eigenvalues, the subspace towards which
    the rule of Balsubramani et al. (2013) converges. Each component has its
    entry of largest absolute value positive.

    Args:
        k: number of components.

    Example:
        >>> X = [[1.0, 0.0], [-1.0, 0.0], [2.0, 0.1], [-2.0, -0.1]]
        >>> m = PCA(k=1).fit(X)
        >>> m.components_.round(2).tolist()
        [[1.0, 0.04]]
    """

    def __init__(self, k: int) -> None:
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        self.k = k

    def fit(self, X, y=None):  # type: ignore[override]
        A = np.asarray(X, dtype=float)
        self.mean_ = A.mean(axis=0)
        C = (A - self.mean_).T @ (A - self.mean_) / len(A)
        val, vec = np.linalg.eigh(C)
        order = np.argsort(val)[::-1][: self.k]
        comp = vec[:, order].T
        for j in range(comp.shape[0]):
            if comp[j, np.argmax(np.abs(comp[j]))] < 0:
                comp[j] = -comp[j]
        self.components_, self.explained_variance_ = comp, val[order]
        return self

    def transform(self, X) -> np.ndarray:
        """Coordinates of the rows on the components."""
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_.T
