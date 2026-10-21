"""Vision: the flow between two frames and the reduction of a table of features.

References:
    Horn, B. K., Schunck, B. G. (1981). Determining optical flow. Artificial
        Intelligence 17(1-3), 185-203: the energy and its Euler-Lagrange
        equations (not in the folder, bibliographic only).
    Ziani (2025). Methods to a Multiresolution Horn-Schunck. arXiv:2511.16535,
        equations 7 and 8: the Euler-Lagrange equations solved by iteration
        (page 4 of the PDF in the folder).
    Johnson, W. B., Lindenstrauss, J. (1984) and Ghojogh et al. (2021), section
        2.2: the Gaussian matrix with variance 1/k (not in the folder,
        bibliographic only).
    Balsubramani, A., Dasgupta, S., Freund, Y. (2013). The fast convergence of
        incremental PCA, NIPS: the target of the rule is the top eigenvector of
        the covariance (not in the folder, bibliographic only).
"""

from __future__ import annotations

import numpy as np

from dense_armor.ml.base import MLBase

__all__ = ["DenseFlow", "PCA", "RandomProjection"]

_FRAME = "it computes one record (a frame, a box, a window of recent frames) from the state kept between calls, and the steps of the docstring act on that state"
_NOREF = "the docstring gives a missing reference and the algorithm is outside the textbooks: it is stopped and reported"
STOPPED = {
    "MotionMask": "the cited pages (Horn and Schunck, Ziani eq. 6-8, TETO) give the flow and the event mask; the threshold on the median and 1.4826 MAD of the flow magnitude is written in the code of the class only",
    "GlobalMotion": "phase correlation is outside Horn and Schunck and outside Ziani (2025); its reference, Kuglin and Hines (1975), lies outside the folder",
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
