"""Bag of visual words: a k-means vocabulary over image patches.

The frame is split into square patches (default 8 by 8 pixels). Each
patch is described with a small histogram of oriented gradients
(Huang and Huang 2017), the same construction used by the
frame features. An online k-means learns a vocabulary of ``n_words``
visual words over the patch descriptors, one word per cluster. A
frame becomes a normalised histogram of visual words: every patch
votes for its nearest word and the histogram is divided by the number
of patches. The result is a fixed-length signature of the frame, the
missing piece of the visual pipeline.

The vocabulary is grown online, so the same estimator can be fed
patch descriptors from many frames without ever storing them.

Examples:
    >>> import numpy as np
    >>> from dense_armor.utility.vision.vocabulary import BagOfVisualWords
    >>> bow = BagOfVisualWords(n_words=3, patch_size=8, n_bins=4)
    >>> img = np.zeros((16, 16), dtype=np.float32)
    >>> img[:, 8:] = 1.0
    >>> out = bow.transform_one({"frame": img})
    >>> len(out["hist"])
    3

References:
    Huang, C., Huang, J. (2017). A fast HOG descriptor using lookup table
        and integral image. arXiv:1703.06256.
"""
from typing import Any

import numpy as np

from dense_armor.roles import Transformer
from dense_armor.utility.cluster.kmeans import OnlineKMeans


def _as_array(x: Any) -> np.ndarray | None:
    if hasattr(x, "array"):
        return np.asarray(x.array, dtype=np.float32)
    if isinstance(x, dict):
        if "frame" in x:
            return np.asarray(x["frame"], dtype=np.float32)
        return None
    try:
        return np.asarray(x, dtype=np.float32)
    except (TypeError, ValueError):
        return None


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] >= 3:
        return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    return img


class BagOfVisualWords(Transformer):
    """Visual vocabulary over image patches and per-frame histogram.

    ``learn_one`` extracts patches from a frame and feeds their
    descriptors to an internal :class:`OnlineKMeans`. ``transform_one``
    returns a normalised histogram of word counts. A frame that is
    empty, has a wrong shape, or contains a non-finite value is
    skipped and counted in ``n_missing``.

    Memory grows with the vocabulary (``n_words`` centres of
    ``2 * n_bins`` floats each) and the number of patches per frame,
    both bounded, so the footprint is fixed.

    Args:
        n_words: size of the visual vocabulary.
        patch_size: side of a square patch in pixels.
        n_bins: number of gradient orientation bins inside a patch.
        seed: passed to the internal :class:`OnlineKMeans`.

    Raises:
        ValueError: if any parameter is not positive.
    """

    budget_s = 1e-3
    memory_class = "O(window)"

    def __init__(
        self,
        n_words: int = 8,
        patch_size: int = 8,
        n_bins: int = 8,
        seed: int = 0,
    ) -> None:
        if n_words < 1:
            raise ValueError(f"n_words must be >= 1, got {n_words}")
        if patch_size < 2:
            raise ValueError(f"patch_size must be >= 2, got {patch_size}")
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}")
        self.n_words = n_words
        self.patch_size = patch_size
        self.n_bins = n_bins
        self.seed = seed
        self.kmeans_ = OnlineKMeans(k=n_words, seed=seed)
        self.n_missing_ = 0

    def _unit_test_skips(self) -> set:
        return set()

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    def _descriptors(self, img: np.ndarray) -> np.ndarray:
        lum = _to_gray(img)
        if lum.ndim != 2 or lum.size == 0 or not np.isfinite(lum).all():
            return np.zeros((0, self.n_bins))
        ix = np.zeros_like(lum)
        iy = np.zeros_like(lum)
        ix[:, 1:-1] = 0.5 * (lum[:, 2:] - lum[:, :-2])
        iy[1:-1, :] = 0.5 * (lum[2:, :] - lum[:-2, :])
        mag = np.sqrt(ix * ix + iy * iy)
        ang = np.mod(np.arctan2(iy, ix), 2.0 * np.pi)
        bins = np.floor(ang / (2.0 * np.pi / self.n_bins)).astype(np.intp)
        bins = np.clip(bins, 0, self.n_bins - 1)
        ps = self.patch_size
        h, w = lum.shape
        ny, nx = h // ps, w // ps
        descs: list[np.ndarray] = []
        for j in range(ny):
            for i in range(nx):
                sl = (slice(j * ps, (j + 1) * ps), slice(i * ps, (i + 1) * ps))
                m = mag[sl].ravel()
                b = bins[sl].ravel()
                counts = np.bincount(b, weights=m, minlength=self.n_bins)
                nrm = float(np.linalg.norm(counts)) + 1e-8
                descs.append(counts / nrm)
        if not descs:
            return np.zeros((0, self.n_bins))
        return np.stack(descs, axis=0)

    def learn_one(
        self, x: Any, y: Any = None, t: float | None = None
    ) -> "BagOfVisualWords":
        """Update the vocabulary with the patches of one frame."""
        self._time_step(t)
        img = _as_array(x)
        if img is None:
            self.n_missing_ += 1
            return self
        descs = self._descriptors(img)
        if descs.shape[0] == 0:
            self.n_missing_ += 1
            return self
        for d in descs:
            self.kmeans_.learn_one({"_d": d.tolist()})
        return self

    def transform_one(self, x: Any, t: float | None = None) -> dict:
        """Return ``{"hist": [...]}``, the normalised word histogram."""
        img = _as_array(x)
        if img is None:
            self.n_missing_ += 1
            return {"hist": [0.0] * self.n_words}
        descs = self._descriptors(img)
        if descs.shape[0] == 0 or self.kmeans_.centers_ is None:
            return {"hist": [0.0] * self.n_words}
        hist = np.zeros(self.n_words)
        for d in descs:
            j = self.kmeans_.predict_one({"_d": d.tolist()})
            hist[j] += 1.0
        s = hist.sum()
        if s > 0:
            hist = hist / s
        return {"hist": hist.tolist()}
