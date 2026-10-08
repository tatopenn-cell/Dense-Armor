"""Local patch descriptors and a bounded patch memory for anomaly detection.

The PatchCore method (Roth et al. 2021) detects anomalies by comparing
each local patch of a test image against a memory bank of normal patches.
An image is anomalous as soon as one of its patches is anomalous, so the
image score is the maximum distance of a patch to its nearest neighbour
in the bank; the per-patch distances give a localisation map. The paper
builds its patch descriptors from the intermediate feature maps of an
ImageNet-pretrained network. This module replaces them with native
descriptors and keeps the rest of the method.

- :class:`PatchFeatures` (a :class:`~dense_armor.roles.Transformer`) cuts
  the grey frame into overlapping patches of ``patch`` x ``patch`` pixels
  every ``stride`` pixels. Each patch is described by an oriented-gradient
  histogram (same gradients and L2 normalisation as
  :class:`~dense_armor.utility.vision.features.FrameFeatures`), plus the
  mean and the contrast of the patch. The descriptor of each patch is
  then averaged with the descriptors of its
  ``neighbourhood`` x ``neighbourhood`` neighbours, as in PatchCore
  equations 1 and 2: the local aggregation widens the effective receptive
  field and makes the descriptor more robust to small spatial variations.
  ``transform_one`` returns a dict with the array of patch descriptors
  (``n_patches`` x ``d``) under ``"descriptors"`` and an integer array
  of grid positions (``n_patches`` x 2) under ``"positions"``.
- :class:`PatchMemory` (an :class:`~dense_armor.roles.AnomalyDetector`)
  collects the descriptors of the normal training views (PatchCore eq. 4)
  and bounds them at ``max_size`` with one of two explicit modes:

  - ``mode="reservoir"`` (default): every descriptor is offered to a
    reservoir of size ``max_size``, replaced with probability
    ``max_size / n_seen`` (reservoir sampling, Algorithm R). Memory never
    grows past ``max_size`` and the cost is amortised ``O(1)`` per
    descriptor. This is the streaming default.
  - ``mode="coreset"``: :meth:`learn_one` keeps every patch of the
    training images; :meth:`finalize` (called once at the end of
    training) reduces the bank to ``max_size`` with the greedy minimax
    facility location selection of PatchCore eq. 5. :meth:`score_one`
    before :meth:`finalize` raises :class:`RuntimeError`. Cost of
    :meth:`finalize` is ``O(n * max_size * d)``. Until :meth:`finalize`
    the memory holds every training patch; this is the batch step of
    the paper, done once at the end of training.

The nearest-neighbour distances are computed in numpy with a matrix
form for the squared norms (``||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b``),
chunked over the queries so the temporary array stays bounded.

References
----------
Roth, K., Pemula, L., Zepeda, J., Scholkopf, B., Brox, T., Gehler, P.
    (2021). Towards total recall in industrial anomaly detection.
    arXiv:2106.08265.
Chang, X., Ye, Z., et al. (2024). RAD: A Realistic Multi-View
    Benchmark for Pose-Agnostic Anomaly Detection. arXiv:2410.00713.
Huang, C., Huang, J. (2017). A fast HOG descriptor using lookup table
    and integral image. arXiv:1703.06256.
"""

from typing import Any

import numpy as np

from dense_armor.roles import AnomalyDetector, Transformer
from dense_armor.utility.vision.features import _as_array, _to_gray


def _gradients(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Central differences of ``img``, same kernel as the frame features."""
    ix = np.zeros_like(img)
    iy = np.zeros_like(img)
    ix[:, 1:-1] = 0.5 * (img[:, 2:] - img[:, :-2])
    iy[1:-1, :] = 0.5 * (img[2:, :] - img[:-2, :])
    return ix, iy


class PatchFeatures(Transformer):
    """Cut a frame into patches and describe each one locally.

    Args:
        patch: side of each patch in pixels. Default 16.
        stride: step between patch top-left corners, in pixels. Must be
            positive and not larger than ``patch`` (overlap is allowed).
            Default 8.
        n_bins: number of angular bins in each patch histogram.
            Default 9.
        neighbourhood: side of the neighbourhood used to aggregate each
            patch descriptor, in patches. Default 3 (the value chosen by
            PatchCore, Figure 4). Must be >= 1.
        eps: numerical guard for the histogram normalisation.

    ``budget_s`` = 0.15 s: :meth:`transform_one` measured 80.7 ms on a
    320 x 240 frame on a Colab CPU, about half the budget.

    Raises:
        ValueError: if a parameter is out of range.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.vision.patches import PatchFeatures
        >>> pf = PatchFeatures(patch=8, stride=4, n_bins=4, neighbourhood=1)
        >>> img = np.zeros((16, 16), dtype=np.float32)
        >>> out = pf.transform_one(img)
        >>> out["descriptors"].shape
        (9, 6)
        >>> out["positions"].shape
        (9, 2)
    """

    budget_s = 0.15
    memory_class = "O(1)"

    def __init__(
        self,
        patch: int = 16,
        stride: int = 8,
        n_bins: int = 9,
        neighbourhood: int = 3,
        eps: float = 1e-8,
    ) -> None:
        if patch < 1:
            raise ValueError(f"patch must be >= 1, got {patch}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if stride > patch:
            raise ValueError(
                f"stride ({stride}) must not be larger than patch ({patch})"
            )
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}")
        if neighbourhood < 1:
            raise ValueError(f"neighbourhood must be >= 1, got {neighbourhood}")
        self.patch = patch
        self.stride = stride
        self.n_bins = n_bins
        self.neighbourhood = neighbourhood
        self.eps = eps
        self.n_missing_ = 0

    def _reset(self) -> None:
        self.n_missing_ = 0

    def _unit_test_skips(self) -> set:
        return set()

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    @property
    def output_dim(self) -> int:
        return self.n_bins + 2

    def _patch_grid(self, shape: tuple[int, int]) -> tuple[int, int] | None:
        h, w = shape
        p = self.patch
        s = self.stride
        if h < p or w < p:
            return None
        ny = (h - p) // s + 1
        nx = (w - p) // s + 1
        if ny < 1 or nx < 1:
            return None
        return ny, nx

    def _describe_all(self, lum: np.ndarray, ny: int, nx: int) -> np.ndarray:
        p = self.patch
        s = self.stride
        d = self.output_dim
        ix, iy = _gradients(lum)
        mag = np.sqrt(ix * ix + iy * iy)
        ang = np.mod(np.arctan2(iy, ix), 2.0 * np.pi)
        bin_float = np.floor(ang / (2.0 * np.pi / self.n_bins))
        bin_idx = np.clip(bin_float, 0, self.n_bins - 1).astype(np.intp)
        desc = np.zeros((ny, nx, d), dtype=np.float64)
        for j in range(ny):
            y = j * s
            sy = slice(y, y + p)
            for i in range(nx):
                x = i * s
                sx = slice(x, x + p)
                m = mag[sy, sx].ravel()
                b = bin_idx[sy, sx].ravel()
                hist = np.bincount(b, weights=m, minlength=self.n_bins).astype(
                    np.float64
                )
                nrm = float(np.linalg.norm(hist)) + self.eps
                hist = hist / nrm
                vals = lum[sy, sx]
                desc[j, i, : self.n_bins] = hist
                desc[j, i, self.n_bins] = float(vals.mean())
                desc[j, i, self.n_bins + 1] = float(vals.std())
        return desc

    def _aggregate(self, desc: np.ndarray) -> np.ndarray:
        ny, nx, d = desc.shape
        n = self.neighbourhood
        if n <= 1:
            return desc.copy()
        hl = (n - 1) // 2
        hh = n // 2
        out = np.zeros_like(desc)
        for j in range(ny):
            j0 = max(0, j - hl)
            j1 = min(ny, j + hh + 1)
            for i in range(nx):
                i0 = max(0, i - hl)
                i1 = min(nx, i + hh + 1)
                block = desc[j0:j1, i0:i1].reshape(-1, d)
                out[j, i] = block.mean(axis=0)
        return out

    def learn_one(
        self, x: Any, y: Any = None, t: float | None = None
    ) -> "PatchFeatures":
        """No-op: the transformer is stateless.

        Args:
            x: unused.
            y: unused.
            t: unused.

        Returns:
            ``self``.
        """
        self._time_step(t)
        return self

    def transform_one(self, x: Any, t: float | None = None) -> dict[str, Any]:
        """Describe one frame as a set of local patch descriptors.

        Args:
            x: a Frame, a dict with a ``frame`` key, or a 2D array.
            t: unused, kept for the interface.

        Returns:
            A dict with keys ``"descriptors"`` and ``"positions"``. The
            former is a float array of shape ``(n_patches, n_bins + 2)``,
            the latter an integer array of shape ``(n_patches, 2)`` with
            the grid coordinates ``(row, col)`` of each patch. If the
            frame is missing or too small for one patch, both values are
            ``None`` and ``n_missing`` is incremented.
        """
        img = _as_array(x)
        if img is None or img.ndim not in (2, 3):
            self.n_missing_ += 1
            return {"descriptors": None, "positions": None}
        if not np.isfinite(img).all():
            self.n_missing_ += 1
            return {"descriptors": None, "positions": None}
        lum = _to_gray(img).astype(np.float64, copy=False)
        grid = self._patch_grid(lum.shape)
        if grid is None:
            self.n_missing_ += 1
            return {"descriptors": None, "positions": None}
        ny, nx = grid
        raw = self._describe_all(lum, ny, nx)
        agg = self._aggregate(raw)
        flat = agg.reshape(ny * nx, -1)
        positions = np.array(
            [(j, i) for j in range(ny) for i in range(nx)], dtype=np.intp
        )
        return {"descriptors": flat, "positions": positions}


class PatchMemory(AnomalyDetector):
    """Memory bank of normal patch descriptors with a bounded size.

    Args:
        max_size: maximum number of descriptors kept. Default 20000.
        seed: seed of the random generator used by the reservoir.
        mode: ``"reservoir"`` (default) or ``"coreset"``. In reservoir
            mode :meth:`learn_one` bounds the bank online with Algorithm
            R; in coreset mode :meth:`learn_one` keeps every patch and
            :meth:`finalize` reduces the bank to ``max_size`` with the
            greedy selection of PatchCore eq. 5.

    ``budget_s`` = 0.5 s: :meth:`score_one` measured 267.8 ms (p50) on
    the patches of a 320 x 240 frame against a 20000-row bank on a
    Colab CPU, about half the budget.

    Raises:
        ValueError: if ``max_size < 1`` or ``mode`` is not one of the
            two allowed values.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.vision.patches import PatchMemory
        >>> m = PatchMemory(max_size=10, seed=0)
        >>> bank = np.random.default_rng(0).normal(size=(5, 4))
        >>> pos = np.array([[0, i] for i in range(5)])
        >>> _ = m.learn_one({"descriptors": bank, "positions": pos})
        >>> round(float(m.score_one({"descriptors": bank, "positions": pos})), 6)
        0.0
    """

    budget_s = 0.5
    memory_class = "O(window)"

    _MODES = ("reservoir", "coreset")

    def __init__(
        self,
        max_size: int = 20000,
        seed: int | None = 0,
        mode: str = "reservoir",
    ) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        if mode not in self._MODES:
            raise ValueError(f"mode must be one of {self._MODES}, got {mode!r}")
        self.max_size = max_size
        self.seed = seed
        self.mode = mode
        self._bank: np.ndarray | None = None
        self._sq_bank: np.ndarray | None = None
        self._accumulator: list[np.ndarray] | None = None
        self._n_seen = 0
        self._finalized = False
        self._rng = np.random.default_rng(seed)
        self.last_patch_scores_: np.ndarray | None = None
        self.last_positions_: np.ndarray | None = None
        self.n_missing_ = 0
        self.n_reductions_ = 0

    def _reset(self) -> None:
        self._bank = None
        self._sq_bank = None
        self._accumulator = None
        self._n_seen = 0
        self._finalized = False
        self._rng = np.random.default_rng(self.seed)
        self.last_patch_scores_ = None
        self.last_positions_ = None
        self.n_missing_ = 0
        self.n_reductions_ = 0

    def _unit_test_skips(self) -> set:
        return set()

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    @property
    def bank_size(self) -> int:
        if self.mode == "coreset" and not self._finalized:
            return 0
        return 0 if self._bank is None else int(self._bank.shape[0])

    @property
    def n_seen(self) -> int:
        return self._n_seen

    @property
    def finalized(self) -> bool:
        return self._finalized

    def _unpack(self, x: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
        if isinstance(x, dict):
            desc = x.get("descriptors")
            pos = x.get("positions")
        elif isinstance(x, tuple) and len(x) == 2:
            desc, pos = x
        else:
            return None, None
        if desc is None:
            return None, None
        desc = np.asarray(desc, dtype=np.float64)
        if desc.ndim != 2 or desc.size == 0:
            return None, None
        if not np.isfinite(desc).all():
            return None, None
        if pos is None:
            pos = np.zeros((desc.shape[0], 2), dtype=np.intp)
        else:
            pos = np.asarray(pos, dtype=np.intp)
            if pos.shape[0] != desc.shape[0]:
                pos = np.zeros((desc.shape[0], 2), dtype=np.intp)
        return desc, pos

    def _check_dim(self, desc: np.ndarray) -> None:
        if self._bank is not None and self._bank.shape[1] != desc.shape[1]:
            raise ValueError(
                f"descriptor dimension changed from {self._bank.shape[1]} "
                f"to {desc.shape[1]}"
            )

    def _learn_reservoir(self, desc: np.ndarray) -> None:
        m = desc.shape[0]
        if self._bank is None:
            if m <= self.max_size:
                self._bank = desc.copy()
                self._n_seen = m
            else:
                self._bank = desc[: self.max_size].copy()
                self._n_seen = self.max_size
                for i in range(self.max_size, m):
                    self._n_seen += 1
                    j = int(self._rng.integers(0, self._n_seen))
                    if j < self.max_size:
                        self._bank[j] = desc[i]
            self._sq_bank = None
            return
        space = self.max_size - self._bank.shape[0]
        if space >= m:
            self._bank = np.vstack([self._bank, desc])
            self._n_seen += m
        else:
            if space > 0:
                self._bank = np.vstack([self._bank, desc[:space]])
                self._n_seen += space
                desc = desc[space:]
            for i in range(desc.shape[0]):
                self._n_seen += 1
                j = int(self._rng.integers(0, self._n_seen))
                if j < self.max_size:
                    self._bank[j] = desc[i]
        self._sq_bank = None

    def learn_one(self, x: Any, t: float | None = None) -> "PatchMemory":
        """Add the patches of one normal image to the bank.

        Args:
            x: the dict returned by :meth:`PatchFeatures.transform_one`,
                with keys ``"descriptors"`` and ``"positions"``.
            t: unused.

        Returns:
            ``self``.

        Raises:
            ValueError: if a new descriptor has a different dimension
                than the bank.
            RuntimeError: in ``mode="coreset"`` after :meth:`finalize`.
        """
        self._time_step(t)
        desc, _ = self._unpack(x)
        if desc is None:
            self.n_missing_ += 1
            return self
        if self._bank is not None:
            self._check_dim(desc)
        elif self._accumulator:
            first = self._accumulator[0]
            if first.shape[1] != desc.shape[1]:
                raise ValueError(
                    f"descriptor dimension changed from {first.shape[1]} "
                    f"to {desc.shape[1]}"
                )
        if self.mode == "reservoir":
            self._learn_reservoir(desc)
        else:
            if self._finalized:
                raise RuntimeError("learn_one after finalize in coreset mode")
            if self._accumulator is None:
                self._accumulator = []
            self._accumulator.append(desc.copy())
            self._n_seen += desc.shape[0]
        return self

    def finalize(self) -> "PatchMemory":
        """Reduce the accumulated bank to ``max_size`` (coreset mode only).

        Applies the greedy approximation of the minimax facility
        location coreset of PatchCore eq. 5 (Algorithm 1): at each step
        add the descriptor that maximises its distance to the closest
        already chosen one, until ``max_size`` are picked. The first
        descriptor is the one closest to the mean, so the result is
        deterministic. The descriptors are used as they are, without
        the random projection ``psi`` of Algorithm 1, since their
        dimension is small. Cost is ``O(n * max_size * d)``.

        In ``mode="reservoir"`` this is a no-op.

        Returns:
            ``self``.

        Raises:
            RuntimeError: if called twice.
        """
        if self.mode == "reservoir":
            return self
        if self._finalized:
            raise RuntimeError("finalize called twice")
        if not self._accumulator:
            self._bank = None
            self._finalized = True
            return self
        bank = np.vstack(self._accumulator)
        self._accumulator = None
        n = bank.shape[0]
        if n <= self.max_size:
            self._bank = bank
            self._sq_bank = None
            self._finalized = True
            return self
        sq = np.einsum("ij,ij->i", bank, bank)
        mean = bank.mean(axis=0)
        d0 = bank - mean
        i0 = int(np.argmin(np.einsum("ij,ij->i", d0, d0)))
        chosen = np.zeros(n, dtype=bool)
        chosen[i0] = True
        min_d2 = sq + sq[i0] - 2.0 * (bank @ bank[i0])
        min_d2[chosen] = -np.inf
        out = np.empty(self.max_size, dtype=np.intp)
        out[0] = i0
        for k in range(1, self.max_size):
            j = int(np.argmax(min_d2))
            out[k] = j
            chosen[j] = True
            d2 = sq + sq[j] - 2.0 * (bank @ bank[j])
            np.minimum(min_d2, d2, out=min_d2)
            min_d2[chosen] = -np.inf
        self._bank = bank[out]
        self._sq_bank = None
        self._finalized = True
        self.n_reductions_ += 1
        return self

    def _nn_distances(self, q: np.ndarray, chunk: int = 512) -> np.ndarray:
        bank = self._bank
        if bank is None:
            return np.zeros(q.shape[0], dtype=np.float64)
        if self._sq_bank is None:
            self._sq_bank = np.einsum("ij,ij->i", bank, bank)
        sq_bank = self._sq_bank
        if sq_bank is None:
            return np.zeros(q.shape[0], dtype=np.float64)
        out = np.empty(q.shape[0], dtype=np.float64)
        for i in range(0, q.shape[0], chunk):
            qi = q[i : i + chunk]
            sq_q = np.einsum("ij,ij->i", qi, qi)
            d2 = sq_q[:, None] + sq_bank[None, :] - 2.0 * (qi @ bank.T)
            out[i : i + chunk] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
        return out

    def score_one(self, x: Any, t: float | None = None) -> float:
        """Score one image by its most unusual patch.

        The image score is ``s*`` of PatchCore equation 6: the maximum
        over the patches of the distance to the nearest neighbour in
        the bank. The reweighting ``w`` the paper applies after that
        equation is not used here.

        Args:
            x: the dict returned by :meth:`PatchFeatures.transform_one`.
            t: unused.

        Returns:
            The maximum nearest-neighbour distance over the patches.
            ``0.0`` if the bank is empty or the input is invalid.

        Raises:
            RuntimeError: in ``mode="coreset"`` before :meth:`finalize`.
        """
        self._time_step(t)
        if self.mode == "coreset" and not self._finalized:
            raise RuntimeError("score_one before finalize in coreset mode")
        desc, pos = self._unpack(x)
        if desc is None or self._bank is None or self._bank.shape[0] == 0:
            self.last_patch_scores_ = None
            self.last_positions_ = None
            return 0.0
        scores = self._nn_distances(desc)
        self.last_patch_scores_ = scores
        self.last_positions_ = pos
        return float(scores.max())

    def is_outlier(self, x: Any, threshold: float = 1.0) -> bool:
        """Boolean wrapper around :meth:`score_one`."""
        return self.score_one(x) > threshold

    def patch_scores(self, x: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return the per-patch distances and their grid positions.

        Args:
            x: the dict returned by :meth:`PatchFeatures.transform_one`.

        Returns:
            ``(scores, positions)`` where ``scores`` is a float array of
            shape ``(n_patches,)`` and ``positions`` is the integer
            array of shape ``(n_patches, 2)`` returned by
            :meth:`PatchFeatures.transform_one`; two empty arrays if the
            bank is empty or the input is invalid.
        """
        _ = self.score_one(x)
        if self.last_patch_scores_ is None or self.last_positions_ is None:
            return np.zeros(0), np.zeros((0, 2), dtype=np.intp)
        return self.last_patch_scores_, self.last_positions_
