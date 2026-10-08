"""Per-frame features: oriented gradients, moments, and short-range flow.

A single frame on its own carries little motion information, but three
families of descriptors are cheap, local, and enough to tell a moving
square from a still one:

- a histogram of oriented gradients (HOG) on a fixed grid of cells.
  The image is divided into ``cells`` rectangles; inside each cell the
  gradient orientation of every pixel votes into one of ``n_bins``
  angular bins, weighted by the gradient magnitude. The cell histogram
  is normalised to unit L2 norm, so a change in brightness does not
  change the descriptor. The construction is the one in Huang and
  Huang (2017), section 2.
- three global moments of the intensity: mean, standard deviation (the
  contrast), and the centroid of the intensity, which is where the
  "mass" of the image is.
- short-range optical flow against the previous frame, on a coarser
  grid: in every flow cell the classic Lucas and Kanade (1981) least
  squares is solved, as written in Ziani (2025), equations 4 and 5.

The first frame has no previous, so its flow block is zero. A frame
that is empty, has a wrong shape, or contains a non-finite value is
skipped and counted in ``n_missing``.
"""

from typing import Any

import numpy as np

from dense_armor.roles import Transformer


def _as_array(x: Any) -> np.ndarray | None:
    """Extract the image array from a Frame, a dict, or an array."""
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


class FrameFeatures(Transformer):
    """Turn one frame into a fixed-length dict of float features.

    Keys of the returned dict (all float):

    - ``hog_{cy}_{cx}_{b}``: the L2-normalised HOG histogram of cell
      ``(cy, cx)``, one value per angular bin ``b``. There are
      ``cells[0] * cells[1]`` cells and ``n_bins`` bins per cell.
    - ``mean``, ``contrast``: the mean and the standard deviation of the
      image intensity.
    - ``centroid_x``, ``centroid_y``: the coordinates of the intensity
      centroid, normalised in ``[0, 1]``.
    - ``flow_u_{fy}_{fx}``, ``flow_v_{fy}_{fx}``: the mean optical flow
      components inside the coarse flow cell ``(fy, fx)``.
    - ``flow_mag_{fy}_{fx}``: the magnitude of the same mean flow.

    Args:
        n_bins: number of angular bins in each cell. Default 9, the
            value used in Huang and Huang (2017).
        cells: ``(rows, cols)`` of the HOG grid. Default ``(3, 4)``.
        flow_cells: ``(rows, cols)`` of the flow grid. Default ``(2, 3)``.
        eps: numerical guard for the normalisation and the least squares.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.vision.features import FrameFeatures
        >>> ff = FrameFeatures()
        >>> frame = np.zeros((6, 8), dtype=np.float32)
        >>> out = ff.transform_one(frame)
        >>> out["mean"], out["contrast"]
        (0.0, 0.0)
    """

    budget_s = 0.02
    memory_class = "O(1)"

    def __init__(
        self,
        n_bins: int = 9,
        cells: tuple[int, int] = (3, 4),
        flow_cells: tuple[int, int] = (2, 3),
        eps: float = 1e-8,
    ) -> None:
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}")
        if cells[0] < 1 or cells[1] < 1:
            raise ValueError(f"cells must be positive, got {cells}")
        if flow_cells[0] < 1 or flow_cells[1] < 1:
            raise ValueError(f"flow_cells must be positive, got {flow_cells}")
        self.n_bins = n_bins
        self.cells = cells
        self.flow_cells = flow_cells
        self.eps = eps
        self._prev: np.ndarray | None = None
        self.n_missing_ = 0

    def _reset(self) -> None:
        self._prev = None
        self.n_missing_ = 0

    def _unit_test_skips(self) -> set:
        return set()

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    def _grid_slices(
        self, shape: tuple[int, int], grid: tuple[int, int]
    ) -> list[tuple[slice, slice]]:
        h, w = shape
        ny, nx = grid
        ys = np.linspace(0, h, ny + 1).astype(int)
        xs = np.linspace(0, w, nx + 1).astype(int)
        out: list[tuple[slice, slice]] = []
        for j in range(ny):
            for i in range(nx):
                out.append((slice(ys[j], ys[j + 1]), slice(xs[i], xs[i + 1])))
        return out

    def _gradients(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ix = np.zeros_like(img)
        iy = np.zeros_like(img)
        ix[:, 1:-1] = 0.5 * (img[:, 2:] - img[:, :-2])
        iy[1:-1, :] = 0.5 * (img[2:, :] - img[:-2, :])
        return ix, iy

    def _hog(self, img: np.ndarray) -> dict[str, float]:
        ix, iy = self._gradients(img)
        mag = np.sqrt(ix * ix + iy * iy)
        ang = np.mod(np.arctan2(iy, ix), 2.0 * np.pi)
        bin_index = np.floor(
            ang / (2.0 * np.pi / self.n_bins)
        ).astype(np.intp)
        bin_index = np.clip(bin_index, 0, self.n_bins - 1)
        out: dict[str, float] = {}
        for k, (sy, sx) in enumerate(self._grid_slices(img.shape, self.cells)):
            cy = k // self.cells[1]
            cx = k % self.cells[1]
            m = mag[sy, sx].ravel()
            b = bin_index[sy, sx].ravel()
            counts = np.bincount(b, weights=m, minlength=self.n_bins)
            norm = float(np.linalg.norm(counts)) + self.eps
            hist = counts / norm
            for j in range(self.n_bins):
                out[f"hog_{cy}_{cx}_{j}"] = float(hist[j])
        return out

    def _moments(self, img: np.ndarray) -> dict[str, float]:
        h, w = img.shape
        total = float(img.sum()) + self.eps
        ys, xs = np.indices(img.shape)
        cy = float((img * ys).sum() / total) / max(1, h - 1)
        cx = float((img * xs).sum() / total) / max(1, w - 1)
        return {
            "mean": float(img.mean()),
            "contrast": float(img.std()),
            "centroid_x": cx,
            "centroid_y": cy,
        }

    def _flow(
        self, cur: np.ndarray, prev: np.ndarray
    ) -> dict[str, float]:
        ix, iy = self._gradients(prev)
        it = cur - prev
        out: dict[str, float] = {}
        ny, nx = self.flow_cells
        for k, (sy, sx) in enumerate(self._grid_slices(cur.shape, (ny, nx))):
            fy = k // nx
            fx = k % nx
            a = np.column_stack(
                [ix[sy, sx].ravel(), iy[sy, sx].ravel()]
            )
            b = -it[sy, sx].ravel()
            try:
                w, *_ = np.linalg.lstsq(a, b, rcond=None)
                u, v = float(w[0]), float(w[1])
            except np.linalg.LinAlgError:
                u, v = 0.0, 0.0
            out[f"flow_u_{fy}_{fx}"] = u
            out[f"flow_v_{fy}_{fx}"] = v
            out[f"flow_mag_{fy}_{fx}"] = float(np.hypot(u, v))
        return out

    def _empty(self, shape: tuple[int, int]) -> dict[str, float]:
        out: dict[str, float] = {}
        for k in range(self.cells[0] * self.cells[1]):
            cy = k // self.cells[1]
            cx = k % self.cells[1]
            for j in range(self.n_bins):
                out[f"hog_{cy}_{cx}_{j}"] = 0.0
        out.update(
            {"mean": 0.0, "contrast": 0.0, "centroid_x": 0.0, "centroid_y": 0.0}
        )
        ny, nx = self.flow_cells
        for k in range(ny * nx):
            fy = k // nx
            fx = k % nx
            out[f"flow_u_{fy}_{fx}"] = 0.0
            out[f"flow_v_{fy}_{fx}"] = 0.0
            out[f"flow_mag_{fy}_{fx}"] = 0.0
        return out

    def learn_one(
        self, x: Any, y: Any = None, t: float | None = None
    ) -> "FrameFeatures":
        """Store the current frame so the next one can compute flow."""
        self._time_step(t)
        img = _as_array(x)
        if (
            img is None
            or img.ndim != 2
            or img.size == 0
            or not np.isfinite(img).all()
        ):
            self.n_missing_ += 1
            return self
        self._prev = img.copy()
        return self

    def transform_one(
        self, x: Any, t: float | None = None
    ) -> dict[str, float]:
        """Return the feature dict of one frame.

        Args:
            x: a Frame, a dict with a ``frame`` key, or a 2D array.
            t: unused, kept for the interface.

        Returns:
            A flat dict of float features.
        """
        img = _as_array(x)
        if (
            img is None
            or img.ndim != 2
            or img.size == 0
            or not np.isfinite(img).all()
        ):
            self.n_missing_ += 1
            return self._empty((self.cells[0], self.cells[1]))
        out: dict[str, float] = {}
        out.update(self._hog(img))
        out.update(self._moments(img))
        if self._prev is None or self._prev.shape != img.shape:
            out.update(self._flow(img, img))
        else:
            out.update(self._flow(img, self._prev))
        return out
