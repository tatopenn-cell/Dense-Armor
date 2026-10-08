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
  "mass" of the image is. With ``gray=False`` the same two moments are
  computed for each colour channel separately, and three extra pairs of
  keys appear.
- short-range optical flow against the previous frame, on a coarser
  grid: in every flow cell the classic Lucas and Kanade (1981) least
  squares is solved, as written in Ziani (2025), equations 4 and 5.
  With ``flow_levels > 1`` the flow is estimated coarse-to-fine on a
  pyramid (each level a 2x2 average of the one below; Ziani uses Gaussian
  pyramids) with bilinear interpolation between levels, the scheme of
  Ziani (2025), section 3.2, applied here to Lucas-Kanade.

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


def _to_gray(img: np.ndarray) -> np.ndarray:
    """Luminance of an HxWx3 image, identity for HxW."""
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] >= 3:
        return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    return img


def _downsample(img: np.ndarray) -> np.ndarray:
    """Half-resolution image by 2x2 box averaging.

    Args:
        img: 2D array.

    Returns:
        A new array with half the height and half the width (rounded
        down). If the image is too small to halve, the input is
        returned unchanged.
    """
    h, w = img.shape
    h2, w2 = h // 2, w // 2
    if h2 < 2 or w2 < 2:
        return img
    return img[: h2 * 2, : w2 * 2].reshape(h2, 2, w2, 2).mean(axis=(1, 3))


def _bilinear(img: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Bilinear sample of ``img`` at fractional coordinates.

    Args:
        img: 2D array.
        rows, cols: arrays of the same shape with fractional row and
            column coordinates. Out-of-range values are clamped to the
            image borders.

    Returns:
        An array of the same shape as ``rows`` with the interpolated
        values. This is the bilinear interpolation of Ziani (2025),
        equation 9, applied to a whole block of coordinates at once.
    """
    h, w = img.shape
    r = np.clip(rows, 0.0, h - 1.0)
    c = np.clip(cols, 0.0, w - 1.0)
    r0 = np.floor(r).astype(np.intp)
    c0 = np.floor(c).astype(np.intp)
    r1 = np.clip(r0 + 1, 0, h - 1)
    c1 = np.clip(c0 + 1, 0, w - 1)
    dr = r - r0
    dc = c - c0
    return (
        (1 - dr) * (1 - dc) * img[r0, c0]
        + (1 - dr) * dc * img[r0, c1]
        + dr * (1 - dc) * img[r1, c0]
        + dr * dc * img[r1, c1]
    )


class FrameFeatures(Transformer):
    """Turn one frame into a fixed-length dict of float features.

    Keys of the returned dict (all float):

    - ``hog_{cy}_{cx}_{b}``: the L2-normalised HOG histogram of cell
      ``(cy, cx)``, one value per angular bin ``b``. There are
      ``cells[0] * cells[1]`` cells and ``n_bins`` bins per cell.
    - ``mean``, ``contrast``: the mean and the standard deviation of
      the luminance.
    - ``centroid_x``, ``centroid_y``: the coordinates of the luminance
      centroid, normalised in ``[0, 1]``.
    - ``flow_u_{fy}_{fx}``, ``flow_v_{fy}_{fx}``: the mean optical flow
      components inside the coarse flow cell ``(fy, fx)``.
    - ``flow_mag_{fy}_{fx}``: the magnitude of the same mean flow.
    - if ``gray=False``, six extra keys ``mean_R``, ``mean_G``,
      ``mean_B``, ``contrast_R``, ``contrast_G``, ``contrast_B`` for
      the per-channel moments.

    Args:
        n_bins: number of angular bins in each cell. Default 9, the
            value used in Huang and Huang (2017).
        cells: ``(rows, cols)`` of the HOG grid. Default ``(3, 4)``.
        flow_cells: ``(rows, cols)`` of the flow grid. Default ``(2, 3)``.
        flow_levels: number of pyramid levels for the optical flow.
            Default ``1``: a single level, the same behaviour as
            without a pyramid. ``n > 1`` builds a pyramid of 2x2 averages
            with ``n`` levels, solves Lucas-Kanade at the coarsest
            level and refines the flow down to the finest level with
            bilinear interpolation, the coarse-to-fine scheme of
            Ziani (2025), section 3.2.
        gray: ``True`` (default) takes 2D images and computes the
            moments on the luminance. ``False`` takes ``(H, W, 3)``
            images, adds the per-channel moments, and keeps the HOG
            and the flow on the luminance, so the other keys do not
            change.
        eps: numerical guard for the normalisation and the least
            squares.

    Raises:
        ValueError: if ``n_bins < 1``, if a grid size is not positive,
            or if ``flow_levels < 1``.

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
        flow_levels: int = 1,
        gray: bool = True,
        eps: float = 1e-8,
    ) -> None:
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}")
        if cells[0] < 1 or cells[1] < 1:
            raise ValueError(f"cells must be positive, got {cells}")
        if flow_cells[0] < 1 or flow_cells[1] < 1:
            raise ValueError(f"flow_cells must be positive, got {flow_cells}")
        if flow_levels < 1:
            raise ValueError(f"flow_levels must be >= 1, got {flow_levels}")
        self.n_bins = n_bins
        self.cells = cells
        self.flow_cells = flow_cells
        self.flow_levels = flow_levels
        self.gray = gray
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
        bin_index = np.floor(ang / (2.0 * np.pi / self.n_bins)).astype(np.intp)
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

    def _color_moments(self, img: np.ndarray) -> dict[str, float]:
        out: dict[str, float] = {}
        names = ("R", "G", "B")
        for j, name in enumerate(names):
            ch = img[..., j]
            out[f"mean_{name}"] = float(ch.mean())
            out[f"contrast_{name}"] = float(ch.std())
        return out

    def _warp_grid(
        self,
        prev: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
    ) -> np.ndarray:
        """Warp the whole image by a per-cell flow field.

        The flow arrays are broadcast to every pixel of their cell,
        then the image is sampled bilinearly at ``(r - v, c - u)``.
        """
        h, w = prev.shape
        ny, nx = self.flow_cells
        u_pp = np.zeros((h, w))
        v_pp = np.zeros((h, w))
        for k, (sy, sx) in enumerate(self._grid_slices(prev.shape, (ny, nx))):
            fy = k // nx
            fx = k % nx
            u_pp[sy, sx] = float(u[fy, fx])
            v_pp[sy, sx] = float(v[fy, fx])
        rr = np.arange(h, dtype=np.float64)[:, None] - v_pp
        cc = np.arange(w, dtype=np.float64)[None, :] - u_pp
        return _bilinear(prev, rr, cc)

    def _flow_cells_one_level(
        self,
        cur: np.ndarray,
        prev: np.ndarray,
        u0: np.ndarray,
        v0: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        ny, nx = self.flow_cells
        if not (u0.any() or v0.any()):
            warped = prev
        else:
            warped = self._warp_grid(prev, u0, v0)
        wix, wiy = self._gradients(warped)
        u_new = np.zeros((ny, nx))
        v_new = np.zeros((ny, nx))
        for k, (sy, sx) in enumerate(self._grid_slices(cur.shape, (ny, nx))):
            fy = k // nx
            fx = k % nx
            it = cur[sy, sx] - warped[sy, sx]
            a = np.column_stack([wix[sy, sx].ravel(), wiy[sy, sx].ravel()])
            b = -it.ravel()
            try:
                w, *_ = np.linalg.lstsq(a, b, rcond=None)
                du, dv = float(w[0]), float(w[1])
            except np.linalg.LinAlgError:
                du, dv = 0.0, 0.0
            u_new[fy, fx] = float(u0[fy, fx]) + du
            v_new[fy, fx] = float(v0[fy, fx]) + dv
        return u_new, v_new

    def _flow(self, cur: np.ndarray, prev: np.ndarray) -> dict[str, float]:
        ny, nx = self.flow_cells
        if self.flow_levels == 1:
            u0 = np.zeros((ny, nx))
            v0 = np.zeros((ny, nx))
            u, v = self._flow_cells_one_level(cur, prev, u0, v0)
        else:
            cur_pyr: list[np.ndarray] = [cur]
            prev_pyr: list[np.ndarray] = [prev]
            for _ in range(self.flow_levels - 1):
                cur_pyr.append(_downsample(cur_pyr[-1]))
                prev_pyr.append(_downsample(prev_pyr[-1]))
            u = np.zeros((ny, nx))
            v = np.zeros((ny, nx))
            for lvl in reversed(range(len(cur_pyr))):
                cur_l = cur_pyr[lvl]
                prev_l = prev_pyr[lvl]
                if lvl == len(cur_pyr) - 1:
                    u_l = np.zeros((ny, nx))
                    v_l = np.zeros((ny, nx))
                else:
                    u_l = u * 2.0
                    v_l = v * 2.0
                u, v = self._flow_cells_one_level(cur_l, prev_l, u_l, v_l)
        out: dict[str, float] = {}
        for fy in range(ny):
            for fx in range(nx):
                uu = float(u[fy, fx])
                vv = float(v[fy, fx])
                out[f"flow_u_{fy}_{fx}"] = uu
                out[f"flow_v_{fy}_{fx}"] = vv
                out[f"flow_mag_{fy}_{fx}"] = float(np.hypot(uu, vv))
        return out

    def _empty(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for k in range(self.cells[0] * self.cells[1]):
            cy = k // self.cells[1]
            cx = k % self.cells[1]
            for j in range(self.n_bins):
                out[f"hog_{cy}_{cx}_{j}"] = 0.0
        out.update({"mean": 0.0, "contrast": 0.0, "centroid_x": 0.0, "centroid_y": 0.0})
        if not self.gray:
            for name in ("R", "G", "B"):
                out[f"mean_{name}"] = 0.0
                out[f"contrast_{name}"] = 0.0
        ny, nx = self.flow_cells
        for k in range(ny * nx):
            fy = k // nx
            fx = k % nx
            out[f"flow_u_{fy}_{fx}"] = 0.0
            out[f"flow_v_{fy}_{fx}"] = 0.0
            out[f"flow_mag_{fy}_{fx}"] = 0.0
        return out

    def _valid(self, img: np.ndarray | None) -> bool:
        if img is None:
            return False
        if img.ndim == 2:
            return img.size > 0 and bool(np.isfinite(img).all())
        if img.ndim == 3 and img.shape[2] >= 3:
            return img.size > 0 and bool(np.isfinite(img).all())
        return False

    def learn_one(
        self, x: Any, y: Any = None, t: float | None = None
    ) -> "FrameFeatures":
        """Store the current frame so the next one can compute flow."""
        self._time_step(t)
        img = _as_array(x)
        if not self._valid(img):
            self.n_missing_ += 1
            return self
        assert img is not None
        self._prev = _to_gray(img).copy()
        return self

    def transform_one(self, x: Any, t: float | None = None) -> dict[str, float]:
        """Return the feature dict of one frame.

        Args:
            x: a Frame, a dict with a ``frame`` key, or a 2D array
                (or 3D with ``gray=False``).
            t: unused, kept for the interface.

        Returns:
            A flat dict of float features.
        """
        img = _as_array(x)
        if not self._valid(img):
            self.n_missing_ += 1
            return self._empty()
        assert img is not None
        lum = _to_gray(img)
        out: dict[str, float] = {}
        out.update(self._hog(lum))
        out.update(self._moments(lum))
        if not self.gray:
            out.update(self._color_moments(img if img.ndim == 3 else np.stack([lum] * 3, axis=-1)))
        if self._prev is None or self._prev.shape != lum.shape:
            out.update(self._flow(lum, lum))
        else:
            out.update(self._flow(lum, self._prev))
        return out
