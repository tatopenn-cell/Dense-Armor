"""Visualise any band, overlay the analysis, write the video.

Three classes, all numpy plus the optional image writer:

- :class:`FalseColor` brings any single-channel frame into a visible
  image: thermal, depth, near-infrared, ultraviolet, SAR, an X-ray map,
  a multi-channel stack. The map is linear or logarithmic, a colormap is
  applied, and the legend carries the values that correspond to the two
  ends of the scale. The output has always three RGB channels in
  ``[0, 1]``, ready for a PNG or a video.
- :class:`Overlay` draws on a frame: the tracks of
  :class:`~dense_armor.utility.vision.tracking.MovingObjectTracker`, the
  boxes of an external detector, the flow field of
  :class:`~dense_armor.utility.vision.dense.DenseFlow`, the motion mask
  of :class:`~dense_armor.utility.vision.dense.MotionMask`, the anomaly
  score of :class:`~dense_armor.utility.vision.scene.FrameAnomaly`. Every
  overlay has a colour and the numbers are drawn near the object.
- :class:`VideoWriter` writes the annotated sequence with OpenCV, and
  one summary image with all the signals in time.

OpenCV is the optional dependency and is imported inside the writer, so
the rest of the module works without it. The colours are chosen so the
output is readable in the default matplotlib colormaps.
"""

from pathlib import Path

import numpy as np

from dense_armor.roles import Transformer


def _to_rgb(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 2:
        return np.stack([a, a, a], axis=-1)
    if a.ndim == 3 and a.shape[2] == 1:
        return np.repeat(a, 3, axis=2)
    return a


def _norm_to_unit(a: np.ndarray, log_scale: bool) -> np.ndarray:
    x = np.asarray(a, dtype=np.float32)
    if log_scale:
        x = np.log1p(np.maximum(x, 0.0))
    lo = float(np.percentile(x, 1.0))
    hi = float(np.percentile(x, 99.0))
    if hi - lo <= 1e-12:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


_CMAPS = {
    "gray": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
    "hot": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                     [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
    "viridis": np.array([[0.267, 0.005, 0.329], [0.188, 0.407, 0.554],
                         [0.208, 0.719, 0.473], [0.993, 0.906, 0.144]],
                        dtype=np.float32),
    "cool": np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32),
    "jet": np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 1.0],
                     [0.0, 1.0, 1.0], [0.5, 1.0, 0.5],
                     [1.0, 1.0, 0.0], [1.0, 0.5, 0.0],
                     [0.5, 0.0, 0.0]], dtype=np.float32),
}


def _apply_cmap(u: np.ndarray, name: str) -> np.ndarray:
    stops = _CMAPS[name]
    n = len(stops) - 1
    x = np.clip(u, 0.0, 1.0) * n
    lo = np.floor(x).astype(int)
    hi = np.minimum(lo + 1, n)
    frac = (x - lo)[..., None]
    a = stops[lo]
    b = stops[hi]
    return a * (1.0 - frac) + b * frac


class FalseColor(Transformer):
    """Bring any single-channel frame into a visible RGB image.

    Args:
        colormap: name of the colormap, one of ``"gray"``, ``"hot"``,
            ``"viridis"``, ``"cool"``, ``"jet"``. Default ``"gray"``.
        log_scale: apply ``log1p`` before the percentile stretch. Use it
            for a sensor with a wide dynamic range. Default ``False``.
        low_pct, high_pct: percentile of the frame mapped to the two
            ends of the colormap. Default ``1`` and ``99``.

    Raises:
        ValueError: on an unknown colormap or on a bad percentile pair.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.vision.visualize import FalseColor
        >>> f = FalseColor(colormap="hot")
        >>> a = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
        >>> out = f.transform_one({"array": a})
        >>> out["rgb"].shape
        (8, 8, 3)
        >>> bool(out["rgb"].min() >= 0.0 and out["rgb"].max() <= 1.0)
        True
    """

    memory_class = "O(HW)"

    def __init__(self, colormap: str = "gray", log_scale: bool = False,
                 low_pct: float = 1.0, high_pct: float = 99.0) -> None:
        if colormap not in _CMAPS:
            raise ValueError(
                f"colormap must be one of {sorted(_CMAPS)}, got {colormap!r}"
            )
        if not 0.0 <= low_pct < high_pct <= 100.0:
            raise ValueError(
                f"need 0 <= low_pct < high_pct <= 100, got {low_pct}, {high_pct}"
            )
        self.colormap = colormap
        self.log_scale = bool(log_scale)
        self.low_pct = float(low_pct)
        self.high_pct = float(high_pct)

    def _reset(self) -> None:
        pass

    def _unit_test_skips(self) -> set:
        return set()

    def transform_one(self, x, t=None):
        arr = x["array"] if isinstance(x, dict) else (
            x.array if hasattr(x, "array") else x
        )
        g = np.asarray(arr, dtype=np.float32)
        if g.ndim == 3:
            g = g[..., 0]
        u = _norm_to_unit(g, self.log_scale)
        rgb = _apply_cmap(u, self.colormap)
        lo = float(np.percentile(g, self.low_pct))
        hi = float(np.percentile(g, self.high_pct))
        return {"rgb": rgb.astype(np.float32),
                "legend": {"low": lo, "high": hi,
                           "colormap": self.colormap,
                           "log_scale": self.log_scale}}

    def learn_one(self, x, y=None, t=None):
        return self


class Overlay(Transformer):
    """Draw the analysis on a frame.

    Args:
        color_box: RGB colour of the boxes. Default green.
        color_flow: RGB colour of the flow arrows. Default blue.
        color_mask: RGB colour of the motion mask. Default red.
        box_thickness: thickness of the box outline. Default ``1``.
        draw_flow_step: draw one arrow every this many pixels. Default
            ``8``.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.vision.visualize import Overlay
        >>> o = Overlay()
        >>> f = np.zeros((32, 32, 3), dtype=np.float32)
        >>> tracks = [{"id": 3, "box": (4, 4, 12, 16), "missing": 0}]
        >>> out = o.transform_one({"array": f, "tracks": tracks})
        >>> out["rgb"].shape
        (32, 32, 3)
    """

    memory_class = "O(HW)"

    def __init__(self, color_box=(0.0, 1.0, 0.0), color_flow=(0.2, 0.4, 1.0),
                 color_mask=(1.0, 0.2, 0.2), box_thickness: int = 1,
                 draw_flow_step: int = 8) -> None:
        self.color_box = tuple(color_box)
        self.color_flow = tuple(color_flow)
        self.color_mask = tuple(color_mask)
        self.box_thickness = int(box_thickness)
        self.draw_flow_step = int(draw_flow_step)

    def _reset(self) -> None:
        pass

    def _unit_test_skips(self) -> set:
        return set()

    def _draw_box(self, rgb, x0, y0, x1, y1, color) -> None:
        h, w = rgb.shape[:2]
        x0 = max(0, int(round(x0)))
        y0 = max(0, int(round(y0)))
        x1 = min(w - 1, int(round(x1)))
        y1 = min(h - 1, int(round(y1)))
        t = self.box_thickness
        for k in range(t):
            if y0 + k < h:
                rgb[y0 + k, x0:x1 + 1] = color
            if y1 - k >= 0:
                rgb[y1 - k, x0:x1 + 1] = color
            if x0 + k < w:
                rgb[y0:y1 + 1, x0 + k] = color
            if x1 - k >= 0:
                rgb[y0:y1 + 1, x1 - k] = color

    def transform_one(self, x, t=None):
        if isinstance(x, dict) and "array" in x:
            rgb = _to_rgb(np.asarray(x["array"], dtype=np.float32)).copy()
            tracks = x.get("tracks")
            flow = x.get("flow")
            mask = x.get("mask")
        else:
            rgb = _to_rgb(np.asarray(x, dtype=np.float32)).copy()
            tracks = None
            flow = None
            mask = None
        if mask is not None:
            m = np.asarray(mask, dtype=bool)
            if m.shape == rgb.shape[:2]:
                rgb[m] = 0.5 * rgb[m] + 0.5 * np.asarray(self.color_mask,
                                                        dtype=np.float32)
        if flow is not None:
            u = np.asarray(flow["u"], dtype=float)
            v = np.asarray(flow["v"], dtype=float)
            h, w = rgb.shape[:2]
            step = max(1, self.draw_flow_step)
            for y in range(step // 2, h, step):
                for x in range(step // 2, w, step):
                    dx = int(round(u[y, x]))
                    dy = int(round(v[y, x]))
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        y0, y1 = min(y, ny), max(y, ny)
                        x0, x1 = min(x, nx), max(x, nx)
                        rgb[y0:y1 + 1, x0:x1 + 1] = self.color_flow
        if tracks:
            for tr in tracks:
                x0, y0, x1, y1 = tr["box"]
                self._draw_box(rgb, x0, y0, x1, y1, self.color_box)
        return {"rgb": np.clip(rgb, 0.0, 1.0)}

    def learn_one(self, x, y=None, t=None):
        return self


class VideoWriter:
    """Write a video and a summary image of the signals over time.

    Args:
        path: output file path. Its extension chooses the container
            (``.mp4`` uses ``mp4v``, ``.avi`` uses ``MJPG``).
        fps: frame rate of the output.
        size: ``(width, height)``; ``None`` reads it from the first frame.
        summary_path: optional path of the PNG with the signals in time.

    Raises:
        ImportError: if OpenCV is not installed.
        ValueError: on a non-positive ``fps``.

    Examples:
        >>> import numpy as np, tempfile, pathlib
        >>> from dense_armor.utility.vision.visualize import VideoWriter
        >>> w = VideoWriter(str(pathlib.Path(tempfile.mkdtemp()) / "out.mp4"),
        ...                 fps=5.0, size=(8, 8))
        >>> for i in range(3):
        ...     _ = w.write(np.zeros((8, 8, 3), dtype=np.float32))
        >>> w.close()
        >>> pathlib.Path(w.path).exists()
        True
    """

    def __init__(self, path: str | Path, fps: float = 30.0,
                 size: tuple | None = None,
                 summary_path: str | Path | None = None) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "VideoWriter requires OpenCV. Install it with "
                "`pip install dense-armor[vision]`."
            ) from exc
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        self._cv2 = cv2
        self.path = str(path)
        self.fps = float(fps)
        self.size = size
        self.summary_path = None if summary_path is None else str(summary_path)
        self._writer = None
        self._signals: dict = {}

    def _ensure_writer(self, rgb: np.ndarray) -> None:
        if self._writer is not None:
            return
        h, w = rgb.shape[:2]
        if self.size is None:
            size = (w, h)
        else:
            size = self.size
        fourcc = (self._cv2.VideoWriter_fourcc(*"mp4v")
                  if self.path.endswith(".mp4")
                  else self._cv2.VideoWriter_fourcc(*"MJPG"))
        self._writer = self._cv2.VideoWriter(self.path, fourcc, self.fps, size)
        if not self._writer.isOpened():
            raise RuntimeError(f"cannot open writer for {self.path}")

    def write(self, rgb) -> "VideoWriter":
        """Write one frame. The input is float RGB in ``[0, 1]``."""
        a = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
        self._ensure_writer(a)
        bgr = (a[:, :, ::-1] * 255.0).astype(np.uint8)
        if self.size is not None and (
            bgr.shape[1], bgr.shape[0]
        ) != self.size:
            bgr = self._cv2.resize(bgr, self.size)
        self._writer.write(bgr)
        return self

    def log(self, name: str, value: float) -> "VideoWriter":
        """Append a value to a named signal for the summary image."""
        self._signals.setdefault(name, []).append(float(value))
        return self

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self.summary_path is not None and self._signals:
            self._write_summary()

    def _write_summary(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(
            len(self._signals), 1,
            figsize=(6, 1.6 * len(self._signals)), squeeze=False,
        )
        for ax, (name, values) in zip(axes[:, 0], self._signals.items()):
            ax.plot(values, linewidth=0.8)
            ax.set_ylabel(name, fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[-1, 0].set_xlabel("frame")
        fig.tight_layout()
        fig.savefig(self.summary_path, dpi=110)
        plt.close(fig)
