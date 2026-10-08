"""Frame streams for the vision module.

A stream is an iterator of :class:`Frame` objects, one image per sample.
A frame carries a numpy image (float in ``[0, 1]`` when grayscale) and a
timestamp ``t`` in seconds. Three streams cover the three sources a
robot or a developer has:

- :class:`CameraStream` reads from a camera through OpenCV. OpenCV is
  an optional dependency (extra ``[vision]``) and imported lazily, so
  the rest of the vision pipeline works without it.
- :class:`ImageFolderStream` reads a folder of image files in sorted
  order (Pillow).
- :class:`ArrayStream` takes an array of frames already in memory, used
  by tests and by anyone with the frames at hand.

Every stream yields the same object, :class:`Frame`, so the rest of the
pipeline is source-agnostic.
"""

import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Frame:
    """One image frame with its timestamp.

    Attributes:
        array: H x W float array in ``[0, 1]`` for grayscale, or H x W x 3
            for color.
        t: timestamp in seconds.
    """

    array: np.ndarray
    t: float

    def to_dict(self) -> dict:
        """Return a dict with the image and the timestamp.

        Returns:
            A dict ``{"frame": array, "t": t}``.
        """
        return {"frame": self.array, "t": self.t}


def _to_float01(arr: np.ndarray, gray: bool, bgr: bool = False) -> np.ndarray:
    a = np.asarray(arr)
    integer = a.dtype.kind in "ui"
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[..., :3]
    if a.ndim == 3 and a.shape[2] == 3 and bgr:
        a = a[..., ::-1]
    if a.ndim == 3 and a.shape[2] == 3 and gray:
        a = a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    a = a.astype(np.float32, copy=False)
    if integer:
        a = a / 255.0
    return a


def _resize_nearest(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    if arr.shape[0] == h and arr.shape[1] == w:
        return arr
    yi = (np.arange(h) * arr.shape[0] / h).astype(np.intp)
    xi = (np.arange(w) * arr.shape[1] / w).astype(np.intp)
    return arr[yi][:, xi]


class ArrayStream:
    """Iterate over a numpy array of frames.

    Args:
        frames: array of shape ``(n, H, W)`` or ``(n, H, W, C)``.
        fps: frame rate used to set ``t = i / fps``, in Hz. Must be
            positive.
        size: ``(width, height)`` to resize each frame to, or ``None``
            to keep the original size.
        gray: convert color frames to grayscale.

    Raises:
        ValueError: if ``fps <= 0``.

    Examples:
        >>> import numpy as np
        >>> frames = np.zeros((3, 4, 4), dtype=np.float32)
        >>> s = ArrayStream(frames, fps=10.0)
        >>> len(s)
        3
        >>> f = next(iter(s))
        >>> f.array.shape, round(f.t, 3)
        ((4, 4), 0.0)
    """

    def __init__(
        self,
        frames: np.ndarray,
        fps: float = 30.0,
        size: tuple[int, int] | None = None,
        gray: bool = True,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        self.frames = np.asarray(frames)
        self.fps = float(fps)
        self.size = size
        self.gray = gray

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[Frame]:
        for i, raw in enumerate(self.frames):
            arr = _to_float01(raw, self.gray)
            if self.size is not None:
                arr = _resize_nearest(arr, self.size)
            yield Frame(array=arr, t=i / self.fps)


class ImageFolderStream:
    """Iterate over image files in a folder.

    Args:
        path: folder path, or a sequence of file paths.
        fps: if given, ``t = i / fps``; otherwise ``t = float(i)``.
        size: ``(width, height)`` to resize each frame to, or ``None``.
        gray: convert to grayscale.

    Raises:
        ValueError: if the folder contains no image file.

    Examples:
        >>> import tempfile, pathlib
        >>> import numpy as np
        >>> from PIL import Image
        >>> d = tempfile.mkdtemp()
        >>> for i in range(2):
        ...     _ = Image.fromarray(
        ...         (np.zeros((4, 4), dtype=np.uint8) + i)
        ...     ).save(pathlib.Path(d) / f"{i:03d}.png")
        >>> s = ImageFolderStream(d, fps=2.0)
        >>> len(s)
        2
        >>> round(next(iter(s)).t, 3)
        0.0
    """

    _SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    def __init__(
        self,
        path: str | Path | Sequence[str | Path],
        fps: float | None = None,
        size: tuple[int, int] | None = None,
        gray: bool = True,
    ) -> None:
        if isinstance(path, (str, Path)):
            files = sorted(
                q for q in Path(path).iterdir() if q.suffix.lower() in self._SUFFIXES
            )
        else:
            files = [Path(q) for q in path]
        if not files:
            raise ValueError("ImageFolderStream: no image files found")
        self.files = files
        self.fps = fps
        self.size = size
        self.gray = gray

    def __len__(self) -> int:
        return len(self.files)

    def __iter__(self) -> Iterator[Frame]:
        from PIL import Image

        for i, p in enumerate(self.files):
            arr = _to_float01(np.asarray(Image.open(p)), self.gray)
            if self.size is not None:
                arr = _resize_nearest(arr, self.size)
            t = i / self.fps if self.fps is not None else float(i)
            yield Frame(array=arr, t=t)


class CameraStream:
    """Iterate over camera frames through OpenCV.

    OpenCV is imported lazily inside ``__init__``. If it is not
    installed, the constructor raises :class:`ImportError` with a clear
    message. OpenCV delivers color frames in BGR order; they are turned
    to RGB before the grayscale conversion.

    Args:
        device: camera index (default ``0``).
        fps: requested frame rate, in Hz, passed to the camera. Must be
            positive.
        size: ``(width, height)`` to resize each frame to, or ``None``.
        gray: convert to grayscale.
        max_frames: stop after this many frames (``None`` = infinite).

    Raises:
        ImportError: if OpenCV is not installed.
        ValueError: if ``fps <= 0``.
    """

    def __init__(
        self,
        device: int = 0,
        fps: float = 30.0,
        size: tuple[int, int] | None = (160, 120),
        gray: bool = True,
        max_frames: int | None = None,
    ) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "CameraStream requires OpenCV. Install it with "
                "`pip install dense-armor[vision]`."
            ) from exc
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        self._cv2 = cv2
        self.device = device
        self.fps = float(fps)
        self.size = size
        self.gray = gray
        self.max_frames = max_frames

    def __iter__(self) -> Iterator[Frame]:
        cap = self._cv2.VideoCapture(self.device)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open camera {self.device}")
        cap.set(self._cv2.CAP_PROP_FPS, self.fps)
        try:
            t0 = time.monotonic()
            i = 0
            while self.max_frames is None or i < self.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                arr = _to_float01(frame, self.gray, bgr=True)
                if self.size is not None:
                    arr = _resize_nearest(arr, self.size)
                yield Frame(array=arr, t=time.monotonic() - t0)
                i += 1
        finally:
            cap.release()
