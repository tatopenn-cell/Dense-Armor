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
            for color; or H x W x C for a channel count different from 3.
        t: timestamp in seconds.
        dropped: number of frames lost between the previous yielded frame
            and this one. ``0`` means no loss. Only the streaming sources
            that can lose frames (a network camera, a reader slower than
            the source) set this; in-memory and file sources leave it at
            ``0``.
    """

    array: np.ndarray
    t: float
    dropped: int = 0

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


_INT_MAX = {"uint8": 255.0, "uint16": 65535.0}


def _to_dtype(arr: np.ndarray, dtype: str, gray: bool, bgr: bool = False) -> np.ndarray:
    """Convert a raw image to the requested dtype.

    ``float32`` scales integer inputs to ``[0, 1]`` and leaves floats
    untouched; ``uint8`` and ``uint16`` rescale an integer input of a
    different integer dtype to the target range, and scale a float input
    in ``[0, 1]`` to the target integer range. The 16-bit path loses no
    value on a 16-bit sensor readout.
    """
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[..., :3]
    if a.ndim == 3 and a.shape[2] == 3 and bgr:
        a = a[..., ::-1]
    if a.ndim == 3 and a.shape[2] == 3 and gray:
        a = a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    if dtype == "float32":
        integer = a.dtype.kind in "ui"
        if integer:
            f = a.astype(np.float32, copy=False)
            if a.dtype == np.uint16:
                return f / 65535.0
            return f / 255.0
        return a.astype(np.float32, copy=False)
    top = _INT_MAX[dtype]
    if a.dtype.kind in "ui":
        if dtype == "uint8":
            if a.dtype == np.uint8:
                return a.astype(np.uint8, copy=False)
            f = a.astype(np.float64) * (255.0 / float(np.iinfo(a.dtype).max))
            return np.clip(f, 0.0, 255.0).astype(np.uint8)
        if a.dtype == np.uint16:
            return a.astype(np.uint16, copy=False)
        f = a.astype(np.float64) * (65535.0 / float(np.iinfo(a.dtype).max))
        return np.clip(f, 0.0, 65535.0).astype(np.uint16)
    f = np.clip(a.astype(np.float64), 0.0, 1.0) * top
    return np.round(f).astype(np.dtype(dtype))


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
        dtype: target dtype of the output array. ``"float32"`` (default)
            scales integer inputs to ``[0, 1]`` and leaves floats
            untouched; ``"uint8"`` keeps the 8-bit scale; ``"uint16"``
            keeps the 16-bit scale (a 16-bit sensor readout loses no
            value). When ``dtype`` is ``"uint8"`` or ``"uint16"`` the
            output array carries that dtype and ``gray`` conversion is
            done on the integer values.

    Raises:
        ValueError: if ``fps <= 0`` or ``dtype`` is not supported.

    Examples:
        >>> import numpy as np
        >>> frames = np.zeros((3, 4, 4), dtype=np.float32)
        >>> s = ArrayStream(frames, fps=10.0)
        >>> len(s)
        3
        >>> f = next(iter(s))
        >>> f.array.shape, round(f.t, 3)
        ((4, 4), 0.0)
        >>> frames16 = np.array([[[0, 1000], [60000, 65535]]], dtype=np.uint16)
        >>> s16 = ArrayStream(frames16, dtype="uint16")
        >>> next(iter(s16)).array.tolist()
        [[0, 1000], [60000, 65535]]
    """

    _DTYPES = ("float32", "uint8", "uint16")

    def __init__(
        self,
        frames: np.ndarray,
        fps: float = 30.0,
        size: tuple[int, int] | None = None,
        gray: bool = True,
        dtype: str = "float32",
    ) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        if dtype not in self._DTYPES:
            raise ValueError(
                f"dtype must be one of {self._DTYPES}, got {dtype!r}"
            )
        self.frames = np.asarray(frames)
        self.fps = float(fps)
        self.size = size
        self.gray = gray
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[Frame]:
        for i, raw in enumerate(self.frames):
            arr = _to_dtype(raw, self.dtype, self.gray)
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

class VideoStream:
    """Common interface of the frame sources.

    Every video source in the library is an iterator of :class:`Frame`
    and exposes the same knobs: an optional resize, an optional grayscale
    conversion, a target dtype, a ``start`` and ``stop`` index, a
    ``stride`` to read one frame every ``k``. The base class does not
    read anything; it declares the interface that :class:`ArrayStream`,
    :class:`ImageFolderStream`, :class:`CameraStream`,
    :class:`VideoFileStream` and :class:`CubeStream` implement.

    Args:
        fps: frame rate used when the source has no timestamp of its
            own. Must be positive.
        size: ``(width, height)`` to resize each frame to, or ``None``.
        gray: convert color frames to grayscale.
        dtype: target dtype of the output array; see :class:`ArrayStream`.
        start: first frame index to yield.
        stop: last frame index to yield, exclusive, or ``None``.
        stride: yield one frame every ``stride``, starting at ``start``.
            Must be ``>= 1``.

    Raises:
        ValueError: on a bad ``fps``, ``start``, ``stop`` or ``stride``.
    """

    def __init__(
        self,
        fps: float = 30.0,
        size: tuple[int, int] | None = None,
        gray: bool = True,
        dtype: str = "float32",
        start: int = 0,
        stop: int | None = None,
        stride: int = 1,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if start < 0:
            raise ValueError(f"start must be >= 0, got {start}")
        if stop is not None and stop < start:
            raise ValueError(
                f"stop ({stop}) must be >= start ({start})"
            )
        self.fps = float(fps)
        self.size = size
        self.gray = gray
        self.dtype = dtype
        self.start = int(start)
        self.stop = stop
        self.stride = int(stride)

    def _in_range(self, index: int) -> bool:
        if index < self.start:
            return False
        if self.stop is not None and index >= self.stop:
            return False
        return (index - self.start) % self.stride == 0

    def __iter__(self):
        raise NotImplementedError(
            "VideoStream is an interface; use a concrete subclass"
        )


class VideoFileStream(VideoStream):
    """Read a video from a file, an URL, a glob of files or a network stream.

    OpenCV is imported lazily inside ``__init__``, so a session without it
    can still import the module and use the numpy-only streams. The
    timestamp of each frame is the container's own clock, read from the
    decoder in milliseconds and converted to seconds; it is not
    ``frame_index / fps``, so a video whose frame rate varies with time
    keeps the true time of every frame. A frame that the reader skips
    because the analysis is slower than the source is counted in
    :attr:`Frame.dropped` of the next yielded frame.

    Args:
        source: a file path, an URL (``file://``, ``http://``,
            ``rtsp://``), a glob (``shots/*.mp4``) or a sequence of
            paths. When a glob or a sequence matches more than one file,
            the frames of the files are concatenated in order and the
            time of each frame is the container time of the file it
            comes from plus the offset accumulated from the previous
            files.
        fps: fallback rate used only if the container does not report
            its own frame rate; it is never used to set ``t``.
        size: ``(width, height)`` to resize each frame to, or ``None``.
        gray: convert color frames to grayscale.
        dtype: target dtype; see :class:`ArrayStream`.
        start, stop, stride: as in :class:`VideoStream`.
        max_frames: stop after this many yielded frames (``None`` = all).
        rotate: ``"auto"`` (default) reads the rotation from the
            container metadata when present; ``False`` ignores it; a
            number in ``{0, 90, 180, 270}`` forces that rotation in
            degrees counterclockwise.
        follow: if ``True``, on a source that keeps growing (a file being
            written by another process, a live stream) the reader waits
            and retries instead of stopping at the first read failure;
            a ``False`` value stops at the first EOF.
        wait_s: seconds to sleep between retries when ``follow`` is
            ``True``.
        max_waits: maximum number of consecutive retries before giving
            up (``None`` = forever).

    Raises:
        ImportError: if OpenCV is not installed.
        FileNotFoundError: if ``source`` is a path that does not exist.
        ValueError: on bad parameters.

    Examples:
        >>> import tempfile, pathlib, math
        >>> import numpy as np
        >>> import cv2
        >>> d = pathlib.Path(tempfile.mkdtemp()) / "clip.mp4"
        >>> w = cv2.VideoWriter(
        ...     str(d), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (8, 8)
        ... )
        >>> for i in range(5):
        ...     frame = np.full((8, 8, 3), i * 50, dtype=np.uint8)
        ...     _ = w.write(frame)
        >>> w.release()
        >>> s = VideoFileStream(str(d))
        >>> frames = list(s)
        >>> len(frames)
        5
        >>> round(frames[0].t, 3)
        0.0
        >>> round(frames[-1].t, 3)
        0.4
    """

    def __init__(
        self,
        source,
        fps: float = 30.0,
        size: tuple[int, int] | None = None,
        gray: bool = True,
        dtype: str = "float32",
        start: int = 0,
        stop: int | None = None,
        stride: int = 1,
        max_frames: int | None = None,
        rotate="auto",
        follow: bool = False,
        wait_s: float = 0.05,
        max_waits: int | None = 200,
    ) -> None:
        super().__init__(fps, size, gray, dtype, start, stop, stride)
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "VideoFileStream requires OpenCV. Install it with "
                "`pip install dense-armor[vision]`."
            ) from exc
        self._cv2 = cv2
        self.source = source
        self.max_frames = max_frames
        self.rotate = rotate
        self.follow = follow
        self.wait_s = float(wait_s)
        self.max_waits = max_waits
        self._paths = self._resolve(source)

    @staticmethod
    def _resolve(source):
        if isinstance(source, (list, tuple)):
            return [str(q) for q in source]
        if isinstance(source, Path):
            return [str(source)]
        if isinstance(source, str):
            if any(s in source for s in ("://",)):
                return [source]
            import glob as _glob
            if any(ch in source for ch in "*?["):
                hits = sorted(_glob.glob(source))
                if not hits:
                    raise FileNotFoundError(
                        f"VideoFileStream: glob matches no file: {source}"
                    )
                return hits
            p = Path(source)
            if not p.exists():
                raise FileNotFoundError(
                    f"VideoFileStream: file not found: {source}"
                )
            return [str(p)]
        raise TypeError(
            f"source must be a path, an URL, a glob or a sequence, "
            f"got {type(source).__name__}"
        )

    def _open_one(self, path):
        cap = self._cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(
                f"VideoFileStream: cannot open {path!r}"
            )
        return cap

    def _apply_rotation(self, frame):
        if self.rotate is False or self.rotate == 0:
            return frame
        k = None
        if self.rotate == "auto":
            meta = float(
                self._last_cap.get(self._cv2.CAP_PROP_ORIENTATION_META)
                or 0.0
            )
            if meta in (90.0, 180.0, 270.0):
                k = int(meta) // 90
        elif isinstance(self.rotate, (int, float)):
            k = int(self.rotate) // 90 % 4
        if not k:
            return frame
        return np.rot90(frame, k=k)

    def __iter__(self):
        yielded = 0
        global_index = 0
        t_offset = 0.0
        last_t = 0.0
        for path in self._paths:
            cap = self._open_one(path)
            self._last_cap = cap
            try:
                container_fps = float(cap.get(self._cv2.CAP_PROP_FPS) or 0.0)
                if container_fps <= 0.0:
                    container_fps = self.fps
                waits = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        if self.follow and (
                            self.max_waits is None or waits < self.max_waits
                        ):
                            time.sleep(self.wait_s)
                            waits += 1
                            continue
                        break
                    waits = 0
                    t_ms = float(cap.get(self._cv2.CAP_PROP_POS_MSEC) or 0.0)
                    t_here = t_offset + t_ms / 1000.0
                    if t_here < last_t:
                        t_here = last_t
                    last_t = t_here
                    if self._in_range(global_index):
                        arr = self._apply_rotation(frame)
                        a = _to_dtype(arr, self.dtype, self.gray, bgr=True)
                        if self.size is not None:
                            a = _resize_nearest(a, self.size)
                        yield Frame(array=a, t=t_here, dropped=0)
                        yielded += 1
                        if self.max_frames is not None and yielded >= self.max_frames:
                            return
                    global_index += 1
            finally:
                cap.release()
            t_offset = last_t + 1.0 / container_fps


class CubeStream(VideoStream):
    """Iterate over a multi-channel cube ``(T, H, W, C)``.

    A cube is the format of a thermal sequence (one channel), a
    multispectral or hyperspectral stack (many channels), a depth map
    (one channel of floats) and, in general, of any video whose channel
    count is not three. The time of the frame ``i`` is ``i / fps``; the
    cube carries no clock of its own.

    Args:
        cube: array of shape ``(T, H, W)`` or ``(T, H, W, C)``.
        fps: frame rate; sets ``t = i / fps``.
        size: ``(width, height)`` to resize each frame to, or ``None``.
        gray: reduce a 3-channel frame to grayscale; a cube with a
            channel count different from 3 is left untouched.
        dtype: target dtype; ``"uint16"`` keeps a 16-bit thermal readout
            without loss.
        start, stop, stride, max_frames: as in :class:`VideoStream`.

    Raises:
        ValueError: on a cube whose first dimension is not the time, or
            on bad parameters.

    Examples:
        >>> import numpy as np
        >>> cube = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
        >>> s = CubeStream(cube, fps=2.0, dtype="uint16", gray=False)
        >>> frames = list(s)
        >>> len(frames)
        2
        >>> frames[0].array.dtype
        dtype('uint16')
        >>> round(frames[1].t, 3)
        0.5
    """

    def __init__(
        self,
        cube: np.ndarray,
        fps: float = 30.0,
        size: tuple[int, int] | None = None,
        gray: bool = False,
        dtype: str = "float32",
        start: int = 0,
        stop: int | None = None,
        stride: int = 1,
        max_frames: int | None = None,
    ) -> None:
        super().__init__(fps, size, gray, dtype, start, stop, stride)
        a = np.asarray(cube)
        if a.ndim not in (3, 4):
            raise ValueError(
                f"cube must have shape (T, H, W) or (T, H, W, C), "
                f"got {a.shape}"
            )
        self.cube = a
        self.max_frames = max_frames

    def __len__(self) -> int:
        return len(self.cube)

    def __iter__(self):
        yielded = 0
        for i, raw in enumerate(self.cube):
            if not self._in_range(i):
                continue
            arr = _to_dtype(raw, self.dtype, self.gray)
            if self.size is not None:
                arr = _resize_nearest(arr, self.size)
            yield Frame(array=arr, t=i / self.fps, dropped=0)
            yielded += 1
            if self.max_frames is not None and yielded >= self.max_frames:
                return

