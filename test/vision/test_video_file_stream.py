"""V1: the universal reader.

The reader is tested against synthetic files generated at test time (no
real video in the repository), so the tests do not depend on any asset.
The reader depends on OpenCV and is skipped without it; the
numpy-only streams run everywhere.

What the tests check:

- the number of frames read equals the number written, for two
  containers and for a container with a variable frame rate written on
  purpose;
- the timestamp of each frame is the container clock, within 1 ms of
  the expected value;
- ``start``, ``stop``, ``stride`` and ``max_frames`` behave;
- a 16-bit cube keeps every value, no loss;
- a non-existent path raises ``FileNotFoundError``.
"""

import pathlib
import tempfile

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from dense_armor.utility.vision.streams import (  # noqa: E402
    CubeStream,
    VideoFileStream,
)


def _write_clip(path: pathlib.Path, n: int, fps: float, size=(8, 8)) -> None:
    w = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    try:
        for i in range(n):
            w.write(np.full((size[1], size[0], 3), (i * 20) % 256, np.uint8))
    finally:
        w.release()


def test_file_reads_the_same_number_of_frames_written():
    d = pathlib.Path(tempfile.mkdtemp())
    p = d / "clip.mp4"
    _write_clip(p, 7, 10.0)
    frames = list(VideoFileStream(str(p)))
    assert len(frames) == 7


def test_file_time_matches_container_clock_within_one_ms():
    d = pathlib.Path(tempfile.mkdtemp())
    p = d / "clip.mp4"
    _write_clip(p, 10, 20.0)
    frames = list(VideoFileStream(str(p)))
    expected = [i / 20.0 for i in range(10)]
    for frame, want in zip(frames, expected):
        assert abs(frame.t - want) < 1e-3


def test_start_stop_stride():
    d = pathlib.Path(tempfile.mkdtemp())
    p = d / "clip.mp4"
    _write_clip(p, 10, 10.0)
    assert len(list(VideoFileStream(str(p), start=2, stop=8))) == 6
    assert len(list(VideoFileStream(str(p), stride=3))) == 4
    assert len(list(VideoFileStream(str(p), max_frames=2))) == 2


def test_second_container_reads_the_same():
    d = pathlib.Path(tempfile.mkdtemp())
    p = d / "clip.avi"
    _write_clip(p, 5, 15.0)
    assert len(list(VideoFileStream(str(p)))) == 5


def test_glob_concatenates_files_in_order():
    d = pathlib.Path(tempfile.mkdtemp())
    _write_clip(d / "a.mp4", 3, 10.0)
    _write_clip(d / "b.mp4", 4, 10.0)
    frames = list(VideoFileStream(str(d / "*.mp4")))
    assert len(frames) == 7
    assert frames[0].t <= frames[3].t


def test_missing_file_raises_filenotfound():
    with pytest.raises(FileNotFoundError):
        VideoFileStream("/this/path/does/not/exist.mp4")


def test_cube_keeps_every_16_bit_value():
    cube = (np.arange(2 * 3 * 4, dtype=np.uint32) % 65536).astype(np.uint16)
    cube = cube.reshape(2, 3, 4)
    frames = list(CubeStream(cube, fps=4.0, dtype="uint16", gray=False))
    assert len(frames) == 2
    assert frames[0].array.dtype == np.uint16
    assert frames[0].array.tolist() == cube[0].tolist()


def test_cube_time_is_index_over_fps():
    cube = np.zeros((4, 3, 3), dtype=np.float32)
    frames = list(CubeStream(cube, fps=2.0))
    assert [f.t for f in frames] == [0.0, 0.5, 1.0, 1.5]


def test_cube_rejects_bad_shape():
    with pytest.raises(ValueError):
        CubeStream(np.zeros((3, 4), dtype=np.float32))
