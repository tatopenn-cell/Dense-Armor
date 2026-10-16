"""Readers and writers that need OpenCV or Pillow, and the message when OpenCV is missing."""

import sys

import numpy as np
import pytest

from dense_armor.utility.vision.streams import CameraStream, ImageFolderStream, VideoFileStream
from dense_armor.utility.vision.visualize import VideoWriter


def test_missing_opencv_gives_a_clear_import_error(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(ImportError, match="OpenCV"):
        CameraStream()
    with pytest.raises(ImportError, match="OpenCV"):
        VideoWriter(tmp_path / "a.mp4")
    with pytest.raises(ImportError, match="OpenCV"):
        VideoFileStream(str(tmp_path / "a.mp4"))


def test_image_folder_stream_reads_png_files(tmp_path):
    pil = pytest.importorskip("PIL.Image")
    for i in range(3):
        pil.fromarray((np.full((8, 8, 3), 40 * i, dtype=np.uint8))).save(tmp_path / f"f{i}.png")
    frames = list(ImageFolderStream(tmp_path, fps=2.0, size=(4, 4), gray=True))
    assert len(frames) == 3 and frames[2].t == 1.0 and frames[0].array.shape == (4, 4)
    with pytest.raises(ValueError):
        ImageFolderStream([])


def test_video_writer_writes_a_file_and_a_summary(tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("matplotlib")
    out = tmp_path / "out.mp4"
    summary = tmp_path / "summary.png"
    w = VideoWriter(out, fps=5.0, size=(16, 12), summary_path=summary)
    frame = np.random.default_rng(0).random((12, 16, 3)).astype(np.float32)
    for i in range(4):
        w.write(frame)
        w.log("energy", float(i))
        w.log("score", float(i) ** 2)
    w.write(np.zeros((24, 32, 3), dtype=np.float32))
    w.close()
    assert out.exists() and out.stat().st_size > 0
    assert summary.exists() and summary.stat().st_size > 0
    w.close()
    with pytest.raises(ValueError):
        VideoWriter(tmp_path / "b.mp4", fps=0.0)


def test_video_writer_refuses_an_unwritable_path(tmp_path):
    pytest.importorskip("cv2")
    w = VideoWriter(tmp_path / "no_such_dir" / "x.mp4", fps=5.0)
    with pytest.raises(RuntimeError):
        w.write(np.zeros((8, 8, 3), dtype=np.float32))


def test_camera_stream_with_a_fake_device(monkeypatch):
    cv2 = pytest.importorskip("cv2")

    class Fake:
        def __init__(self, device):
            self.n = 0
            self.released = False

        def isOpened(self):
            return True

        def set(self, prop, value):
            return True

        def read(self):
            self.n += 1
            if self.n > 3:
                return False, None
            return True, np.full((6, 8, 3), 50 * self.n, dtype=np.uint8)

        def release(self):
            self.released = True

    monkeypatch.setattr(cv2, "VideoCapture", Fake)
    frames = list(CameraStream(device=0, fps=10.0, size=(4, 3), gray=True))
    assert len(frames) == 3 and frames[0].array.shape == (3, 4)
    capped = list(CameraStream(max_frames=2, size=None, gray=False))
    assert len(capped) == 2 and capped[0].array.shape == (6, 8, 3)
    with pytest.raises(ValueError):
        CameraStream(fps=0.0)


def test_camera_stream_that_cannot_open(monkeypatch):
    cv2 = pytest.importorskip("cv2")

    class Closed:
        def __init__(self, device):
            pass

        def isOpened(self):
            return False

    monkeypatch.setattr(cv2, "VideoCapture", Closed)
    with pytest.raises(RuntimeError):
        list(CameraStream())
