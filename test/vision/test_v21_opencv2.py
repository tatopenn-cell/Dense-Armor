"""The file reader: source resolution, rotation and a scripted capture."""

import numpy as np
import pytest

from dense_armor.utility.vision.streams import VideoFileStream


def _clip_file(tmp_path):
    p = tmp_path / "x.mp4"
    p.write_bytes(b"not a video")
    return p


def test_video_file_stream_resolves_its_source(tmp_path):
    pytest.importorskip("cv2")
    p = _clip_file(tmp_path)
    assert VideoFileStream([p, p])._paths == [str(p), str(p)]
    assert VideoFileStream(p)._paths == [str(p)]
    url = "http://example.invalid/clip.mp4"
    assert VideoFileStream(url)._paths == [url]
    with pytest.raises(FileNotFoundError):
        VideoFileStream(str(tmp_path / "none_*.mp4"))
    with pytest.raises(FileNotFoundError):
        VideoFileStream(str(tmp_path / "missing.mp4"))
    with pytest.raises(TypeError):
        VideoFileStream(5)
    with pytest.raises(RuntimeError):
        list(VideoFileStream(p))


def test_video_file_stream_rotation(tmp_path):
    pytest.importorskip("cv2")
    p = _clip_file(tmp_path)
    frame = np.zeros((4, 6, 3), dtype=np.uint8)

    class Cap:
        def get(self, prop):
            return 90.0

    assert VideoFileStream(p, rotate=False)._apply_rotation(frame) is frame
    assert VideoFileStream(p, rotate=90)._apply_rotation(frame).shape[:2] == (6, 4)
    auto = VideoFileStream(p, rotate="auto")
    auto._last_cap = Cap()
    assert auto._apply_rotation(frame).shape[:2] == (6, 4)


def test_video_file_stream_with_a_scripted_capture(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    p = _clip_file(tmp_path)

    class Cap:
        def __init__(self):
            self.i = 0
            self.pos = [100.0, 50.0, 200.0]

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == 5:
                return 0.0
            return self.pos[min(self.i - 1, 2)] if self.i else 0.0

        def read(self):
            self.i += 1
            if self.i > 3:
                return False, None
            return True, np.full((6, 8, 3), 60, dtype=np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(VideoFileStream, "_open_one", lambda self, path: Cap())
    s = VideoFileStream(p, fps=10.0, size=(4, 3), follow=True, wait_s=0.0, max_waits=2, rotate=False)
    frames = list(s)
    ts = [f.t for f in frames]
    assert len(frames) == 3 and ts == sorted(ts) and frames[0].array.shape == (3, 4)
