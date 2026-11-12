"""Audio streams: from a WAV file and from a live input.

The layout follows ``utility/signals/vision/streams.py``: the live library is imported
inside ``__init__``, and a block that the reader is too slow to take is counted as dropped.
"""

import queue
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from dense_armor.utility.audio.signal import Audio
from dense_armor.utility.audio.wav import read_wav


class AudioFileStream:
    """Blocks of ``block`` samples every ``hop`` samples from a WAV file.

    ``t0`` of a block is the index of its first sample divided by the sample rate. A last
    block shorter than ``block`` is not yielded.

    Args:
        path: WAV file readable by :func:`read_wav`.
        block: block length in samples.
        hop: step between blocks; defaults to ``block``.

    Example:
        >>> import tempfile, os
        >>> from dense_armor.utility.audio.wav import write_wav
        >>> p = os.path.join(tempfile.mkdtemp(), "a.wav")
        >>> write_wav(p, Audio(np.zeros(100), 100))
        >>> len(list(AudioFileStream(p, 50)))
        2
    """

    def __init__(
        self, path: str | Path, block: int = 1024, hop: int | None = None
    ) -> None:
        if block < 1:
            raise ValueError("block must be >= 1")
        self.path = path
        self.block = int(block)
        self.hop = int(hop) if hop else int(block)

    def __iter__(self) -> Iterator[Audio]:
        a = read_wav(self.path)
        for s in range(0, a.n - self.block + 1, self.hop):
            yield Audio(a.x[s : s + self.block], a.sr, s / a.sr)


class AudioLiveStream:
    """Blocks from a live input, taken through a bounded queue.

    A callback puts each block in a queue of ``queue_size`` places; when it is full the
    block is dropped and ``dropped`` grows by one. With no ``backend`` the input is
    ``sounddevice.InputStream`` (extra ``audio``), imported here; any object with
    ``start()``, ``stop()`` and ``close()`` that takes the callback can replace it.

    Args:
        sr: sample rate.
        block: block length in samples.
        queue_size: places in the queue.
        device: device passed to ``sounddevice``.
        backend: a function ``backend(callback, sr, block, device)`` returning a stream.
    """

    def __init__(
        self,
        sr: int = 44100,
        block: int = 1024,
        queue_size: int = 8,
        device: Any = None,
        backend: Any = None,
    ) -> None:
        self.sr = int(sr)
        self.block = int(block)
        self.queue_size = int(queue_size)
        self.device = device
        self.dropped = 0
        self._q: queue.Queue = queue.Queue(maxsize=self.queue_size)
        self._n = 0
        if backend is None:
            import sounddevice as sd

            def backend(cb, sr, block, device):
                return sd.InputStream(
                    samplerate=sr,
                    blocksize=block,
                    device=device,
                    channels=1,
                    callback=cb,
                )

        self._stream = backend(self._callback, self.sr, self.block, device)

    def _callback(self, indata, frames, time, status) -> None:
        b = Audio(
            np.array(indata, dtype=float).reshape(frames, -1)[:, 0],
            self.sr,
            self._n / self.sr,
        )
        self._n += frames
        try:
            self._q.put_nowait(b)
        except queue.Full:
            self.dropped += 1

    def start(self) -> "AudioLiveStream":
        """Start the input."""
        self._stream.start()
        return self

    def stop(self) -> None:
        """Stop and close the input."""
        self._stream.stop()
        self._stream.close()

    def read(self, timeout: float | None = None) -> Audio | None:
        """The next block, or None if none arrives within ``timeout`` seconds."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None
