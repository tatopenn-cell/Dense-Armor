"""Reading and writing WAV files (RIFF) in PCM 16, 24, 32 bits and IEEE float 32."""

import struct
from pathlib import Path

import numpy as np

from dense_armor.core.audio.signal import Audio

_SUBTYPES = {
    "pcm16": (1, 16),
    "pcm24": (1, 24),
    "pcm32": (1, 32),
    "float32": (3, 32),
}


def write_wav(path: str | Path, audio: Audio, subtype: str = "pcm16") -> None:
    """Write ``audio`` to a WAV file.

    PCM samples are ``round(x * 2**(bits - 1))`` clipped to the integer range, so a
    sample in ``[-1, 1)`` is exact to one quantisation step.

    Args:
        path: output file.
        audio: the sound, samples in ``[-1, 1]``.
        subtype: ``"pcm16"``, ``"pcm24"``, ``"pcm32"`` or ``"float32"``.

    Raises:
        ValueError: if ``subtype`` is unknown.
    """
    if subtype not in _SUBTYPES:
        raise ValueError(
            f"unknown subtype {subtype!r}, expected one of {sorted(_SUBTYPES)}"
        )
    tag, bits = _SUBTYPES[subtype]
    x = np.asarray(audio.x, dtype=np.float64)
    x2 = x.reshape(-1, 1) if x.ndim == 1 else x
    if tag == 3:
        raw = x2.astype("<f4").tobytes()
    else:
        top = 2 ** (bits - 1)
        q = np.clip(np.round(x2 * top), -top, top - 1).astype(np.int64)
        if bits == 16:
            raw = q.astype("<i2").tobytes()
        elif bits == 32:
            raw = q.astype("<i4").tobytes()
        else:
            raw = b"".join(
                int(v).to_bytes(3, "little", signed=True) for v in q.reshape(-1)
            )
    ch = x2.shape[1]
    align = ch * bits // 8
    fmt = struct.pack(
        "<HHIIHH", tag, ch, int(audio.sr), int(audio.sr) * align, align, bits
    )
    body = b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt
    body += (
        b"data" + struct.pack("<I", len(raw)) + raw + (b"\x00" if len(raw) % 2 else b"")
    )
    Path(path).write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def read_wav(path: str | Path) -> Audio:
    """Read a WAV file written in PCM 16, 24, 32 bits or IEEE float 32.

    Args:
        path: input file.

    Returns:
        The sound with samples as float64 in ``[-1, 1)``.

    Raises:
        ValueError: if the file is not RIFF/WAVE or its format is not supported;
            the message names the format found.
    """
    b = Path(path).read_bytes()
    if b[:4] != b"RIFF" or b[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    pos = 12
    fmt = None
    data = None
    while pos + 8 <= len(b):
        cid = b[pos : pos + 4]
        size = struct.unpack("<I", b[pos + 4 : pos + 8])[0]
        chunk = b[pos + 8 : pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", chunk[:16])
        elif cid == b"data":
            data = chunk
        pos += 8 + size + (size % 2)
    if fmt is None or data is None:
        raise ValueError("missing fmt or data chunk")
    tag, ch, sr, _, _, bits = fmt
    if (tag, bits) not in set(_SUBTYPES.values()):
        raise ValueError(f"unsupported format: tag {tag}, {bits} bits")
    if tag == 3:
        x = np.frombuffer(data, dtype="<f4").astype(np.float64)
    elif bits == 16:
        x = np.frombuffer(data, dtype="<i2").astype(np.float64) / 2**15
    elif bits == 32:
        x = np.frombuffer(data, dtype="<i4").astype(np.float64) / 2**31
    else:
        v = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3).astype(np.int64)
        i = v[:, 0] | (v[:, 1] << 8) | (v[:, 2] << 16)
        i = np.where(i >= 2**23, i - 2**24, i)
        x = i.astype(np.float64) / 2**23
    x = x[: (x.size // ch) * ch]
    return Audio(x if ch == 1 else x.reshape(-1, ch), int(sr))
