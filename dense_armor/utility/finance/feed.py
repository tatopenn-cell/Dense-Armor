"""Read a file of framed Nasdaq ITCH 5.0 order messages."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from dense_armor.utility.finance.itch import decode_message

MAX_MESSAGE = 65535


def frame(message: bytes | bytearray | memoryview) -> bytes:
    """Return the 2-byte length followed by the message bytes.

    The container is our choice, not the ITCH specification: each record is a 2-byte
    big-endian unsigned length ``L`` followed by ``L`` bytes of one
    message. It is not part of the ITCH specification.

    Args:
        message: the bytes of one ITCH message, without framing.
            ``bytes``, ``bytearray`` and ``memoryview`` are accepted.

    Returns:
        ``len(message).to_bytes(2, "big") + message``, as ``bytes``.

    Raises:
        TypeError: if ``message`` is not ``bytes``, ``bytearray`` or
            ``memoryview``.
        ValueError: if ``message`` is empty or longer than 65,535 bytes.

    Examples:
        >>> len(frame(b"abc")), frame(b"abc")[2:]
        (5, b'abc')
        >>> frame(b"")
        Traceback (most recent call last):
        ...
        ValueError: message is empty
    """
    if isinstance(message, (bytearray, memoryview)):
        message = bytes(message)
    if not isinstance(message, bytes):
        raise TypeError(
            f"message must be bytes, bytearray or memoryview, "
            f"got {type(message).__name__}"
        )
    if len(message) == 0:
        raise ValueError("message is empty")
    if len(message) > MAX_MESSAGE:
        raise ValueError(f"message longer than {MAX_MESSAGE} bytes: {len(message)}")
    return len(message).to_bytes(2, "big") + message


def iter_itch(path: str | Path) -> Iterator[dict]:
    """Yield each ITCH message of a framed file, decoded by ``decode_message``.

    The file is opened when the first item is requested. One record is read
    at a time (``read(2)`` then ``read(L)``); the file is closed when the
    generator ends or is closed. An empty file yields nothing, without
    error.

    Args:
        path: the path of the file, as :class:`str` or :class:`pathlib.Path`.

    Yields:
        The dict returned by :func:`decode_message` for each message, in
        the order of the file.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if a length prefix is one byte only, if a body is
            shorter than the length, if a length is 0, or if
            :func:`decode_message` refuses a record. The message
            contains ``offset <n>`` (the byte offset of the start of the
            bad record) and how many messages were already yielded.
    """
    p = Path(path)
    offset = 0
    yielded = 0
    with p.open("rb") as f:
        while True:
            head = f.read(2)
            if not head:
                return
            if len(head) < 2:
                raise ValueError(
                    f"offset {offset}: length prefix is 1 byte, not 2 "
                    f"(after {yielded} messages)"
                )
            n = int.from_bytes(head, "big")
            if n == 0:
                raise ValueError(
                    f"offset {offset}: record has length 0 (after {yielded} messages)"
                )
            body = f.read(n)
            if len(body) < n:
                raise ValueError(
                    f"offset {offset}: body is {len(body)} bytes, "
                    f"expected {n} (after {yielded} messages)"
                )
            try:
                msg = decode_message(body)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"offset {offset}: {e} (after {yielded} messages)"
                ) from e
            yield msg
            offset += 2 + n
            yielded += 1
