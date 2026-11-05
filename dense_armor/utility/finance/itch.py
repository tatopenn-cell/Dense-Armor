"""Decode one Nasdaq TotalView-ITCH 5.0 order message."""

import struct

_LEN = {
    "A": 36,
    "F": 40,
    "E": 31,
    "C": 36,
    "X": 23,
    "D": 19,
    "U": 35,
}


def _head(buf: bytes) -> dict:
    return {
        "locate": struct.unpack_from(">H", buf, 1)[0],
        "tracking": struct.unpack_from(">H", buf, 3)[0],
        "timestamp": struct.unpack_from(">Q", b"\x00\x00" + buf[5:11])[0],
    }


def _alpha(b: bytes) -> str:
    return b.decode("ascii").rstrip(" ")


def _price(raw: int) -> float:
    return raw / 10000.0


def decode_message(buf: bytes) -> dict:
    """Decode one Nasdaq TotalView-ITCH 5.0 order message.

    The seven order messages ``A``, ``F``, ``E``, ``C``, ``X``, ``D`` and
    ``U`` are decoded. Integers are big endian and unsigned; alpha fields
    are ASCII and right padded with spaces (stripped); a ``Price (4)`` is
    returned as an integer number of ten-thousandths (``price_raw``) and
    as a float (``price``); the timestamp is 6 bytes of nanoseconds since
    midnight.

    Args:
        buf: the bytes of one message, without any framing.

    Returns:
        A dict with the key ``"type"`` and one key per field of the table.

    Raises:
        TypeError: if ``buf`` is not bytes.
        ValueError: if ``buf`` is empty, if its type is not one of the
            seven, or if its length is not the length of its type.

    Examples:
        >>> m = b"A" + (1).to_bytes(2, "big") + (2).to_bytes(2, "big")
        >>> m += (0).to_bytes(6, "big") + (7).to_bytes(8, "big")
        >>> m += b"B" + (100).to_bytes(4, "big")
        >>> m += b"AAPL    " + (1234567).to_bytes(4, "big")
        >>> d = decode_message(m)
        >>> d["type"], d["stock"], d["price"], d["price_raw"]
        ('A', 'AAPL', 123.4567, 1234567)
        >>> d["side"]
        'B'

    References:
        Nasdaq TotalView-ITCH 5.0, Data Types PDF page 4 and message
        layouts PDF pages 13 to 16.
    """
    if not isinstance(buf, bytes):
        raise TypeError(f"buf must be bytes, got {type(buf).__name__}")
    if len(buf) == 0:
        raise ValueError("buf is empty")
    t = chr(buf[0])
    if t not in _LEN:
        raise ValueError(f"unknown type {t!r}; expected one of {sorted(_LEN)}")
    if len(buf) != _LEN[t]:
        raise ValueError(f"length {len(buf)} is not the length of type {t} ({_LEN[t]})")
    out: dict = {"type": t}
    out.update(_head(buf))
    if t in ("A", "F"):
        out["order_ref"] = struct.unpack_from(">Q", buf, 11)[0]
        out["side"] = _alpha(buf[19:20])
        out["shares"] = struct.unpack_from(">I", buf, 20)[0]
        out["stock"] = _alpha(buf[24:32])
        raw = struct.unpack_from(">I", buf, 32)[0]
        out["price_raw"] = raw
        out["price"] = _price(raw)
        if t == "F":
            out["attribution"] = _alpha(buf[36:40])
    elif t == "E":
        out["order_ref"] = struct.unpack_from(">Q", buf, 11)[0]
        out["executed_shares"] = struct.unpack_from(">I", buf, 19)[0]
        out["match_number"] = struct.unpack_from(">Q", buf, 23)[0]
    elif t == "C":
        out["order_ref"] = struct.unpack_from(">Q", buf, 11)[0]
        out["executed_shares"] = struct.unpack_from(">I", buf, 19)[0]
        out["match_number"] = struct.unpack_from(">Q", buf, 23)[0]
        out["printable"] = _alpha(buf[31:32])
        raw = struct.unpack_from(">I", buf, 32)[0]
        out["price_raw"] = raw
        out["price"] = _price(raw)
    elif t == "X":
        out["order_ref"] = struct.unpack_from(">Q", buf, 11)[0]
        out["cancelled_shares"] = struct.unpack_from(">I", buf, 19)[0]
    elif t == "D":
        out["order_ref"] = struct.unpack_from(">Q", buf, 11)[0]
    elif t == "U":
        out["original_ref"] = struct.unpack_from(">Q", buf, 11)[0]
        out["new_ref"] = struct.unpack_from(">Q", buf, 19)[0]
        out["shares"] = struct.unpack_from(">I", buf, 27)[0]
        raw = struct.unpack_from(">I", buf, 31)[0]
        out["price_raw"] = raw
        out["price"] = _price(raw)
    return out
