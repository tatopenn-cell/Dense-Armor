"""Synthetic Nasdaq ITCH 5.0 order messages, deterministic from SHA-256."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

from dense_armor.utility.finance.feed import frame

STOCKS = ("AAPL", "MSFT", "NVDA", "AMZN")
START_TS = 34_200_000_000_000  # 09:30:00, nanoseconds since midnight


def _stream(seed: int) -> Iterator[int]:
    i = 0
    while True:
        block = hashlib.sha256(
            seed.to_bytes(8, "big") + (i // 32).to_bytes(8, "big")
        ).digest()
        for b in block:
            yield b
            i += 1


def _take(n: int, it: Iterator[int]) -> bytes:
    return bytes(next(it) for _ in range(n))


def _choose_type(b: int) -> str:
    if b < 96:
        return "A"
    if b < 128:
        return "F"
    if b < 160:
        return "E"
    if b < 176:
        return "C"
    if b < 200:
        return "X"
    if b < 232:
        return "D"
    return "U"


def _check_args(n: int, seed: int) -> tuple[int, int]:
    if isinstance(n, bool) or not isinstance(n, int):
        raise ValueError(f"n must be an int, got {type(n).__name__}")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an int, got {type(seed).__name__}")
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    return n, seed


def _build_head(msg_type: str, locate: int, ts: int) -> bytearray:
    b = bytearray()
    b.append(ord(msg_type))
    b.extend(locate.to_bytes(2, "big"))
    b.extend((0).to_bytes(2, "big"))  # tracking
    b.extend(ts.to_bytes(6, "big"))
    return b


def _stock_name(locate: int) -> bytes:
    return STOCKS[locate - 1].encode("ascii").ljust(8)


def synthetic_itch(n: int, seed: int) -> bytes:
    """Return ``n`` framed synthetic ITCH order messages for ``seed``.

    The same ``(n, seed)`` gives the same bytes on every machine: there is
    no ``random``, no clock, no NumPy, no Python-version dependent number.
    The source of the bytes is SHA-256: byte ``i`` of the stream of a
    seed is byte ``i % 32`` of
    ``sha256(seed.to_bytes(8, "big") + (i // 32).to_bytes(8, "big")).digest()``.

    The generator consumes bytes from this stream in the order written in
    the paragraph below, so that another implementation can reproduce the
    same bytes.

    Per message: one byte chooses the type (``< 96`` ``A``, ``< 128``
    ``F``, ``< 160`` ``E``, ``< 176`` ``C``, ``< 200`` ``X``, ``< 232``
    ``D``, else ``U``). If the type is not ``A`` or ``F`` and no order is
    live, the type is forced to ``A``. Then: 1 byte for ``locate``
    (``1 + byte % 4``, the stock is ``STOCKS[locate - 1]`` padded to 8
    characters with spaces, ``tracking`` is 0); 2 bytes for the timestamp
    delta ``1_000 + value % 1_000_000`` nanoseconds above the previous
    timestamp (start ``09:30:00``, always strictly increasing). For ``A``
    and ``F``: 1 byte for side (``B`` if even, ``S`` if odd), 2 bytes for
    ``shares`` (``1 + value % 1000``), 2 bytes for ``price_raw``
    (``1_000_000 + locate * 100_000 + (value % 2001) - 1000``), and ``F``
    adds the attribution ``b"SYNT"``. For ``E``, ``C``, ``X``: 2 bytes to
    choose one live order (``value % len(live)``), 1 byte for
    ``executed`` or ``cancelled`` shares (``1 + byte % 100``); ``E`` and
    ``C`` add a ``match_number`` counter from 1, ``C`` sets ``printable``
    to ``b"Y"`` and the executed ``price_raw``. ``D`` consumes 2 bytes to
    choose the live order to delete. ``U`` consumes 2 bytes for
    ``original_ref``, 2 bytes for ``shares``, 2 bytes for ``price_raw``,
    and uses the same stock and side of the original order.

    The ``locate`` is drawn from one byte for every message, including
    ``E``, ``C``, ``X``, ``D`` and ``U``. A real feed keeps the locate of
    the order an update refers to; the synthetic feed does not, and the
    ``locate`` of an update can differ from the one of the ``A`` it
    updates. This is a declared difference from the real feed, not a bug
    of the bytes.

    The feed has the byte layout of ITCH and none of its statistics: it is
    not a model of a market.

    Args:
        n: the number of messages; an ``int`` ``>= 0`` (a ``bool`` is
            refused).
        seed: the seed of the byte stream; an ``int`` ``>= 0``.

    Returns:
        The framed bytes of the ``n`` messages, in the container of
        :func:`frame`. ``n = 0`` returns ``b""``.

    Raises:
        ValueError: if ``n`` or ``seed`` is not a non-negative ``int``, or
            is a ``bool``.

    Examples:
        >>> data = synthetic_itch(3, 0)
        >>> len(data) > 0
        True
        >>> synthetic_itch(0, 0)
        b''
    """
    n, seed = _check_args(n, seed)
    if n == 0:
        return b""

    it = _stream(seed)
    parts: list[bytes] = []
    live: list[int] = []  # order_ref, in insertion order
    live_info: dict[int, tuple[int, str, int]] = {}  # ref -> (locate, side, price)
    ref_counter = 0
    match_counter = 0
    ts = START_TS

    for _ in range(n):
        b = next(it)
        msg_type = _choose_type(b)
        if msg_type not in ("A", "F") and not live:
            msg_type = "A"

        locate = 1 + next(it) % 4
        ts = ts + 1_000 + int.from_bytes(_take(2, it), "big") % 1_000_000
        head = _build_head(msg_type, locate, ts)

        if msg_type in ("A", "F"):
            ref_counter += 1
            ref = ref_counter
            side_b = next(it)
            side = "B" if side_b % 2 == 0 else "S"
            shares = 1 + int.from_bytes(_take(2, it), "big") % 1000
            price = (
                1_000_000
                + locate * 100_000
                + int.from_bytes(_take(2, it), "big") % 2001
                - 1000
            )
            head.extend(ref.to_bytes(8, "big"))
            head.extend(side.encode("ascii"))
            head.extend(shares.to_bytes(4, "big"))
            head.extend(_stock_name(locate))
            head.extend(price.to_bytes(4, "big"))
            if msg_type == "F":
                head.extend(b"SYNT")
            live.append(ref)
            live_info[ref] = (locate, side, price)
            parts.append(frame(bytes(head)))
            continue

        if msg_type == "D":
            idx = int.from_bytes(_take(2, it), "big") % len(live)
            ref = live.pop(idx)
            del live_info[ref]
            head.extend(ref.to_bytes(8, "big"))
            parts.append(frame(bytes(head)))
            continue

        if msg_type == "U":
            idx = int.from_bytes(_take(2, it), "big") % len(live)
            original = live.pop(idx)
            locate_o, side_o, _price_o = live_info.pop(original)
            ref_counter += 1
            new_ref = ref_counter
            shares = 1 + int.from_bytes(_take(2, it), "big") % 1000
            new_price = (
                1_000_000
                + locate_o * 100_000
                + int.from_bytes(_take(2, it), "big") % 2001
                - 1000
            )
            head.extend(original.to_bytes(8, "big"))
            head.extend(new_ref.to_bytes(8, "big"))
            head.extend(shares.to_bytes(4, "big"))
            head.extend(new_price.to_bytes(4, "big"))
            live.append(new_ref)
            live_info[new_ref] = (locate_o, side_o, new_price)
            parts.append(frame(bytes(head)))
            continue

        # E, C, X
        idx = int.from_bytes(_take(2, it), "big") % len(live)
        ref = live[idx]
        shares = 1 + next(it) % 100
        head.extend(ref.to_bytes(8, "big"))
        if msg_type == "X":
            head.extend(shares.to_bytes(4, "big"))
        else:
            match_counter += 1
            head.extend(shares.to_bytes(4, "big"))
            head.extend(match_counter.to_bytes(8, "big"))
            if msg_type == "C":
                head.extend(b"Y")
                _l, _s, price = live_info[ref]
                head.extend(price.to_bytes(4, "big"))
        parts.append(frame(bytes(head)))

    return b"".join(parts)
