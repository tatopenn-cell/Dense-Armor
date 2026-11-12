"""A decision log with a SHA-256 hash chain, one JSON line per record."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

GENESIS_HASH: str = "0" * 64

_KEYS = ("hash", "payload", "prev", "seq", "t")


def _canonical(seq: int, t: float, payload: Any, prev: str) -> bytes:
    text = json.dumps(
        {"seq": seq, "t": t, "payload": payload, "prev": prev},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return text.encode("ascii")


def record_hash(seq: int, t: float, payload: Any, prev: str) -> str:
    """Return the SHA-256 of the canonical text of four keys.

    The canonical text is ``json.dumps({"seq": seq, "t": t,
    "payload": payload, "prev": prev}, sort_keys=True,
    separators=(",", ":"), ensure_ascii=True, allow_nan=False)`` encoded
    in ASCII. This is the function that computes the hash of a record of
    ``DecisionLog`` and of the row of ``AuditDB``: both use it, so the
    same record has the same hash in the file and in the table.

    Args:
        seq: the sequence number, 0 for the first record.
        t: the timestamp, a finite number.
        payload: the payload, a JSON-serializable dict.
        prev: the ``hash`` of the previous record, ``GENESIS_HASH`` for
            the first.

    Returns:
        The 64 lowercase hexadecimal characters of the SHA-256.

    Raises:
        ValueError: if ``payload`` is not JSON-serializable with
            ``allow_nan=False``.

    Examples:
        >>> record_hash(0, 1.5, {"a": 1}, "0" * 64)[:8]
        'c1b491c7'
    """
    return hashlib.sha256(_canonical(seq, t, payload, prev)).hexdigest()


def _hash(seq: int, t: float, payload: Any, prev: str) -> str:
    return record_hash(seq, t, payload, prev)


def _is_finite_number(v: Any) -> bool:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    try:
        return math.isfinite(float(v))
    except (OverflowError, ValueError):
        return False


def _validate_input(t: Any, payload: Any) -> tuple[float, dict]:
    if not _is_finite_number(t):
        raise ValueError(f"t must be a finite number, got {t!r}")
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")
    try:
        json.dumps(payload, allow_nan=False, ensure_ascii=True)
    except (TypeError, ValueError) as e:
        raise ValueError(f"payload is not JSON serializable: {e}") from e
    return float(t), payload


def _parse_line(line: str, index: int) -> dict:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as e:
        raise ValueError(f"line {index} is not JSON: {e}") from e
    if not isinstance(obj, dict) or tuple(sorted(obj.keys())) != _KEYS:
        raise ValueError(f"line {index} has the wrong keys")
    if not isinstance(obj["seq"], int) or isinstance(obj["seq"], bool):
        raise ValueError(f"line {index} has a non-integer seq")
    if not _is_finite_number(obj["t"]):
        raise ValueError(f"line {index} has a non-finite t")
    if not isinstance(obj["payload"], dict):
        raise ValueError(f"line {index} has a non-dict payload")
    for k in ("prev", "hash"):
        v = obj[k]
        if (
            not isinstance(v, str)
            or len(v) != 64
            or any(c not in "0123456789abcdef" for c in v)
        ):
            raise ValueError(f"line {index} has an invalid {k}")
    return obj


class DecisionLog:
    r"""A JSON-lines file whose records carry the SHA-256 of the previous one.

    A record is one UTF-8 JSON line with five keys: ``seq`` (0 for the first
    line), ``t`` (finite float), ``payload`` (JSON object), ``prev`` (64 hex
    chars, the ``hash`` of the previous record or ``GENESIS_HASH`` for the
    first) and ``hash`` (64 hex chars). The hash is the SHA-256 of the
    canonical text of the other four keys, produced with
    ``json.dumps(..., sort_keys=True, separators=(",", ":"),
    ensure_ascii=True, allow_nan=False)`` encoded in ASCII.

    The file is created at the first :meth:`append`, not by the constructor.
    :meth:`verify` re-reads the file from disk and returns ``True`` only if
    every line is a valid record, ``seq`` is the line index, ``prev`` is the
    ``hash`` of the previous line (or ``GENESIS_HASH`` for the first) and
    ``hash`` equals the recomputed one.

    ``append`` re-reads and re-hashes the whole file at every call, so the
    cost is O(n) per record and O(n**2) for n records. ``append_many``
    verifies the file once for a whole batch and costs O(n).

    Args:
        path: the file path, as :class:`str` or :class:`pathlib.Path`.

    Attributes:
        path: the file path as a :class:`pathlib.Path`.
        last_hash: the ``hash`` of the last record, ``GENESIS_HASH`` for
            an empty log.
        GENESIS_HASH: 64 zeros, the ``prev`` of the first record.

    Raises:
        ValueError: from :meth:`append` when ``t`` is not a finite number,
            ``payload`` is not a JSON-serializable dict, or the existing
            log has a broken chain.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     log = DecisionLog(Path(d) / "log.jsonl")
        ...     h = log.append(1.5, {"a": 1})
        ...     (len(log), log.verify(), log.last_hash == h)
        (1, True, True)

    References:
        NIST. FIPS 180-4, *Secure Hash Standard*. August 2015. SHA-256 is
        section 6.2 (pages 27 and 28 of the PDF); message padding is
        section 5.1.1 (page 18). JSON Lines, jsonlines.org, version 1.0:
        UTF-8, one JSON value per line, line terminator ``\n``. The PDFs
        are not in this repository; the citations are bibliographic.
    """

    GENESIS_HASH = GENESIS_HASH

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.last_hash, self._len = self._read_state()

    def __len__(self) -> int:
        return self._len

    def _read_state(self) -> tuple[str, int]:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return GENESIS_HASH, 0
        try:
            with self.path.open("r", encoding="utf-8", newline="\n") as f:
                lines = f.read().splitlines()
        except (OSError, UnicodeDecodeError):
            return GENESIS_HASH, 0
        try:
            last = _parse_line(lines[-1], len(lines) - 1)
        except (ValueError, RecursionError):
            return GENESIS_HASH, 0
        return last["hash"], len(lines)

    def _verify_file(self) -> tuple[bool, str, int]:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return True, GENESIS_HASH, 0
        try:
            with self.path.open("r", encoding="utf-8", newline="\n") as f:
                text = f.read()
        except (OSError, UnicodeDecodeError):
            return False, GENESIS_HASH, 0
        if text and not text.endswith("\n"):
            return False, GENESIS_HASH, 0
        lines = text.splitlines()
        prev = GENESIS_HASH
        for i, line in enumerate(lines):
            try:
                obj = _parse_line(line, i)
                if obj["seq"] != i or obj["prev"] != prev:
                    return False, GENESIS_HASH, 0
                expected = _hash(obj["seq"], obj["t"], obj["payload"], obj["prev"])
                if obj["hash"] != expected:
                    return False, GENESIS_HASH, 0
            except (ValueError, RecursionError, OverflowError):
                return False, GENESIS_HASH, 0
            prev = obj["hash"]
        return True, prev, len(lines)

    def append(self, t: float, payload: dict) -> str:
        """Append one record and return its ``hash``.

        Every check runs before the write: a refused input leaves the file
        byte-for-byte unchanged. The file is opened, written and closed for
        each record (no ``fsync``, see the class docstring).

        Args:
            t: the timestamp in seconds, a finite number.
            payload: a JSON object (``dict``) that ``json.dumps`` can write
                with ``allow_nan=False``.

        Returns:
            The 64 hex characters of the ``hash`` of the new record.

        Raises:
            ValueError: if ``t`` is not finite, ``payload`` is not a
                serializable dict, or the existing file does not verify.
        """
        return self.append_many([(t, payload)])[0]

    def append_many(self, items: Iterable[tuple[float, dict]]) -> list[str]:
        """Append several records and return their ``hash`` values, in order.

        The file is verified once for the whole batch, so ``n`` records cost
        O(n) and not the O(n**2) of ``n`` calls of :meth:`append`. Every
        record is checked before anything is written: if one is refused, the
        file is byte-for-byte unchanged. The file is the same one that ``n``
        calls of :meth:`append` would write.

        Args:
            items: pairs ``(t, payload)``, each as for :meth:`append`.

        Returns:
            The hashes of the new records; an empty list for no items.

        Raises:
            ValueError: if a ``t`` is not finite, a ``payload`` is not a
                serializable dict, or the existing file does not verify.

        Examples:
            >>> import tempfile
            >>> from pathlib import Path
            >>> with tempfile.TemporaryDirectory() as d:
            ...     log = DecisionLog(Path(d) / "log.jsonl")
            ...     hs = log.append_many([(1.0, {"a": 1}), (2.0, {"a": 2})])
            ...     (len(hs), len(log), log.verify(), log.last_hash == hs[-1])
            (2, 2, True, True)
        """
        checked = [_validate_input(t, payload) for t, payload in items]
        if not checked:
            return []
        ok, prev, n = self._verify_file()
        if not ok:
            raise ValueError("log chain is broken; refusing to extend it")
        lines: list[str] = []
        hashes: list[str] = []
        for k, (t, payload) in enumerate(checked):
            h = _hash(n + k, t, payload, prev)
            record = {"seq": n + k, "t": t, "payload": payload, "prev": prev, "hash": h}
            lines.append(
                json.dumps(
                    record,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            )
            hashes.append(h)
            prev = h
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write("".join(lines))
        self.last_hash = prev
        self._len = n + len(checked)
        return hashes

    def verify(self) -> bool:
        """Return ``True`` if the file is a valid chain, ``False`` otherwise.

        A missing file and an empty file are a chain of zero records, so
        ``True``. The method never raises for the content of the file.

        Returns:
            ``True`` if every line is a valid record whose ``prev`` is the
            previous ``hash`` and whose ``hash`` is the SHA-256 of the
            canonical text of the other four keys; ``False`` otherwise.
        """
        return self._verify_file()[0]
