"""Configuration of the finance blocks: one frozen dataclass per block."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dense_armor.utility.finance.pretrade import PreTradeChecker
from dense_armor.utility.finance.ring import RingBuffer
from dense_armor.utility.finance.schedule import twap_schedule


@dataclass(frozen=True)
class ScheduleConfig:
    """Parameters of :func:`twap_schedule`."""

    total: float
    n: int


@dataclass(frozen=True)
class PreTradeConfig:
    """Parameters of :class:`PreTradeChecker`."""

    max_notional: float
    max_size: float
    price_band: float
    max_orders: int
    window_s: float
    duplicate_window_s: float


@dataclass(frozen=True)
class RingConfig:
    """Parameters of :class:`RingBuffer`."""

    capacity: int
    width: int


@dataclass(frozen=True)
class PathsConfig:
    """The three paths a configuration names, resolved against the file."""

    log: Path
    feed: Path
    emergency_stop: Path


@dataclass(frozen=True)
class FinanceConfig:
    """A configuration that the finance blocks accept.

    The four attributes are :class:`ScheduleConfig`,
    :class:`PreTradeConfig`, :class:`RingConfig` and :class:`PathsConfig`.
    A configuration that loads is a configuration that the blocks accept:
    :func:`load_config` builds each block once and raises the block's own
    ``ValueError``, prefixed with the table name.

    Examples:
        >>> from pathlib import Path
        >>> c = FinanceConfig(
        ...     schedule=ScheduleConfig(total=10.0, n=4),
        ...     pretrade=PreTradeConfig(
        ...         max_notional=1e6, max_size=10.0, price_band=0.01,
        ...         max_orders=5, window_s=1.0, duplicate_window_s=0.1),
        ...     ring=RingConfig(capacity=3, width=2),
        ...     paths=PathsConfig(
        ...         log=Path("log.jsonl"), feed=Path("feed.bin"),
        ...         emergency_stop=Path("STOP")))
        >>> isinstance(c.make_checker(), PreTradeChecker)
        True
        >>> isinstance(c.make_ring(), RingBuffer)
        True
    """

    schedule: ScheduleConfig
    pretrade: PreTradeConfig
    ring: RingConfig
    paths: PathsConfig

    def make_checker(self) -> PreTradeChecker:
        """Return a new :class:`PreTradeChecker` with the pretrade values."""
        return PreTradeChecker(
            max_notional=self.pretrade.max_notional,
            max_size=self.pretrade.max_size,
            price_band=self.pretrade.price_band,
            max_orders=self.pretrade.max_orders,
            window_s=self.pretrade.window_s,
            duplicate_window_s=self.pretrade.duplicate_window_s,
        )

    def make_ring(self) -> RingBuffer:
        """Return a new :class:`RingBuffer` with the ring values."""
        return RingBuffer(capacity=self.ring.capacity, width=self.ring.width)


_TABLES: dict[str, dict[str, str]] = {
    "schedule": {"total": "number", "n": "integer"},
    "pretrade": {
        "max_notional": "number",
        "max_size": "number",
        "price_band": "number",
        "window_s": "number",
        "duplicate_window_s": "number",
        "max_orders": "integer",
    },
    "ring": {"capacity": "integer", "width": "integer"},
    "paths": {"log": "string", "feed": "string", "emergency_stop": "string"},
}


def _check_type(table: str, key: str, kind: str, v: Any) -> None:
    name = f"{table}.{key}"
    if kind == "number":
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{name} must be a number, got {type(v).__name__}")
    elif kind == "integer":
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"{name} must be an integer, got {type(v).__name__}")
    elif kind == "string":
        if not isinstance(v, str):
            raise ValueError(f"{name} must be a string, got {type(v).__name__}")


def _check_structure(doc: dict[str, Any]) -> None:
    for table in doc:
        if table not in _TABLES:
            raise ValueError(f"unknown table: {table}")
    for table, keys in _TABLES.items():
        if table not in doc:
            raise ValueError(f"missing table: {table}")
        if not isinstance(doc[table], dict):
            raise ValueError(
                f"{table} must be a table, got {type(doc[table]).__name__}"
            )
        for key in doc[table]:
            if key not in keys:
                raise ValueError(f"unknown key: {table}.{key}")
        for key, kind in keys.items():
            if key not in doc[table]:
                raise ValueError(f"missing key: {table}.{key}")
            _check_type(table, key, kind, doc[table][key])


def _resolve_paths(folder: Path, raw: dict[str, str]) -> PathsConfig:
    def one(name: str) -> Path:
        p = Path(raw[name])
        return p if p.is_absolute() else (folder / p)

    return PathsConfig(
        log=one("log"), feed=one("feed"), emergency_stop=one("emergency_stop")
    )


def _build(doc: dict[str, Any], path: Path) -> FinanceConfig:
    folder = path.resolve().parent

    total = float(doc["schedule"]["total"])
    n = int(doc["schedule"]["n"])

    max_notional = float(doc["pretrade"]["max_notional"])
    max_size = float(doc["pretrade"]["max_size"])
    price_band = float(doc["pretrade"]["price_band"])
    max_orders = int(doc["pretrade"]["max_orders"])
    window_s = float(doc["pretrade"]["window_s"])
    duplicate_window_s = float(doc["pretrade"]["duplicate_window_s"])

    capacity = int(doc["ring"]["capacity"])
    width = int(doc["ring"]["width"])

    paths = _resolve_paths(folder, doc["paths"])

    try:
        twap_schedule(total, n)
    except ValueError as e:
        raise ValueError(f"[schedule]: {e}") from e

    try:
        _ = PreTradeChecker(
            max_notional=max_notional,
            max_size=max_size,
            price_band=price_band,
            max_orders=max_orders,
            window_s=window_s,
            duplicate_window_s=duplicate_window_s,
        )
    except ValueError as e:
        raise ValueError(f"[pretrade]: {e}") from e

    try:
        _ = RingBuffer(capacity=capacity, width=width)
    except ValueError as e:
        raise ValueError(f"[ring]: {e}") from e

    return FinanceConfig(
        schedule=ScheduleConfig(total=total, n=n),
        pretrade=PreTradeConfig(
            max_notional=max_notional,
            max_size=max_size,
            price_band=price_band,
            max_orders=max_orders,
            window_s=window_s,
            duplicate_window_s=duplicate_window_s,
        ),
        ring=RingConfig(capacity=capacity, width=width),
        paths=paths,
    )


def load_config(path: str | Path) -> FinanceConfig:
    """Read a TOML file and return a frozen :class:`FinanceConfig`.

    The file has four tables, ``schedule``, ``pretrade``, ``ring`` and
    ``paths``. An integer is accepted where a number is expected and is
    stored as ``float``; a ``bool``, a string, or a float where an integer
    is expected is refused. Each table is a frozen dataclass. The three
    paths are :class:`pathlib.Path` objects, resolved against the folder
    of the configuration file when they are relative.

    After the structure is checked, the function builds once a
    :class:`PreTradeChecker` and a :class:`RingBuffer` and calls
    :func:`twap_schedule`, so the ranges are validated by the blocks of
    F1, F2 and F4 and not by a second copy of their rules. An error from
    them is raised again as :class:`ValueError` whose message starts with
    the table name, ``[pretrade]: ...``.

    The TOML syntax (tables, integers, floats, strings, ``nan`` and ``inf``
    as valid floats) is the one of the TOML v1.0.0 specification in
    specification, sections ``Table``, ``Integer``, ``Float``, ``String``.
    The specification is not in this repository; the citation is
    bibliographic.

    Args:
        path: the path of the TOML file. A missing file raises
            :class:`FileNotFoundError`.

    Returns:
        A frozen :class:`FinanceConfig` with the four tables.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the text is not TOML, a table or a key is missing,
            a table or a key is unknown, or a value has the wrong type.
            A value out of range raises the ``ValueError`` of the block,
            prefixed with the table name.
        RuntimeError: if ``tomllib`` is not available (Python 3.10).

    Examples:
        >>> from pathlib import Path
        >>> doc = {
        ...     "schedule": {"total": 100.0, "n": 4},
        ...     "pretrade": {"max_notional": 1e6, "max_size": 10.0,
        ...                  "price_band": 0.01, "max_orders": 5,
        ...                  "window_s": 1.0, "duplicate_window_s": 0.1},
        ...     "ring": {"capacity": 3, "width": 2},
        ...     "paths": {"log": "log.jsonl", "feed": "feed.bin",
        ...               "emergency_stop": "STOP"},
        ... }
        >>> c = _build(doc, Path("/tmp/c.toml"))
        >>> (c.schedule.total, c.schedule.n, c.ring.capacity, c.ring.width)
        (100.0, 4, 3, 2)
    """
    try:
        import tomllib
    except ImportError as e:
        raise RuntimeError("tomllib needs Python 3.11 or later") from e
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    doc = tomllib.loads(text)
    _check_structure(doc)
    return _build(doc, p)
