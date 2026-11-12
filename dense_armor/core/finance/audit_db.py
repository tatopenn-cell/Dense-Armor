"""The decision log of F6 in a PostgreSQL table, with the same hash chain."""

from __future__ import annotations

import math
from typing import Any

from dense_armor.core.finance.decision_log import GENESIS_HASH, record_hash

_LOCK_ID = 0x0DA1B  # one fixed integer for pg_advisory_xact_lock


def metadata():
    """Return the SQLAlchemy :class:`MetaData` of the ``decisions`` table.

    The definition matches the Alembic migration ``0001_decisions``, so
    ``alembic.autogenerate.compare_metadata`` after ``upgrade head``
    returns an empty list. The function imports SQLAlchemy inside itself,
    so importing this module does not import SQLAlchemy.

    Returns:
        A :class:`sqlalchemy.MetaData` with one table, ``decisions``.
    """
    import sqlalchemy

    md = sqlalchemy.MetaData()
    sqlalchemy.Table(
        "decisions",
        md,
        sqlalchemy.Column("seq", sqlalchemy.BigInteger, primary_key=True),
        sqlalchemy.Column("t", sqlalchemy.Float, nullable=False),
        sqlalchemy.Column("payload", sqlalchemy.Text, nullable=False),
        sqlalchemy.Column("prev", sqlalchemy.CHAR(64), nullable=False, unique=True),
        sqlalchemy.Column("hash", sqlalchemy.CHAR(64), nullable=False, unique=True),
    )
    return md


def _is_finite_number(v: Any) -> bool:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    try:
        return math.isfinite(float(v))
    except (OverflowError, ValueError):
        return False


def _validate(t: Any, payload: Any) -> tuple[float, dict]:
    if not _is_finite_number(t):
        raise ValueError(f"t must be a finite number, got {t!r}")
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")
    import json

    try:
        json.dumps(payload, allow_nan=False, ensure_ascii=True)
    except (TypeError, ValueError) as e:
        raise ValueError(f"payload is not JSON serializable: {e}") from e
    return float(t), payload


class AuditDB:
    """The decision log of F6 in a PostgreSQL table, with the same hash chain.

    The table ``decisions`` has five columns: ``seq`` (BIGINT primary key),
    ``t`` (DOUBLE PRECISION), ``payload`` (TEXT: the canonical JSON text
    that was hashed, not JSONB, which does not keep the text), ``prev``
    (CHAR(64), unique: a fork is refused by the database) and ``hash``
    (CHAR(64), unique). A trigger refuses any ``UPDATE`` and any ``DELETE``.

    The hash of a row is computed by
    :func:`dense_armor.core.finance.decision_log.record_hash`, the same
    function that ``DecisionLog`` uses, so the same record has the same
    hash in the file and in the table.

    Args:
        url: a SQLAlchemy URL of PostgreSQL, for example
            ``postgresql+psycopg2://user:pass@host:5432/db``. Any other
            dialect is ``ValueError``.

    Raises:
        ValueError: if the dialect of ``url`` is not ``postgresql``.
        sqlalchemy.exc.ProgrammingError: if the table ``decisions`` does not
            exist: run ``alembic upgrade head`` first.

    Examples:
        >>> import os
        >>> url = os.environ.get("DENSE_ARMOR_TEST_PG_URL")
        >>> url is None or url.startswith("postgresql")
        True
    """

    def __init__(self, url: str) -> None:
        import sqlalchemy

        self.url = url
        engine = sqlalchemy.create_engine(url)
        dialect = engine.dialect.name
        if dialect != "postgresql":
            engine.dispose()
            raise ValueError(f"only PostgreSQL is supported, got dialect {dialect}")
        self._engine = engine
        with self._engine.connect() as conn:
            conn.execute(sqlalchemy.text("select 1 from decisions limit 1"))

    def __len__(self) -> int:
        import sqlalchemy

        with self._engine.connect() as conn:
            return int(
                conn.execute(sqlalchemy.text("select count(*) from decisions")).scalar()
            )

    @property
    def last_hash(self) -> str:
        """The ``hash`` of the last row, read from the table at each call."""
        import sqlalchemy

        with self._engine.connect() as conn:
            h = conn.execute(
                sqlalchemy.text("select hash from decisions order by seq desc limit 1")
            ).scalar()
        return GENESIS_HASH if h is None else str(h)

    def append(self, t: float, payload: dict) -> str:
        """Append one record and return its ``hash``.

        In one transaction: take ``pg_advisory_xact_lock``, read the last
        row, compute the hash with ``record_hash``, insert. A refused input
        writes nothing: validation runs before any statement.

        Args:
            t: the timestamp in seconds, a finite number.
            payload: a JSON object (``dict``) that ``json.dumps`` can write
                with ``allow_nan=False``.

        Returns:
            The 64 hex characters of the ``hash`` of the new record.

        Raises:
            ValueError: if ``t`` is not finite or ``payload`` is not a
                serializable dict.
        """
        import json

        import sqlalchemy

        t, payload = _validate(t, payload)
        text = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        with self._engine.begin() as conn:
            conn.execute(
                sqlalchemy.text("select pg_advisory_xact_lock(:k)"),
                {"k": _LOCK_ID},
            )
            row = conn.execute(
                sqlalchemy.text(
                    "select coalesce(max(seq), -1), "
                    "(select hash from decisions where seq = (select max(seq) from decisions)) "
                    "from decisions"
                )
            ).one()
            max_seq = int(row[0])
            prev = GENESIS_HASH if max_seq < 0 else str(row[1])
            seq = max_seq + 1
            h = record_hash(seq, t, payload, prev)
            conn.execute(
                sqlalchemy.text(
                    "insert into decisions (seq, t, payload, prev, hash) "
                    "values (:seq, :t, :payload, :prev, :hash)"
                ),
                {"seq": seq, "t": t, "payload": text, "prev": prev, "hash": h},
            )
        return h

    def verify(self) -> bool:
        """Return ``True`` if every row is a valid chain, ``False`` otherwise.

        The rows are read in batches, ordered by ``seq``. For each row the
        stored payload text is hashed again, and ``seq``, ``prev`` and
        ``hash`` are checked. A difference, a missing row or a garbage
        text gives ``False``. An empty table is ``True``. The method does
        not raise: a connection that cannot be opened or a failed read
        also gives ``False``, because the chain could not be verified.

        Returns:
            ``True`` for a valid chain, ``False`` otherwise.
        """
        import json

        import sqlalchemy

        prev = GENESIS_HASH
        expected_seq = 0
        batch = 500
        offset = 0
        try:
            with self._engine.connect() as conn:
                while True:
                    rows = conn.execute(
                        sqlalchemy.text(
                            "select seq, t, payload, prev, hash from decisions "
                            "order by seq limit :lim offset :off"
                        ),
                        {"lim": batch, "off": offset},
                    ).all()
                    if not rows:
                        break
                    for row in rows:
                        seq, t, payload_text, row_prev, row_hash = row
                        if int(seq) != expected_seq or str(row_prev) != prev:
                            return False
                        try:
                            payload = json.loads(str(payload_text))
                        except Exception:
                            return False
                        try:
                            h = record_hash(int(seq), float(t), payload, str(row_prev))
                        except (ValueError, RecursionError, OverflowError):
                            return False
                        if h != str(row_hash):
                            return False
                        prev = h
                        expected_seq += 1
                    offset += batch
        except Exception:
            return False
        return True
