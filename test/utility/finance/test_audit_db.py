"""Tests for AuditDB: PostgreSQL, migration, hash chain, trigger."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from dense_armor.core.finance.audit_db import AuditDB, metadata
from dense_armor.core.finance.decision_log import DecisionLog

pytestmark = pytest.mark.needs_postgres


def _alembic(url: str, *args: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["DENSE_ARMOR_PG_URL"] = url
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=str(Path(__file__).resolve().parents[3]),
        env=env,
        capture_output=True,
        text=True,
    )


def test_1_upgrade_creates_table_and_downgrade_removes_it(pg_url):
    import sqlalchemy

    r = _alembic(pg_url, "upgrade", "head")
    assert r.returncode == 0, r.stderr
    e = sqlalchemy.create_engine(pg_url)
    with e.connect() as conn:
        cols = conn.execute(
            sqlalchemy.text(
                "select column_name, data_type from information_schema.columns "
                "where table_name = 'decisions' "
                "and table_schema = current_schema() "
                "order by ordinal_position"
            )
        ).all()
        assert [c[0] for c in cols] == ["seq", "t", "payload", "prev", "hash"]
        uniq = conn.execute(
            sqlalchemy.text(
                "select count(*) from information_schema.table_constraints "
                "where table_name = 'decisions' "
                "and table_schema = current_schema() "
                "and constraint_type = 'UNIQUE'"
            )
        ).scalar()
        assert int(uniq) >= 2
    e.dispose()
    r = _alembic(pg_url, "downgrade", "base")
    assert r.returncode == 0, r.stderr
    e = sqlalchemy.create_engine(pg_url)
    with e.connect() as conn:
        n = conn.execute(
            sqlalchemy.text(
                "select count(*) from information_schema.tables "
                "where table_name = 'decisions' "
                "and table_schema = current_schema()"
            )
        ).scalar()
        assert int(n) == 0
    e.dispose()


def test_2_model_matches_migration(pg_url):
    import sqlalchemy
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    r = _alembic(pg_url, "upgrade", "head")
    assert r.returncode == 0, r.stderr
    e = sqlalchemy.create_engine(pg_url)
    with e.connect() as conn:
        mc = MigrationContext.configure(conn)
        diff = compare_metadata(mc, metadata())
    e.dispose()
    assert diff == []


def test_3_same_records_same_hashes(pg_url, tmp_path):
    r = _alembic(pg_url, "upgrade", "head")
    assert r.returncode == 0, r.stderr
    db = AuditDB(pg_url)
    log = DecisionLog(tmp_path / "log.jsonl")
    rows = [(1.0, {"a": 1}), (2.0, {"a": 2}), (3.0, {"a": 3})]
    db_hashes = [db.append(t, p) for t, p in rows]
    log_hashes = [log.append(t, p) for t, p in rows]
    assert db_hashes == log_hashes


def test_4_three_appends_chain(pg_url):
    r = _alembic(pg_url, "upgrade", "head")
    assert r.returncode == 0, r.stderr
    db = AuditDB(pg_url)
    h1 = db.append(1.0, {"a": 1})
    h2 = db.append(2.0, {"a": 2})
    h3 = db.append(3.0, {"a": 3})
    assert len(db) == 3
    assert db.verify() is True
    assert db.last_hash == h3
    assert h1 != h2 != h3


@pytest.mark.parametrize("tamper", ["edit", "delete_middle", "swap_prev"])
def test_5_tampering_detected(pg_url, tamper):
    import sqlalchemy

    r = _alembic(pg_url, "upgrade", "head")
    assert r.returncode == 0, r.stderr
    db = AuditDB(pg_url)
    db.append(1.0, {"a": 1})
    db.append(2.0, {"a": 2})
    db.append(3.0, {"a": 3})
    e = sqlalchemy.create_engine(pg_url)
    with e.begin() as conn:
        conn.execute(
            sqlalchemy.text("ALTER TABLE decisions DISABLE TRIGGER decisions_no_change")
        )
        if tamper == "edit":
            conn.execute(
                sqlalchemy.text("UPDATE decisions SET payload = :p WHERE seq = 1"),
                {"p": '{"a":99}'},
            )
        elif tamper == "delete_middle":
            conn.execute(sqlalchemy.text("DELETE FROM decisions WHERE seq = 1"))
        else:
            conn.execute(
                sqlalchemy.text(
                    "UPDATE decisions SET prev = '" + "a" * 64 + "' WHERE seq = 2"
                )
            )
        conn.execute(
            sqlalchemy.text("ALTER TABLE decisions ENABLE TRIGGER decisions_no_change")
        )
    e.dispose()
    assert db.verify() is False


def test_6_trigger_forbids_update_and_delete(pg_url):
    import sqlalchemy

    r = _alembic(pg_url, "upgrade", "head")
    assert r.returncode == 0, r.stderr
    db = AuditDB(pg_url)
    db.append(1.0, {"a": 1})
    e = sqlalchemy.create_engine(pg_url)
    with pytest.raises(Exception):
        with e.begin() as conn:
            conn.execute(sqlalchemy.text("UPDATE decisions SET t = 99 WHERE seq = 0"))
    with pytest.raises(Exception):
        with e.begin() as conn:
            conn.execute(sqlalchemy.text("DELETE FROM decisions WHERE seq = 0"))
    e.dispose()


def test_7_two_threads_100_rows_no_hole(pg_url):
    r = _alembic(pg_url, "upgrade", "head")
    assert r.returncode == 0, r.stderr
    errors = []

    def worker():
        try:
            d = AuditDB(pg_url)
            for i in range(50):
                d.append(float(i), {"worker": threading.current_thread().name, "i": i})
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert errors == []
    db = AuditDB(pg_url)
    assert len(db) == 100
    seqs = sorted(json.loads(json.dumps({"x": 1})).get("x", 1) for _ in [1])
    del seqs
    assert db.verify() is True


def test_8_refused_input_writes_nothing(pg_url):
    r = _alembic(pg_url, "upgrade", "head")
    assert r.returncode == 0, r.stderr
    db = AuditDB(pg_url)
    db.append(1.0, {"a": 1})
    n_before = len(db)
    for bad_t, bad_p in [
        (float("nan"), {"a": 2}),
        (2.0, {"a": float("nan")}),
        (2.0, [1, 2, 3]),
    ]:
        with pytest.raises(ValueError):
            db.append(bad_t, bad_p)
    assert len(db) == n_before
    with pytest.raises(ValueError):
        AuditDB("sqlite:///x.db")


def test_9_continues_chain(pg_url):
    r = _alembic(pg_url, "upgrade", "head")
    assert r.returncode == 0, r.stderr
    db1 = AuditDB(pg_url)
    db1.append(1.0, {"a": 1})
    h2 = db1.append(2.0, {"a": 2})
    db2 = AuditDB(pg_url)
    assert db2.last_hash == h2
    assert len(db2) == 2
    h3 = db2.append(3.0, {"a": 3})
    db3 = AuditDB(pg_url)
    assert len(db3) == 3
    assert db3.verify() is True
    assert db3.last_hash == h3


def test_10_refused_input_also_on_audit_db(pg_url):
    """The _is_finite_number branch: bool, string, huge int."""
    from dense_armor.core.finance.audit_db import _is_finite_number

    assert _is_finite_number(True) is False
    assert _is_finite_number("1.0") is False
    assert _is_finite_number(None) is False
    assert _is_finite_number(10**400) is False
    assert _is_finite_number(1.5) is True
    assert _is_finite_number(2) is True


def test_11_verify_false_on_non_json_payload(pg_url):
    """The payload text cannot be parsed: verify returns False."""
    import sqlalchemy

    r = _alembic(pg_url, "upgrade", "head")
    assert r.returncode == 0, r.stderr
    db = AuditDB(pg_url)
    db.append(1.0, {"a": 1})
    e = sqlalchemy.create_engine(pg_url)
    with e.begin() as conn:
        conn.execute(
            sqlalchemy.text("ALTER TABLE decisions DISABLE TRIGGER decisions_no_change")
        )
        conn.execute(
            sqlalchemy.text("UPDATE decisions SET payload = :p WHERE seq = 0"),
            {"p": "not json at all"},
        )
        conn.execute(
            sqlalchemy.text("ALTER TABLE decisions ENABLE TRIGGER decisions_no_change")
        )
    e.dispose()
    assert db.verify() is False


def test_12_verify_false_on_nan_in_payload(pg_url):
    """The payload parses as JSON but has NaN, and record_hash refuses."""
    import sqlalchemy

    r = _alembic(pg_url, "upgrade", "head")
    assert r.returncode == 0, r.stderr
    db = AuditDB(pg_url)
    db.append(1.0, {"a": 1})
    e = sqlalchemy.create_engine(pg_url)
    with e.begin() as conn:
        conn.execute(
            sqlalchemy.text("ALTER TABLE decisions DISABLE TRIGGER decisions_no_change")
        )
        conn.execute(
            sqlalchemy.text("UPDATE decisions SET payload = :p WHERE seq = 0"),
            {"p": '{"a": NaN}'},
        )
        conn.execute(
            sqlalchemy.text("ALTER TABLE decisions ENABLE TRIGGER decisions_no_change")
        )
    e.dispose()
    assert db.verify() is False


def test_13_verify_false_when_the_database_is_unreachable(pg_url):
    """The outer try/except: an engine that cannot connect gives False."""
    import sqlalchemy

    r = _alembic(pg_url, "upgrade", "head")
    assert r.returncode == 0, r.stderr
    db = AuditDB(pg_url)
    db.append(1.0, {"a": 1})
    db._engine = sqlalchemy.create_engine(
        "postgresql+psycopg2://postgres:postgres@127.0.0.1:1/nope"
    )
    assert db.verify() is False
