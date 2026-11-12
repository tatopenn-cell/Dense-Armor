# Audit table: the decision log in PostgreSQL

The decision log of F6 is a file: one writer at a time, no way to query it,
and anyone with the file can edit a record. An audit table in PostgreSQL keeps
the same hash chain and adds two things a file cannot have: a `SELECT` that
returns a page of records, and a trigger that refuses any `UPDATE` and any
`DELETE`. A record has the same hash in the file and in the table, because the
hash is one function that both use.

## 1. Start PostgreSQL and run the migration

`docker-compose.yml` starts a PostgreSQL 16 container. The migration is in
`alembic/versions/0001_decisions.py`; `alembic/env.py` reads the URL from
`DENSE_ARMOR_PG_URL` or from `-x url=...`.

~~~python
import os, subprocess, sys
work = "/content/Dense-Armor"
url = os.environ["DENSE_ARMOR_TEST_PG_URL"]
env = os.environ.copy()
env["DENSE_ARMOR_PG_URL"] = url
r = subprocess.run(
    [sys.executable, "-m", "alembic", "downgrade", "base"],
    capture_output=True, text=True, cwd=work, env=env,
)
r = subprocess.run(
    [sys.executable, "-m", "alembic", "upgrade", "head"],
    capture_output=True, text=True, cwd=work, env=env,
)
print("exit:", r.returncode)
print(r.stderr.strip())
~~~

~~~
exit: 0
INFO  [alembic.runtime.migration] Context impl PostgresqlImpl.
INFO  [alembic.runtime.migration] Will assume transactional DDL.
INFO  [alembic.runtime.migration] Running upgrade  -> 0001_decisions, create the decisions table and the trigger that forbids update and delete
~~~

The migration creates the table `decisions` with five columns (`seq`, `t`,
`payload`, `prev`, `hash`), a unique constraint on `prev` (two records cannot
follow the same one, so a fork is refused by the database), a unique constraint
on `hash`, and the trigger `decisions_no_change`.

## 2. Append three records and read them back

`AuditDB(url)` opens a SQLAlchemy connection. `append(t, payload)` writes one
row in one transaction: `pg_advisory_xact_lock`, read the last row, compute the
hash, insert. `verify()` reads the rows in batches and recomputes every hash.

~~~python
import os, sqlalchemy
from dense_armor.core.finance.audit_db import AuditDB
db = AuditDB(os.environ["DENSE_ARMOR_TEST_PG_URL"])
db.append(1.0, {"account": "AAPL", "reason": "ok"})
db.append(2.0, {"account": "MSFT", "reason": "size"})
db.append(3.0, {"account": "NVDA", "reason": "ok"})
e = sqlalchemy.create_engine(os.environ["DENSE_ARMOR_TEST_PG_URL"])
with e.connect() as conn:
    for row in conn.execute(sqlalchemy.text(
            "select seq, t, payload, prev, hash from decisions order by seq")):
        print(row[0], row[1], row[2], row[3][:8] + "...", row[4][:8] + "...")
print("verify:", db.verify())
~~~

~~~
0 1.0 {"account":"AAPL","reason":"ok"} 00000000... 53c903e6...
1 2.0 {"account":"MSFT","reason":"size"} 53c903e6... 2725282c...
2 3.0 {"account":"NVDA","reason":"ok"} 2725282c... 6b104b87...
verify: True
~~~

`prev` of the first row is `GENESIS_HASH` (64 zeros), `prev` of each other row
is the `hash` of the row before. The `payload` column keeps the canonical JSON
text that was hashed, not JSONB, which would not preserve the exact text. A
record has the same hash in the file of F6 and in this table, because both use
`record_hash` from `decision_log.py`.

## 3. The trigger refuses an UPDATE, and a tamper with the trigger off is detected

The trigger `decisions_no_change` raises an exception before any `UPDATE` and
any `DELETE`. The `verify()` method returns `False` when a record has been
edited with the trigger off.

~~~python
import os, sqlalchemy
e = sqlalchemy.create_engine(os.environ["DENSE_ARMOR_TEST_PG_URL"])
try:
    with e.begin() as conn:
        conn.execute(sqlalchemy.text("UPDATE decisions SET t = 999 WHERE seq = 0"))
except Exception as ex:
    print("UPDATE:", type(ex).__name__, "-", str(ex).splitlines()[0][:80])
from dense_armor.core.finance.audit_db import AuditDB
db = AuditDB(os.environ["DENSE_ARMOR_TEST_PG_URL"])
with e.begin() as conn:
    conn.execute(sqlalchemy.text("ALTER TABLE decisions DISABLE TRIGGER decisions_no_change"))
    conn.execute(
        sqlalchemy.text("UPDATE decisions SET payload = :p WHERE seq = 0"),
        {"p": '{"account":"AAPL","reason":"tampered"}'},
    )
    conn.execute(sqlalchemy.text("ALTER TABLE decisions ENABLE TRIGGER decisions_no_change"))
print("verify after tamper:", db.verify())
~~~

~~~
UPDATE: InternalError - (psycopg2.errors.RaiseException) decisions is append-only
verify after tamper: False
~~~

## Details

**References.**
- NIST. FIPS 180-4, *Secure Hash Standard*. August 2015. SHA-256 is
  section 6.2 (pages 27 and 28 of the PDF).
- Alembic documentation, *Auto Generating Migrations* (alembic.sqlalchemy.org).
  Used to compare the migration and the metadata of the table.
- SQLAlchemy documentation, *Engine Configuration* (docs.sqlalchemy.org).
  Used for the URL and the dialect check.

**Choices stated explicitly.**
- The `payload` column is `TEXT`, not `JSONB`. JSONB normalises the text of
  the object, so `{"a": 1}` and `{"a":1}` would be stored the same way and the
  hash recomputed from the database would differ from the hash stored in the
  row. `TEXT` keeps the exact text.
- `prev` has a unique constraint. Two records cannot follow the same one:
  a fork is refused by the database, not only by `verify()`.
- `append` takes `pg_advisory_xact_lock` on one fixed integer before it reads
  the last row and inserts. Two writers cannot insert the same `seq`.
- The hash is `record_hash` from `decision_log.py`, the same function that
  `DecisionLog` uses. The code is one function, not a copy: a record has the
  same hash in the file and in the table.
- The URL must be PostgreSQL. A SQLite URL is `ValueError`, not a fallback.

**Measured cost.** The `append` method takes an exclusive advisory lock and
reads the last row, so its cost is O(1) in the number of rows. `verify()`
reads the table in batches of 500 rows and recomputes every hash, so its cost
is O(n). The two numbers are not measured here.

**Declared limits.**
- The owner of the table can `ALTER TABLE ... DISABLE TRIGGER` and edit a
  record. The trigger is a guard against mistakes, not a security boundary.
  `verify()` catches the edit, but only the next time it runs.
- There is no signature. The hash is not signed, and no key is involved. A
  reader who trusts the code that wrote the table trusts the hash.
- There is no retention policy and no backup. The table is a live table, not a
  legal record.
- Deleting the last rows of the table leaves a valid chain. `verify()` cannot
  see it; the only defence is to keep `last_hash` somewhere else, and the
  class exposes it.
- One PostgreSQL 16 was tested. Another major version may work; it is not
  measured.
- No network, no credential beyond the throwaway password of the container
  in `docker-compose.yml` and the CI service.
- The migration is `0001_decisions`. If a schema change is needed, it is a
  new migration, not an edit of this one.

*These pages describe what the code does on the data shown. Not a forecast, not financial advice.*
