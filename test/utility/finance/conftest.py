"""Fixtures for the finance tests that need a real service.

``pg_url`` creates a fresh schema in the PostgreSQL of
``DENSE_ARMOR_TEST_PG_URL`` and sets it as the ``search_path``, so each
test starts from an empty database. Without PostgreSQL the fixture skips
with the reason written, unless ``DENSE_ARMOR_REQUIRE_SERVICES=1``, in
which case it fails.

``fix_acceptor`` and ``chromium_available`` are added by later tasks.
"""

from __future__ import annotations

import os
import uuid

import pytest


def _require(name: str) -> bool:
    return os.environ.get("DENSE_ARMOR_REQUIRE_SERVICES", "0") == "1"


@pytest.fixture
def pg_url():
    """A URL with a fresh, empty schema as ``search_path``."""
    base = os.environ.get("DENSE_ARMOR_TEST_PG_URL")
    if not base:
        if _require("postgres"):
            pytest.fail("DENSE_ARMOR_TEST_PG_URL is not set")
        pytest.skip("PostgreSQL not reachable: DENSE_ARMOR_TEST_PG_URL is not set")
    pytest.importorskip("sqlalchemy", reason="sqlalchemy not installed")
    pytest.importorskip("psycopg2", reason="psycopg2 not installed")
    import sqlalchemy
    if "?" in base:
        head, sep, tail = base.partition("?")
    else:
        head, sep, tail = base, "", ""
    schema = f"t_{uuid.uuid4().hex[:8]}"
    try:
        engine = sqlalchemy.create_engine(base)
        with engine.begin() as conn:
            conn.execute(sqlalchemy.text(f'CREATE SCHEMA "{schema}"'))
    except Exception as e:
        if _require("postgres"):
            pytest.fail(f"PostgreSQL not reachable at {base}: {e}")
        pytest.skip(f"PostgreSQL not reachable at {base}: {e}")
    opts = f"options=-csearch_path%3D{schema}"
    url = head + ("?" + tail + "&" + opts if sep else "?" + opts)
    try:
        yield url
    finally:
        with engine.begin() as conn:
            conn.execute(sqlalchemy.text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        engine.dispose()


@pytest.fixture
def chromium_available():
    """Skip if Chromium is not available; fail if REQUIRE_SERVICES=1."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        if _require("browser"):
            pytest.fail(f"Chromium not available: playwright is not installed: {e}")
        pytest.skip(f"Chromium not available: playwright is not installed: {e}")
    try:
        with sync_playwright() as p:
            b = p.chromium.launch()
            version = b.version
            b.close()
    except Exception as e:
        if _require("browser"):
            pytest.fail(f"Chromium not available: {e}")
        pytest.skip(f"Chromium not available: {e}")
    return version


@pytest.fixture
def fix_acceptor(tmp_path):
    """Start the test FIX acceptor in a subprocess, return its initiator cfg path.

    Two settings files are written in ``tmp_path``: one for the acceptor,
    one for the initiator. The port is a free port found by binding a
    socket to port 0. The acceptor is terminated at the end.
    """
    import socket
    import subprocess
    import sys
    import time
    from pathlib import Path as _P

    fix_import_error = None
    try:
        import quickfix  # noqa: F401
    except ImportError as e:
        fix_import_error = e

    if fix_import_error is not None:
        if _require("fix"):
            pytest.fail(f"QuickFIX not available: {fix_import_error}")
        pytest.skip(f"QuickFIX not available: {fix_import_error}")

    root = _P(__file__).resolve().parents[3]
    executor = root / "test" / "fixtures" / "finance" / "fix_executor.py"
    dictionary = root / "test" / "fixtures" / "finance" / "FIX44.xml"
    if not executor.exists() or not dictionary.exists():
        if _require("fix"):
            pytest.fail("acceptor fixture missing")
        pytest.skip("acceptor fixture missing")

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    # due settings file semplici
    acceptor_cfg = tmp_path / "acceptor.cfg"
    acceptor_cfg.write_text(
        "[DEFAULT]\n"
        "ConnectionType=acceptor\n"
        "BeginString=FIX.4.4\n"
        "SenderCompID=SERVER\n"
        "TargetCompID=CLIENT\n"
        f"SocketAcceptPort={port}\n"
        "StartTime=00:00:00\n"
        "EndTime=00:00:00\n"
        "HeartBtInt=5\n"
        f"FileStorePath={tmp_path / 'store_acceptor'}\n"
        f"FileLogPath={tmp_path / 'log_acceptor'}\n"
        f"DataDictionary={dictionary}\n"
        "UseDataDictionary=Y\n"
        "\n"
        "[SESSION]\n"
        "BeginString=FIX.4.4\n"
        "SenderCompID=SERVER\n"
        "TargetCompID=CLIENT\n"
        "ConnectionType=acceptor\n",
        encoding="utf-8",
        newline="\n",
    )
    initiator_cfg = tmp_path / "initiator.cfg"
    initiator_cfg.write_text(
        "[DEFAULT]\n"
        "ConnectionType=initiator\n"
        "BeginString=FIX.4.4\n"
        "SenderCompID=CLIENT\n"
        "TargetCompID=SERVER\n"
        f"SocketConnectPort={port}\n"
        "SocketConnectHost=127.0.0.1\n"
        "StartTime=00:00:00\n"
        "EndTime=00:00:00\n"
        "HeartBtInt=5\n"
        "ReconnectInterval=1\n"
        f"FileStorePath={tmp_path / 'store_init'}\n"
        f"FileLogPath={tmp_path / 'log_init'}\n"
        f"DataDictionary={dictionary}\n"
        "UseDataDictionary=Y\n"
        "\n"
        "[SESSION]\n"
        "BeginString=FIX.4.4\n"
        "SenderCompID=CLIENT\n"
        "TargetCompID=SERVER\n"
        "ConnectionType=initiator\n",
        encoding="utf-8",
        newline="\n",
    )

    proc = subprocess.Popen(
        [sys.executable, str(executor), str(acceptor_cfg)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # wait for the port
    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.1)
    try:
        yield initiator_cfg
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
