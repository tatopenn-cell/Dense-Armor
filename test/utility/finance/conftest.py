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
    import sqlalchemy

    base = os.environ.get("DENSE_ARMOR_TEST_PG_URL")
    if not base:
        if _require("postgres"):
            pytest.fail("DENSE_ARMOR_TEST_PG_URL is not set")
        pytest.skip("PostgreSQL not reachable: DENSE_ARMOR_TEST_PG_URL is not set")
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
