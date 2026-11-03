# -*- coding: utf-8 -*-
import sys

# I test CAT-A/B/D/E (test_bound*.py) stampano box-drawing/emoji decorativi
# nei loro report -- su Windows, sys.stdout di default usa la codepage
# della console (cp1252), che non li rappresenta e solleva
# UnicodeEncodeError a runtime (non un problema del motore, solo dei
# print di diagnostica). Forzare UTF-8 qui vale per l'intera sessione di
# test senza bisogno di impostare PYTHONIOENCODING esternamente.
if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding not in ("utf-8", "UTF-8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


import pytest  # noqa: E402


@pytest.fixture
def float64_default():
    """Run a test with the default dtype of the library at float64, then put it back."""
    from dense_armor import get_default_dtype, set_default_dtype

    old = get_default_dtype()
    set_default_dtype("float64")
    yield
    set_default_dtype(old)


@pytest.fixture
def x64():
    """Run a test with JAX in x64 mode, then put it back."""
    import jax

    old = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtestloop(session):
    """Drop unusable coverage files of the subprocesses before pytest-cov combines them.

    A subprocess that is stopped before it writes its data leaves an empty or truncated
    ``.coverage.*`` file. pytest-cov then fails the whole job at the end with
    ``attempt to write a readonly database``, although every test passed. The files that
    are not valid coverage databases hold no data: they are removed and listed in the log.
    """
    yield
    import glob
    import os
    import sqlite3

    bad = []
    for path in glob.glob(os.path.join(str(session.config.rootpath), ".coverage.*")):
        try:
            con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:
                con.execute("select version from coverage_schema").fetchone()
            finally:
                con.close()
        except Exception:
            bad.append(path)
    for path in bad:
        try:
            os.remove(path)
        except OSError:
            pass
    if bad:
        print(f"\nconftest: removed {len(bad)} unusable coverage file(s): " + ", ".join(os.path.basename(p) for p in bad))
