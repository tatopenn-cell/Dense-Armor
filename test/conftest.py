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
