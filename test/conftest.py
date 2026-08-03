# -*- coding: utf-8 -*-
import sys

import jax

jax.config.update("jax_enable_x64", True)

# I test CAT-A/B/D/E (test_bound*.py) stampano box-drawing/emoji decorativi
# nei loro report -- su Windows, sys.stdout di default usa la codepage
# della console (cp1252), che non li rappresenta e solleva
# UnicodeEncodeError a runtime (non un problema del motore, solo dei
# print di diagnostica). Forzare UTF-8 qui vale per l'intera sessione di
# test senza bisogno di impostare PYTHONIOENCODING esternamente.
if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding not in ("utf-8", "UTF-8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
