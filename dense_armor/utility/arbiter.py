"""Moved to `dense_armor.protect.arbiter`; this path stays as an alias."""
import sys

from dense_armor.protect import arbiter as _module

sys.modules[__name__] = _module
