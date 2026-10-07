"""Moved to `dense_armor.protect.orca`; this path stays as an alias."""
import sys

from dense_armor.protect import orca as _module

sys.modules[__name__] = _module
