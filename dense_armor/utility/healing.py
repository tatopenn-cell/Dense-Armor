"""Moved to `dense_armor.protect.healing`; this path stays as an alias."""
import sys

from dense_armor.protect import healing as _module

sys.modules[__name__] = _module
