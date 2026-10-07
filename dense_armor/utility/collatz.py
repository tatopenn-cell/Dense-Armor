"""Moved to `dense_armor.misc.collatz`; this path stays as an alias."""
import sys

from dense_armor.misc import collatz as _module

sys.modules[__name__] = _module
