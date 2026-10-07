"""Moved to `dense_armor.misc.iodat`; this path stays as an alias."""
import sys

from dense_armor.misc import iodat as _module

sys.modules[__name__] = _module
