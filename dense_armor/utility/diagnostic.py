"""Moved to `dense_armor.misc.diagnostic`; this path stays as an alias."""
import sys

from dense_armor.misc import diagnostic as _module

sys.modules[__name__] = _module
