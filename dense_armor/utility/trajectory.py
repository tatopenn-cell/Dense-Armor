"""Moved to `dense_armor.control.trajectory`; this path stays as an alias."""
import sys

from dense_armor.control import trajectory as _module

sys.modules[__name__] = _module
