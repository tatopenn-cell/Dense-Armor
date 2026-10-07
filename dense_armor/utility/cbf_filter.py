"""Moved to `dense_armor.control.cbf_filter`; this path stays as an alias."""
import sys

from dense_armor.control import cbf_filter as _module

sys.modules[__name__] = _module
