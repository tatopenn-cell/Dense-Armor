"""Moved to `dense_armor.anomaly.filters`; this path stays as an alias."""
import sys

from dense_armor.anomaly import filters as _module

sys.modules[__name__] = _module
