"""Moved to `dense_armor.anomaly.robust_filters`; this path stays as an alias."""
import sys

from dense_armor.anomaly import robust_filters as _module

sys.modules[__name__] = _module
