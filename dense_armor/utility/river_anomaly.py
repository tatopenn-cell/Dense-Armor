"""Moved to `dense_armor.anomaly.deviation`; this path stays as an alias."""
import sys

from dense_armor.anomaly import deviation as _module

sys.modules[__name__] = _module
