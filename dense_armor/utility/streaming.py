"""Moved to `dense_armor.anomaly.streaming`; this path stays as an alias."""
import sys

from dense_armor.anomaly import streaming as _module

sys.modules[__name__] = _module
