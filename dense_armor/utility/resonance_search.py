"""Moved to `dense_armor.anomaly.resonance_search`; this path stays as an alias."""
import sys

from dense_armor.anomaly import resonance_search as _module

sys.modules[__name__] = _module
