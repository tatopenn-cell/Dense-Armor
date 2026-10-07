"""Moved to `dense_armor.anomaly.one_sided`; this path stays as an alias."""
import sys

from dense_armor.anomaly import one_sided as _module

sys.modules[__name__] = _module
