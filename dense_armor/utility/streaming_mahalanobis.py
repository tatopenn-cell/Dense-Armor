"""Moved to `dense_armor.anomaly.mahalanobis`; this path stays as an alias."""
import sys

from dense_armor.anomaly import mahalanobis as _module

sys.modules[__name__] = _module
