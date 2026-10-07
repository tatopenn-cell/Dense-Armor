"""Moved to `dense_armor.drift.cusum`; this path stays as an alias."""
import sys

from dense_armor.drift import cusum as _module

sys.modules[__name__] = _module
