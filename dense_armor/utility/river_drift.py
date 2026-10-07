"""Moved to `dense_armor.drift.detector`; this path stays as an alias."""
import sys

from dense_armor.drift import detector as _module

sys.modules[__name__] = _module
