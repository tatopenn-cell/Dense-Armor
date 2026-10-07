"""Moved to `dense_armor.learn.calibration`; this path stays as an alias."""
import sys

from dense_armor.learn import calibration as _module

sys.modules[__name__] = _module
