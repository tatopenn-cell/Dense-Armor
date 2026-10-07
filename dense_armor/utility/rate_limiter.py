"""Moved to `dense_armor.control.rate_limiter`; this path stays as an alias."""
import sys

from dense_armor.control import rate_limiter as _module

sys.modules[__name__] = _module
