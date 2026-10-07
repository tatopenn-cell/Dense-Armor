"""Moved to `dense_armor.learn.online_dynamics`; this path stays as an alias."""
import sys

from dense_armor.learn import online_dynamics as _module

sys.modules[__name__] = _module
