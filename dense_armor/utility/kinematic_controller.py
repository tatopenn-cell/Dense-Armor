"""Moved to `dense_armor.control.kinematic_controller`; this path stays as an alias."""
import sys

from dense_armor.control import kinematic_controller as _module

sys.modules[__name__] = _module
