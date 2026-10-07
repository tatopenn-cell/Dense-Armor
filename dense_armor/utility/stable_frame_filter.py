"""Moved to `dense_armor.protect.stable_frame_filter`; this path stays as an alias."""
import sys

from dense_armor.protect import stable_frame_filter as _module

sys.modules[__name__] = _module
