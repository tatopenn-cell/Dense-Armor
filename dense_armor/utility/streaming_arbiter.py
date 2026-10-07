"""Moved to `dense_armor.protect.streaming_arbiter`; this path stays as an alias."""
import sys

from dense_armor.protect import streaming_arbiter as _module

sys.modules[__name__] = _module
