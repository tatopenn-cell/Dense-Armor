"""Moved to `dense_armor.learn.metric_learning`; this path stays as an alias."""
import sys

from dense_armor.learn import metric_learning as _module

sys.modules[__name__] = _module
