"""Conjugate distributions updated one observation at a time.

Each distribution keeps its sufficient statistics and supports
``update(x, w)`` to add an observation and ``revert(x, w)`` to
remove one. ``update`` followed by ``revert`` on the same x with
the same weight restores the state exactly (to floating point).
"""

from dense_armor.utility.proba.beta import Beta
from dense_armor.utility.proba.gaussian import Gaussian
from dense_armor.utility.proba.multinomial import Multinomial

__all__ = ["Beta", "Gaussian", "Multinomial"]
