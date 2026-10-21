"""Classic (offline) algorithms: the algorithm of the paper on the whole dataset.

The online classes of ``dense_armor.utility`` are the other side of each pair;
``benchmarks/offline`` saves the difference between the two.
"""

from dense_armor.offline.base import ClassicBase
from dense_armor.offline.stats import *  # noqa: F401,F403

__all__ = ["ClassicBase"]
