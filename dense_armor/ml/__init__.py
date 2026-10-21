"""Machine learning algorithms: the algorithm of the paper on the whole dataset.

Each module is a family of the library. The online classes of
``dense_armor.utility`` are a separate part of the library.
"""

from dense_armor.ml import (  # noqa: F401
    bayes,
    ensemble,
    forest,
    linear,
    multiclass,
    multioutput,
    neighbors,
    optim,
    preprocessing,
    proba,
    rules,
    stats,
    tree,
)
from dense_armor.ml.base import MLBase

__all__ = ["MLBase", "bayes", "ensemble", "forest", "linear", "multiclass", "multioutput", "neighbors", "optim", "preprocessing", "proba", "rules", "stats", "tree"]
