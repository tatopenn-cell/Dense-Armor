"""Machine learning algorithms: the algorithm of the paper on the whole dataset.

Each module is a family of the library. The online classes of
``dense_armor.utility`` are a separate part of the library.
"""

from dense_armor.ml import (  # noqa: F401
    anomaly,
    bayes,
    drift,
    linear,
    multiclass,
    multioutput,
    neighbors,
    optim,
    preprocessing,
    proba,
    stats,
    tree,
)
from dense_armor.ml.base import MLBase

__all__ = ["MLBase", "anomaly", "bayes", "drift", "linear", "multiclass", "multioutput", "neighbors", "optim", "preprocessing", "proba", "stats", "tree"]
