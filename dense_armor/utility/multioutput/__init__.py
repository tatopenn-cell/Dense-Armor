"""Multi-output classifiers and regressors (one prediction per target).

A multi-output estimator takes ``x`` and a dict of targets
``y = {target_name: value}`` and returns a dict with one prediction
per target. Five classes:

- PerOutputClassifier / PerOutputRegressor: independent models, one
  per target (the single-target baseline).
- ClassifierChain / RegressorChain: one model per target, chained;
  each model also sees the targets of the previous links.
- ProbabilisticClassifierChain: computes the joint P(y | x) by the
  product rule and predicts the joint mode.
"""

from dense_armor.utility.multioutput.multioutput import (
    ClassifierChain,
    PerOutputClassifier,
    PerOutputRegressor,
    ProbabilisticClassifierChain,
    RegressorChain,
)

__all__ = [
    "ClassifierChain",
    "PerOutputClassifier",
    "PerOutputRegressor",
    "ProbabilisticClassifierChain",
    "RegressorChain",
]
