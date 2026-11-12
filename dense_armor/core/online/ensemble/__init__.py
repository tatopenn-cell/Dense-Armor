"""Ensembles of online learners.

Each member learns from the stream one sample at a time; the ensemble
combines their predictions with a vote, an average or a weighted rule.
The classes here follow the papers cited in their own module.
"""

from dense_armor.core.online.ensemble.bagging import OnlineBaggingClassifier
from dense_armor.core.online.ensemble.boosting import OnlineBoostingClassifier
from dense_armor.core.online.ensemble.ewa import (
    ExponentiallyWeightedAverageRegressor,
)
from dense_armor.core.online.ensemble.stacking import StackingClassifier
from dense_armor.core.online.ensemble.voting import VotingClassifier

__all__ = [
    "ExponentiallyWeightedAverageRegressor",
    "OnlineBaggingClassifier",
    "OnlineBoostingClassifier",
    "StackingClassifier",
    "VotingClassifier",
]
