"""Strategies to turn a binary classifier into a multiclass one.

Three wrappers, all taking a fresh binary ``Classifier`` per internal
problem: OneVsRest (one-vs-all), OneVsOne (round-robin voting),
OutputCode (error-correcting output codes).
"""

from dense_armor.utility.multiclass.multiclass import (
    OneVsOneClassifier,
    OneVsRestClassifier,
    OutputCodeClassifier,
)

__all__ = [
    "OneVsOneClassifier",
    "OneVsRestClassifier",
    "OutputCodeClassifier",
]
