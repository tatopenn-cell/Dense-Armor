"""Strategies to turn a binary classifier into a multiclass one.

Measured cost (p99 of one learn_one + predict_one, 3000 samples, 4
features, 1 CPU thread; state size in bytes; the base classifier is
LogisticRegression with SGD):

    OneVsRestClassifier    p99 0.116 ms     704 B
    OneVsOneClassifier     p99 0.049 ms     562 B
    OutputCodeClassifier   p99 0.357 ms   1 462 B

OneVsRest trains one model per class, OneVsOne one model per pair of
classes, OutputCode one model per column of the code matrix. budget_s
is 0.005 s, 0.01 s and 0.01 s.


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
