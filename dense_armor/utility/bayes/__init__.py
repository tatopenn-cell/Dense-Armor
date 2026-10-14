"""Naive Bayes classifiers on counts and categories.

Measured cost (p99 of one learn_one + predict_one, 3000 samples, 4
features, 1 CPU thread; state size in bytes after 3000 samples):

    MultinomialNB   p99 0.036 ms   421 B
    BernoulliNB     p99 0.041 ms   380 B
    ComplementNB    p99 0.047 ms   408 B
    CategoricalNB   p99 21.5 ms    244 311 B

The three count-based classifiers are O(1) in the stream length and
O(c d) in the number of classes and features. CategoricalNB is slower
and larger because it keeps one counter per (class, feature, category)
seen so far, and on continuous inputs every value becomes a new
category; the count grows with the number of distinct values, not with
the stream length only. budget_s is 0.001 s for the first three and
0.1 s for CategoricalNB.


Four variants: MultinomialNB (word counts), BernoulliNB (presence
or absence of a feature), ComplementNB (weights from the complement
class), CategoricalNB (categorical features, no counts). Each one
learns from dict inputs and produces a probability distribution
over the classes seen so far.
"""

from dense_armor.utility.bayes.bernoulli import BernoulliNB
from dense_armor.utility.bayes.categorical import CategoricalNB
from dense_armor.utility.bayes.complement import ComplementNB
from dense_armor.utility.bayes.multinomial import MultinomialNB

__all__ = [
    "BernoulliNB",
    "CategoricalNB",
    "ComplementNB",
    "MultinomialNB",
]
