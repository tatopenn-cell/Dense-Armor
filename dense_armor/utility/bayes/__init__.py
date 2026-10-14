"""Naive Bayes classifiers on counts and categories.

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
