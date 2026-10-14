"""Multi-output classifiers and regressors (one prediction per target).

Measured cost (p99 of one learn_one + predict_one, 3000 samples, 4
features, 3 targets, 1 CPU thread; state size in bytes; the base
classifier is LogisticRegression with SGD, the base regressor is
LinearRegression with SGD):

    PerOutputClassifier             p99 0.124 ms     871 B
    PerOutputRegressor              p99 0.121 ms     789 B
    ClassifierChain                 p99 0.284 ms     949 B
    RegressorChain                  p99 0.168 ms     867 B
    ProbabilisticClassifierChain    p99 0.879 ms     949 B

Per-output classifiers train one model per target independently.
Chains train one model per target on an expanded input that includes
the previous targets; ProbabilisticClassifierChain additionally
enumerates the 2**m labelings at prediction time, which is why it is
the slowest of the five. budget_s is 0.005 s for the per-output
baseline and 0.01 s for the chains, except ProbabilisticClassifierChain
at 0.05 s.


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
