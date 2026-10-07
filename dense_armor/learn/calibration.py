"""
Online Platt scaling (Gupta and Ramdas, ICML 2023, arXiv:2305.00070) as a
river classifier wrapper. Needs the optional dependency: pip install dense-armor[river]
"""
from __future__ import annotations

import math

import numpy as np

try:
    from river import base
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "dense_armor.learn.calibration needs river: pip install dense-armor[river]"
    ) from exc


class OnlinePlattScaling(base.Wrapper, base.Classifier):
    """Online Platt scaling.

    Recalibrates the probabilities of a classifier. The update happens at every new example, so
    the mapping adapts even when the data distribution changes over time. Given the probability
    `p` of the base classifier, the calibrated one is `sigmoid(a · logit(p) + b)`, with `a` and `b`
    updated online via Online Newton Step. If the base is already calibrated, the mapping tends to
    the identity and the probabilities remain almost unchanged.

    When the base classifies well but gives wrong numbers: the base classifier produces
    miscalibrated probabilities, typically because it is overconfident in its own predictions.
    This is the classic case of decision trees and random forests, which tend to output
    probabilities close to 0 or 1 even when they have no reason to. The accuracy or the ability to
    rank predictions of the base remain acceptable: the problem is not the classification, it is
    only the number attached to the prediction.

    The paper proves that the total log-loss of OPS over T examples exceeds that of the best fixed
    Platt mapping chosen in hindsight by an amount that grows at most as `log T`. In practice:
    online, without knowing what comes next, OPS does almost as well as the best calibration that
    could have been chosen knowing the future. The guarantee is on the log-loss, so the accuracy
    may vary slightly. The theorem holds under two conditions (with the base probabilities clipped
    to [0.01, 0.99], as the `clip` parameter does, and comparing against mappings with
    `a² + b² ≤ B²`). OPS does not guarantee that the probabilities improve, nor that the log-loss
    decreases: both depend on the base and on the data. OPS guarantees that its log-loss stays
    close to the best fixed Platt mapping in hindsight, which is already a strong result for an
    online method.

    Parameters
    ----------
    classifier
        The base classifier whose probabilities are to be recalibrated. It must expose
        predict_proba_one and learn_one.
    gamma
        Parameter of the Online Newton Step: the Newton step is scaled by 1/gamma. The paper uses
        0.1.
    rho
        Regularization of the ONS Hessian: it starts from rho · I. The paper uses 100.
    radius
        Radius of the l2 ball onto which (a, b) is projected after each step. The paper uses 100.
    clip
        The base probabilities are clipped between clip and 1 - clip before the logit. The paper
        assumes f(x) in [0.01, 0.99].

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import tree
    >>> from dense_armor.learn.calibration import OnlinePlattScaling

    >>> dataset = datasets.Phishing()

    >>> evaluate.progressive_val_score(dataset, tree.HoeffdingTreeClassifier(), metrics.LogLoss())
    LogLoss: 0.4535476064322544

    >>> model = OnlinePlattScaling(tree.HoeffdingTreeClassifier())
    >>> evaluate.progressive_val_score(dataset, model, metrics.LogLoss())
    LogLoss: 0.35021471255578124

    References
    ----------
    [^1]: [Gupta, C. and Ramdas, A., 2023. Online Platt Scaling with Calibeating. In International
    Conference on Machine Learning.](https://arxiv.org/abs/2305.00070)

    """

    def __init__(self, classifier: base.Classifier, gamma: float = 0.1, rho: float = 100.0,
                 radius: float = 100.0, clip: float = 0.01):
        self.classifier = classifier
        self.gamma = gamma
        self.rho = rho
        self.radius = radius
        self.clip = clip
        self._theta = np.array([1.0, 0.0])
        self._A = rho * np.eye(2)

    @property
    def _wrapped_model(self):
        return self.classifier

    @property
    def _multiclass(self):
        return False

    def _u(self, x, **kwargs):
        p = self.classifier.predict_proba_one(x, **kwargs).get(True, 0.5)
        p = min(max(p, self.clip), 1.0 - self.clip)
        return np.array([math.log(p / (1.0 - p)), 1.0])

    def _project(self, t):
        if t @ t <= self.radius**2:
            return t
        lo, hi = 0.0, 1.0
        f = lambda lam: np.linalg.solve(self._A + lam * np.eye(2), self._A @ t)
        while f(hi) @ f(hi) > self.radius**2:
            hi *= 2.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if f(mid) @ f(mid) > self.radius**2 else (lo, mid)
        return f(hi)

    def predict_proba_one(self, x, **kwargs):
        p = 1.0 / (1.0 + math.exp(-float(self._theta @ self._u(x, **kwargs))))
        return {False: 1.0 - p, True: p}

    def learn_one(self, x, y, **kwargs):
        u = self._u(x, **kwargs)
        g = (1.0 / (1.0 + math.exp(-float(self._theta @ u))) - float(y)) * u
        self._A += np.outer(g, g)
        self._theta = self._project(self._theta - np.linalg.solve(self._A, g) / self.gamma)
        self.classifier.learn_one(x, y, **kwargs)

    @classmethod
    def _unit_test_params(cls):
        from river import linear_model
        yield {"classifier": linear_model.LogisticRegression()}
