"""
Online classifiers for robot states with drift adaptation.

Three classifiers for one-sample-at-a-time learning on robot joint signals:
an online Gaussian naive Bayes, an online softmax regression, and a
`DriftAdaptiveClassifier` wrapper that swaps in a background copy when a
drift detector fires. All return calibrated-ready probabilities, so
`OnlinePlattScaling` (binary case) can wrap them.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Optional

import numpy as np

try:
    from river import base
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "dense_armor.learn.online_classifiers needs river: "
        "pip install dense-armor[river]"
    ) from exc


class OnlineGaussianNB(base.Classifier):
    """Gaussian naive Bayes updated one sample at a time.

    Running mean and variance per class and feature, updated by Welford's
    algorithm. With ``alpha = 0`` the statistics equal the batch MLE of a
    Gaussian NB on the observed samples, so ``predict_proba_one`` matches a
    batch Gaussian NB trained on the same data (to numerical precision).
    With ``alpha > 0`` the statistics forget exponentially at rate
    ``alpha``, so the classifier adapts when the class-conditional
    distributions drift.

    Parameters
    ----------
    alpha
        Forgetting factor in ``[0, 1)``. ``0`` is batch behaviour, larger
        values weight recent samples more.
    var_smoothing
        Additive smoothing on the variance. Default 0 so ``alpha = 0``
        matches the batch MLE exactly.
    eps
        Floor on the variance to avoid ``log(0)`` when a class has a single
        sample.

    Examples
    --------
    >>> from dense_armor.learn.online_classifiers import OnlineGaussianNB
    >>> nb = OnlineGaussianNB()
    >>> for x, y in [({"a": 0.0, "b": 0.0}, 0),
    ...              ({"a": 0.1, "b": 0.1}, 0),
    ...              ({"a": 1.0, "b": 1.0}, 1),
    ...              ({"a": 1.1, "b": 1.1}, 1)]:
    ...     _ = nb.learn_one(x, y)
    >>> nb.predict_one({"a": 0.05, "b": 0.05})
    0
    >>> nb.predict_one({"a": 1.05, "b": 1.05})
    1

    References
    ----------
    Welford, B. P. (1962). Note on a method for calculating corrected sums
    of squares and products. Technometrics 4(3), 419-420.
    """

    def __init__(self, alpha: float = 0.0, var_smoothing: float = 0.0,
                 eps: float = 1e-12):
        self.alpha = alpha
        self.var_smoothing = var_smoothing
        self.eps = eps
        self._keys: Optional[list] = None
        self._n: dict = {}
        self._mu: dict = {}
        self._m2: dict = {}

    @property
    def _multiclass(self):
        return True

    def _resolve_keys(self, x) -> list:
        if self._keys is None:
            self._keys = sorted(x, key=str)
        return self._keys

    def _to_vec(self, x) -> np.ndarray:
        keys = self._resolve_keys(x)
        return np.array([float(x.get(k, 0.0)) for k in keys], dtype=float)

    def learn_one(self, x, y):
        z = self._to_vec(x)
        if y not in self._n:
            self._n[y] = 0.0
            self._mu[y] = np.zeros(z.size)
            self._m2[y] = np.zeros(z.size)
        n_old = self._n[y]
        n_new = (1.0 - self.alpha) * n_old + 1.0
        delta = z - self._mu[y]
        new_mu = self._mu[y] + delta / n_new
        new_m2 = (1.0 - self.alpha) * self._m2[y] + delta * (z - new_mu)
        self._n[y] = n_new
        self._mu[y] = new_mu
        self._m2[y] = new_m2
        return self

    def _log_likelihood(self, z, y) -> float:
        var = self._m2[y] / self._n[y] + self.var_smoothing
        var = np.maximum(var, self.eps)
        return float(-0.5 * np.sum(np.log(2.0 * np.pi * var)
                                    + (z - self._mu[y]) ** 2 / var))

    def predict_proba_one(self, x) -> dict:
        if not self._n:
            return {}
        z = self._to_vec(x)
        total = sum(self._n.values())
        log_post = {y: self._log_likelihood(z, y)
                    + float(np.log(self._n[y] / total)) for y in self._n}
        mx = max(log_post.values())
        ex = {y: float(np.exp(v - mx)) for y, v in log_post.items()}
        s = sum(ex.values())
        return {y: p / s for y, p in ex.items()}

    @classmethod
    def _unit_test_params(cls):
        yield {}

    def _unit_test_skips(self):
        return set()


class OnlineSoftmaxRegression(base.Classifier):
    """Multinomial logistic regression with AdaGrad.

    Online softmax over ``W^T z`` where ``z = [features..., 1]``. Classes
    are added on the fly when a new label arrives. Optional L2 penalty,
    optional AdaGrad preconditioning of the learning rate.

    Parameters
    ----------
    eta
        Learning rate. The adaptive step is ``eta / (sqrt(G) + eps)`` where
        ``G`` is the running sum of squared gradients. Default 0.1.
    l2
        L2 penalty added to the loss gradient. Default 0.
    adaptive
        If True, AdaGrad preconditioning. If False, plain SGD with step
        ``eta``.
    eps
        Numerical floor in the AdaGrad denominator.

    Examples
    --------
    >>> from dense_armor.learn.online_classifiers import OnlineSoftmaxRegression
    >>> sr = OnlineSoftmaxRegression(eta=0.5)
    >>> for _ in range(20):
    ...     _ = sr.learn_one({"a": 0.0, "b": 0.0}, 0)
    ...     _ = sr.learn_one({"a": 1.0, "b": 1.0}, 1)
    >>> sr.predict_one({"a": 0.0, "b": 0.0})
    0
    >>> sr.predict_one({"a": 1.0, "b": 1.0})
    1

    References
    ----------
    Duchi, J., Hazan, E., Singer, Y. (2011). Adaptive subgradient methods
    for online learning and stochastic optimization. JMLR 12, 2121-2159.
    """

    def __init__(self, eta: float = 0.1, l2: float = 0.0,
                 adaptive: bool = True, eps: float = 1e-8):
        self.eta = eta
        self.l2 = l2
        self.adaptive = adaptive
        self.eps = eps
        self._keys: Optional[list] = None
        self._W: dict = {}
        self._G: dict = {}
        self._classes: list = []

    @property
    def _multiclass(self):
        return True

    def _resolve_keys(self, x) -> list:
        if self._keys is None:
            self._keys = sorted(x, key=str)
        return self._keys

    def _to_vec(self, x) -> np.ndarray:
        keys = self._resolve_keys(x)
        return np.array([float(x.get(k, 0.0)) for k in keys] + [1.0], dtype=float)

    def _softmax(self, logits):
        mx = max(logits.values())
        ex = {c: float(np.exp(v - mx)) for c, v in logits.items()}
        s = sum(ex.values())
        return {c: p / s for c, p in ex.items()}

    def learn_one(self, x, y):
        z = self._to_vec(x)
        if y not in self._W:
            self._W[y] = np.zeros(z.size)
            self._G[y] = np.ones(z.size)
            self._classes.append(y)
        logits = {c: float(self._W[c] @ z) for c in self._classes}
        probs = self._softmax(logits)
        for c in self._classes:
            g = (probs[c] - (1.0 if c == y else 0.0)) * z
            if self.l2 > 0.0:
                g = g + self.l2 * self._W[c]
            if self.adaptive:
                self._G[c] = self._G[c] + g * g
                step = self.eta / (np.sqrt(self._G[c]) + self.eps)
            else:
                step = self.eta
            self._W[c] = self._W[c] - step * g
        return self

    def predict_proba_one(self, x) -> dict:
        if not self._classes:
            return {}
        z = self._to_vec(x)
        logits = {c: float(self._W[c] @ z) for c in self._classes}
        return self._softmax(logits)

    @classmethod
    def _unit_test_params(cls):
        yield {}

    def _unit_test_skips(self):
        return set()


class DriftAdaptiveClassifier(base.Classifier):
    """Classifier with a background copy swapped in on drift detection.

    Wraps any ``base.Classifier`` with a drift detector. Per sample:
    predict, compute the log-loss ``-log(p_correct)``, feed the (smoothed)
    log-loss to the detector, learn, and if the detector fires start a
    background copy trained only on new samples. Main and background are
    both scored prequentially (each predicts a sample before learning it);
    when the background's accuracy over the last ``window`` samples beats
    the main classifier's over the same samples, the background is promoted
    to main.

    Parameters
    ----------
    classifier
        The wrapped classifier; a fresh copy is trained as the background.
    detector
        A drift detector with ``update(x)`` and ``drift_detected``, e.g.
        ``CUSUMDriftDetector(reference="fixed", radius=20, ref_mult=5,
        two_sided=False)``: one-sided, because only a rise of the loss
        means the classifier went stale.
    window
        Size of the accuracy sliding window used to compare main and
        background.
    smooth_window
        Size of the rolling mean applied to the log-loss before feeding it
        to the detector. The raw log-loss is near zero on most samples, so
        its robust scale is tiny and every single error looks like a shift;
        a rolling mean over ~50 samples gives the detector a stable scale.

    Examples
    --------
    >>> from dense_armor.learn.online_classifiers import (
    ...     OnlineGaussianNB, DriftAdaptiveClassifier,
    ... )
    >>> from dense_armor.drift.detector import CUSUMDriftDetector
    >>> clf = DriftAdaptiveClassifier(
    ...     OnlineGaussianNB(alpha=0.01),
    ...     CUSUMDriftDetector(reference="adaptive", radius=10, ref_mult=3),
    ...     smooth_window=20,
    ... )
    >>> for _ in range(200):
    ...     _ = clf.learn_one({"a": 0.0}, 0)
    ...     _ = clf.learn_one({"a": 1.0}, 1)
    >>> clf.predict_one({"a": 0.05})
    0
    >>> clf.predict_one({"a": 0.95})
    1

    References
    ----------
    Gama, J., Medas, P., Castillo, G., Rodrigues, P. (2004). Learning with
    drift detection. In SBIA.
    """

    def __init__(self, classifier, detector, window: int = 50,
                 smooth_window: int = 50):
        self.classifier = classifier
        self.detector = detector
        self.window = window
        self.smooth_window = smooth_window
        self._main = deepcopy(classifier)
        self._bg = None
        self._main_hits = deque(maxlen=window)
        self._bg_hits = deque(maxlen=window)
        self._err_buf = deque(maxlen=max(1, smooth_window))

    @property
    def _multiclass(self):
        return self.classifier._multiclass

    def learn_one(self, x, y):
        proba = self._main.predict_proba_one(x)
        pred = max(proba, key=proba.get) if proba else None
        self._main_hits.append(1.0 if pred == y else 0.0)
        p = proba.get(y, 1.0 / max(len(proba), 1)) if proba else 0.5
        loss = float(-np.log(max(p, 1e-6)))
        self._err_buf.append(loss)
        self.detector.update(float(np.mean(self._err_buf)))
        if self.detector.drift_detected:
            self._bg = deepcopy(self.classifier)
            self._bg_hits.clear()
        self._main.learn_one(x, y)
        if self._bg is not None:
            self._bg_hits.append(1.0 if self._bg.predict_one(x) == y else 0.0)
            self._bg.learn_one(x, y)
            if (len(self._bg_hits) >= self.window
                    and len(self._main_hits) >= self.window
                    and float(np.mean(self._bg_hits)) > float(np.mean(self._main_hits))):
                self._main = self._bg
                self._bg = None
                self._main_hits = self._bg_hits
                self._bg_hits = deque(maxlen=self.window)
        return self

    def predict_proba_one(self, x) -> dict:
        return self._main.predict_proba_one(x)

    @classmethod
    def _unit_test_params(cls):
        from dense_armor.drift.detector import CUSUMDriftDetector
        yield {"classifier": OnlineGaussianNB(),
               "detector": CUSUMDriftDetector(reference="adaptive")}

    def _unit_test_skips(self):
        return {"check_roc_auc"}
