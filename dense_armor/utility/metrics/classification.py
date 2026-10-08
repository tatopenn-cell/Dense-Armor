"""Online classification metrics.

Every metric keeps a small set of counters and updates them in constant
time per sample. ``merge`` combines two partial states exactly, so the
same metric can be computed over sharded streams.

The definitions follow scikit-learn: ``F1`` is the harmonic mean of
precision and recall (Sasaki 2007), Cohen's kappa is
``(p_o - p_e) / (1 - p_e)`` (Cohen 1960), and the log-loss and Brier
score are the standard proper losses (Brier 1950; Good 1952).

References
----------
Brier, G. W. (1950). Verification of forecasts expressed in terms of
    probability. Monthly Weather Review 78(1), 1-3.
Cohen, J. (1960). A coefficient of agreement for nominal scales.
    Educational and Psychological Measurement 20(1), 37-46.
Sasaki, Y. (2007). The truth of the F-measure. Teach Tutor Mater.
"""

import math
from typing import Any

from dense_armor.roles import Classifier
from dense_armor.utility.metrics.base import Metric


class Accuracy(Metric):
    """Fraction of samples where ``y_pred == y_true``.

    Examples:
        >>> from dense_armor.utility.metrics import Accuracy
        >>> m = Accuracy()
        >>> for a, b in [(0, 0), (1, 1), (0, 1), (1, 1)]:
        ...     _ = m.update(a, b)
        >>> m.get()
        0.75
    """

    role_ = Classifier
    bigger_is_better = True

    def __init__(self) -> None:
        super().__init__()
        self.hits_ = 0.0
        self.total_ = 0.0

    def _update(self, y_true: Any, y_pred: Any, w: float) -> None:
        if y_true == y_pred:
            self.hits_ += w
        self.total_ += w

    def _revert(self, y_true: Any, y_pred: Any, w: float) -> None:
        if y_true == y_pred:
            self.hits_ -= w
        self.total_ -= w

    def get(self) -> float:
        return self.hits_ / self.total_ if self.total_ > 0 else 0.0

    def merge(self, other: Metric) -> "Accuracy":
        """Combine two partial states.

        Args:
            other: another :class:`Accuracy` on a disjoint set of pairs.

        Returns:
            A new :class:`Accuracy` with the combined state.

        Raises:
            TypeError: if ``other`` is not an :class:`Accuracy`.
        """
        if not isinstance(other, Accuracy):
            raise TypeError(
                f"merge expects Accuracy, got {type(other).__name__}"
            )
        m = Accuracy()
        m.hits_ = self.hits_ + other.hits_
        m.total_ = self.total_ + other.total_
        m.n_ = self.n_ + other.n_
        m.n_missing_ = self.n_missing_ + other.n_missing_
        return m


class _PRFBase(Metric):
    """Common machinery for precision, recall and F-beta.

    Tracks true positives, false positives and false negatives per
    class, so both the binary and the multi-class averages come out of
    the same counters. Not part of the public API.
    """

    role_ = Classifier

    def __init__(
        self, positive: Any = True, average: str = "binary"
    ) -> None:
        super().__init__()
        self.positive = positive
        self.average = average
        self.tp_: dict[Any, float] = {}
        self.fp_: dict[Any, float] = {}
        self.fn_: dict[Any, float] = {}
        self.support_: dict[Any, float] = {}

    def _update(self, y_true: Any, y_pred: Any, w: float) -> None:
        self.support_[y_true] = self.support_.get(y_true, 0.0) + w
        if y_true == y_pred:
            self.tp_[y_true] = self.tp_.get(y_true, 0.0) + w
        else:
            self.fp_[y_pred] = self.fp_.get(y_pred, 0.0) + w
            self.fn_[y_true] = self.fn_.get(y_true, 0.0) + w

    def _revert(self, y_true: Any, y_pred: Any, w: float) -> None:
        self.support_[y_true] = self.support_.get(y_true, 0.0) - w
        if y_true == y_pred:
            self.tp_[y_true] = self.tp_.get(y_true, 0.0) - w
        else:
            self.fp_[y_pred] = self.fp_.get(y_pred, 0.0) - w
            self.fn_[y_true] = self.fn_.get(y_true, 0.0) - w

    def _classes(self) -> list:
        keys = set(self.tp_) | set(self.fp_) | set(self.fn_) | set(self.support_)
        return sorted(keys, key=str)

    def _binary_pr(self, cls: Any) -> tuple[float, float]:
        tp = self.tp_.get(cls, 0.0)
        fp = self.fp_.get(cls, 0.0)
        fn = self.fn_.get(cls, 0.0)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return p, r

    def _fbeta_for_class(self, cls: Any, beta: float) -> float:
        p, r = self._binary_pr(cls)
        b2 = beta * beta
        denom = b2 * p + r
        return (1.0 + b2) * p * r / denom if denom > 0 else 0.0

    def _aggregate_fbeta(self, beta: float) -> float:
        if self.average == "binary":
            return self._fbeta_for_class(self.positive, beta)
        classes = self._classes()
        if not classes:
            return 0.0
        if self.average == "micro":
            tp = sum(self.tp_.values())
            fp = sum(self.fp_.values())
            fn = sum(self.fn_.values())
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            b2 = beta * beta
            denom = b2 * p + r
            return (1.0 + b2) * p * r / denom if denom > 0 else 0.0
        values = [self._fbeta_for_class(c, beta) for c in classes]
        if self.average == "macro":
            return sum(values) / len(values)
        if self.average == "weighted":
            weights = [self.support_.get(c, 0.0) for c in classes]
            total = sum(weights)
            if total == 0:
                return 0.0
            return sum(v * w for v, w in zip(values, weights)) / total
        raise ValueError(f"unknown average: {self.average!r}")

    def _aggregate(self, mode: str) -> float:
        if self.average == "binary":
            p, r = self._binary_pr(self.positive)
            return p if mode == "precision" else r
        classes = self._classes()
        if not classes:
            return 0.0
        if self.average == "micro":
            tp = sum(self.tp_.values())
            fp = sum(self.fp_.values())
            fn = sum(self.fn_.values())
            if mode == "precision":
                return tp / (tp + fp) if (tp + fp) > 0 else 0.0
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
        values = []
        weights = []
        for c in classes:
            p, r = self._binary_pr(c)
            if mode == "precision":
                values.append(p)
            else:
                values.append(r)
            weights.append(self.support_.get(c, 0.0))
        if self.average == "macro":
            return sum(values) / len(values)
        if self.average == "weighted":
            total = sum(weights)
            if total == 0:
                return 0.0
            return sum(v * w for v, w in zip(values, weights)) / total
        raise ValueError(f"unknown average: {self.average!r}")

    def _merge_into(self, m: "_PRFBase", other: "_PRFBase") -> None:
        for d_name in ("tp_", "fp_", "fn_", "support_"):
            d = getattr(m, d_name)
            for k, v in getattr(self, d_name).items():
                d[k] = d.get(k, 0.0) + v
            for k, v in getattr(other, d_name).items():
                d[k] = d.get(k, 0.0) + v


class Precision(_PRFBase):
    """Fraction of predicted positives that are correct.

    Supports four averages: ``"binary"`` (uses ``positive``),
    ``"macro"``, ``"micro"``, ``"weighted"``.

    Examples:
        >>> from dense_armor.utility.metrics import Precision
        >>> m = Precision(positive=1, average="binary")
        >>> for a, b in [(1, 1), (0, 1), (1, 1), (0, 0)]:
        ...     _ = m.update(a, b)
        >>> m.get()
        0.6666666666666666
    """

    bigger_is_better = True

    def get(self) -> float:
        return self._aggregate("precision")

    def merge(self, other: Metric) -> "Precision":
        """Combine two partial states.

        Args:
            other: another :class:`Precision` with the same configuration.

        Returns:
            A new :class:`Precision` with the combined state.

        Raises:
            TypeError: if ``other`` is not a :class:`Precision`.
        """
        if not isinstance(other, Precision):
            raise TypeError(
                f"merge expects Precision, got {type(other).__name__}"
            )
        m = Precision(positive=self.positive, average=self.average)
        self._merge_into(m, other)
        m.n_ = self.n_ + other.n_
        m.n_missing_ = self.n_missing_ + other.n_missing_
        return m


class Recall(_PRFBase):
    """Fraction of true positives that are detected.

    Same four averages as :class:`Precision`.

    Examples:
        >>> from dense_armor.utility.metrics import Recall
        >>> m = Recall(positive=1, average="binary")
        >>> for a, b in [(1, 1), (0, 1), (1, 0)]:
        ...     _ = m.update(a, b)
        >>> m.get()
        0.5
    """

    bigger_is_better = True

    def get(self) -> float:
        return self._aggregate("recall")

    def merge(self, other: Metric) -> "Recall":
        """Combine two partial states.

        Args:
            other: another :class:`Recall` with the same configuration.

        Returns:
            A new :class:`Recall` with the combined state.

        Raises:
            TypeError: if ``other`` is not a :class:`Recall`.
        """
        if not isinstance(other, Recall):
            raise TypeError(f"merge expects Recall, got {type(other).__name__}")
        m = Recall(positive=self.positive, average=self.average)
        self._merge_into(m, other)
        m.n_ = self.n_ + other.n_
        m.n_missing_ = self.n_missing_ + other.n_missing_
        return m


class FBeta(_PRFBase):
    """Weighted harmonic mean of precision and recall.

    ``F_beta = (1 + beta**2) * P * R / (beta**2 * P + R)``. ``F1`` is
    the special case ``beta = 1``. The ``average`` follows the same
    conventions as :class:`Precision`.

    Args:
        beta: relative weight of recall over precision.
        positive: the label treated as positive in ``"binary"`` mode.
        average: ``"binary"``, ``"macro"``, ``"micro"`` or ``"weighted"``.

    Examples:
        >>> from dense_armor.utility.metrics import FBeta
        >>> m = FBeta(beta=1.0, positive=1)
        >>> for a, b in [(1, 1), (0, 1), (1, 1), (0, 0)]:
        ...     _ = m.update(a, b)
        >>> round(m.get(), 4)
        0.8
    """

    bigger_is_better = True

    def __init__(
        self,
        beta: float = 1.0,
        positive: Any = True,
        average: str = "binary",
    ) -> None:
        super().__init__(positive=positive, average=average)
        self.beta = beta

    def get(self) -> float:
        return self._aggregate_fbeta(self.beta)

    def merge(self, other: Metric) -> "FBeta":
        """Combine two partial states.

        Args:
            other: another :class:`FBeta` with the same ``beta`` and
                ``average``.

        Returns:
            A new :class:`FBeta` with the combined state.

        Raises:
            TypeError: if ``other`` is not an :class:`FBeta`.
        """
        if not isinstance(other, FBeta):
            raise TypeError(f"merge expects FBeta, got {type(other).__name__}")
        if self.beta != other.beta or self.average != other.average:
            raise ValueError("merge requires the same beta and average")
        m = FBeta(
            beta=self.beta, positive=self.positive, average=self.average
        )
        self._merge_into(m, other)
        m.n_ = self.n_ + other.n_
        m.n_missing_ = self.n_missing_ + other.n_missing_
        return m


class F1(FBeta):
    """F1 score: the harmonic mean of precision and recall.

    A thin subclass of :class:`FBeta` with ``beta = 1.0``.

    Examples:
        >>> from dense_armor.utility.metrics import F1
        >>> m = F1(positive=1)
        >>> for a, b in [(1, 1), (0, 1), (1, 1)]:
        ...     _ = m.update(a, b)
        >>> round(m.get(), 4)
        0.8
    """

    def __init__(self, positive: Any = True, average: str = "binary") -> None:
        super().__init__(beta=1.0, positive=positive, average=average)

    def merge(self, other: Metric) -> "F1":
        """Combine two partial states, keeping the F1 class.

        Args:
            other: another :class:`F1`.

        Returns:
            A new :class:`F1` with the combined state.

        Raises:
            TypeError: if ``other`` is not an :class:`F1`.
        """
        if not isinstance(other, F1):
            raise TypeError(
                f"merge expects F1, got {type(other).__name__}"
            )
        m = F1(positive=self.positive, average=self.average)
        self._merge_into(m, other)
        m.n_ = self.n_ + other.n_
        m.n_missing_ = self.n_missing_ + other.n_missing_
        return m


class BalancedAccuracy(Metric):
    """Mean of the per-class recall.

    For binary classification with classes ``0`` and ``1`` this is
    ``(TPR + TNR) / 2``, where ``TNR`` is the recall of the negative
    class. Balanced accuracy is the natural metric for imbalanced
    streams, since a majority-class predictor scores 0.5.

    Examples:
        >>> from dense_armor.utility.metrics import BalancedAccuracy
        >>> m = BalancedAccuracy()
        >>> for a, b in [(0, 0), (0, 1), (1, 1), (1, 1)]:
        ...     _ = m.update(a, b)
        >>> round(m.get(), 4)
        0.75
    """

    role_ = Classifier
    bigger_is_better = True

    def __init__(self) -> None:
        super().__init__()
        self.tp_: dict[Any, float] = {}
        self.pos_: dict[Any, float] = {}

    def _update(self, y_true: Any, y_pred: Any, w: float) -> None:
        self.pos_[y_true] = self.pos_.get(y_true, 0.0) + w
        if y_true == y_pred:
            self.tp_[y_true] = self.tp_.get(y_true, 0.0) + w

    def _revert(self, y_true: Any, y_pred: Any, w: float) -> None:
        self.pos_[y_true] = self.pos_.get(y_true, 0.0) - w
        if y_true == y_pred:
            self.tp_[y_true] = self.tp_.get(y_true, 0.0) - w

    def get(self) -> float:
        if not self.pos_:
            return 0.0
        recalls = [
            self.tp_.get(c, 0.0) / n for c, n in self.pos_.items() if n > 0
        ]
        return sum(recalls) / len(recalls) if recalls else 0.0

    def merge(self, other: Metric) -> "BalancedAccuracy":
        """Combine two partial states.

        Args:
            other: another :class:`BalancedAccuracy`.

        Returns:
            A new :class:`BalancedAccuracy` with the combined state.

        Raises:
            TypeError: if ``other`` is not a :class:`BalancedAccuracy`.
        """
        if not isinstance(other, BalancedAccuracy):
            raise TypeError(
                f"merge expects BalancedAccuracy, got {type(other).__name__}"
            )
        m = BalancedAccuracy()
        for k, v in self.tp_.items():
            m.tp_[k] = m.tp_.get(k, 0.0) + v
        for k, v in other.tp_.items():
            m.tp_[k] = m.tp_.get(k, 0.0) + v
        for k, v in self.pos_.items():
            m.pos_[k] = m.pos_.get(k, 0.0) + v
        for k, v in other.pos_.items():
            m.pos_[k] = m.pos_.get(k, 0.0) + v
        m.n_ = self.n_ + other.n_
        m.n_missing_ = self.n_missing_ + other.n_missing_
        return m


class CohenKappa(Metric):
    """Cohen's kappa: agreement corrected for chance.

    ``kappa = (p_o - p_e) / (1 - p_e)`` where ``p_o`` is the observed
    agreement and ``p_e`` is the agreement expected by chance under the
    marginal label frequencies. ``1.0`` is perfect, ``0.0`` is chance,
    negative is worse than chance.

    References:
        Cohen, J. (1960). Educational and Psychological Measurement
        20(1), 37-46.
    """

    role_ = Classifier
    bigger_is_better = True

    def __init__(self) -> None:
        super().__init__()
        self.cm_: dict[Any, dict] = {}
        self.total_ = 0.0
        self.row_: dict[Any, float] = {}
        self.col_: dict[Any, float] = {}

    def _update(self, y_true: Any, y_pred: Any, w: float) -> None:
        self.cm_.setdefault(y_true, {})
        self.cm_[y_true][y_pred] = self.cm_[y_true].get(y_pred, 0.0) + w
        self.row_[y_true] = self.row_.get(y_true, 0.0) + w
        self.col_[y_pred] = self.col_.get(y_pred, 0.0) + w
        self.total_ += w

    def _revert(self, y_true: Any, y_pred: Any, w: float) -> None:
        self.cm_[y_true][y_pred] -= w
        self.row_[y_true] -= w
        self.col_[y_pred] -= w
        self.total_ -= w

    def get(self) -> float:
        if self.total_ == 0:
            return 0.0
        p_o = sum(
            self.cm_[a].get(a, 0.0) for a in self.cm_
        ) / self.total_
        p_e = sum(
            self.row_.get(c, 0.0) * self.col_.get(c, 0.0)
            for c in set(self.row_) | set(self.col_)
        ) / (self.total_ * self.total_)
        if p_e == 1.0:
            return 0.0
        return (p_o - p_e) / (1.0 - p_e)

    def merge(self, other: Metric) -> "CohenKappa":
        """Combine two partial states.

        Args:
            other: another :class:`CohenKappa`.

        Returns:
            A new :class:`CohenKappa` with the combined state.

        Raises:
            TypeError: if ``other`` is not a :class:`CohenKappa`.
        """
        if not isinstance(other, CohenKappa):
            raise TypeError(
                f"merge expects CohenKappa, got {type(other).__name__}"
            )
        m = CohenKappa()
        for a, row in self.cm_.items():
            for b, v in row.items():
                m.cm_.setdefault(a, {})
                m.cm_[a][b] = m.cm_[a].get(b, 0.0) + v
        for a, row in other.cm_.items():
            for b, v in row.items():
                m.cm_.setdefault(a, {})
                m.cm_[a][b] = m.cm_[a].get(b, 0.0) + v
        for k, v in self.row_.items():
            m.row_[k] = m.row_.get(k, 0.0) + v
        for k, v in other.row_.items():
            m.row_[k] = m.row_.get(k, 0.0) + v
        for k, v in self.col_.items():
            m.col_[k] = m.col_.get(k, 0.0) + v
        for k, v in other.col_.items():
            m.col_[k] = m.col_.get(k, 0.0) + v
        m.total_ = self.total_ + other.total_
        m.n_ = self.n_ + other.n_
        m.n_missing_ = self.n_missing_ + other.n_missing_
        return m


class ConfusionMatrix(Metric):
    """Multiclass confusion matrix.

    ``get()`` returns a nested dict ``{y_true: {y_pred: count}}``, so
    this metric is the one exception to the ``get() -> float``
    convention; ``works_with`` still uses the classifier role.

    Examples:
        >>> from dense_armor.utility.metrics import ConfusionMatrix
        >>> m = ConfusionMatrix()
        >>> for a, b in [(0, 0), (0, 1), (1, 1)]:
        ...     _ = m.update(a, b)
        >>> m.get()[0][1]
        1.0
    """

    role_ = Classifier
    bigger_is_better = False

    def __init__(self) -> None:
        super().__init__()
        self.cm_: dict[Any, dict] = {}

    def _update(self, y_true: Any, y_pred: Any, w: float) -> None:
        self.cm_.setdefault(y_true, {})
        self.cm_[y_true][y_pred] = self.cm_[y_true].get(y_pred, 0.0) + w

    def _revert(self, y_true: Any, y_pred: Any, w: float) -> None:
        self.cm_[y_true][y_pred] -= w

    def get(self) -> dict:
        return {a: dict(row) for a, row in self.cm_.items()}

    def merge(self, other: Metric) -> "ConfusionMatrix":
        """Combine two partial states.

        Args:
            other: another :class:`ConfusionMatrix`.

        Returns:
            A new :class:`ConfusionMatrix` with the combined state.

        Raises:
            TypeError: if ``other`` is not a :class:`ConfusionMatrix`.
        """
        if not isinstance(other, ConfusionMatrix):
            raise TypeError(
                f"merge expects ConfusionMatrix, got {type(other).__name__}"
            )
        m = ConfusionMatrix()
        for a, row in self.cm_.items():
            for b, v in row.items():
                m.cm_.setdefault(a, {})
                m.cm_[a][b] = m.cm_[a].get(b, 0.0) + v
        for a, row in other.cm_.items():
            for b, v in row.items():
                m.cm_.setdefault(a, {})
                m.cm_[a][b] = m.cm_[a].get(b, 0.0) + v
        m.n_ = self.n_ + other.n_
        m.n_missing_ = self.n_missing_ + other.n_missing_
        return m


class LogLoss(Metric):
    """Binary log-loss (cross-entropy).

    ``-mean(y log p + (1 - y) log(1 - p))`` with ``p`` the predicted
    probability of the positive class. The predicted probability is
    clipped to ``[eps, 1 - eps]`` to avoid ``log(0)``.

    Args:
        eps: clipping floor, default ``1e-15``.

    Examples:
        >>> from dense_armor.utility.metrics import LogLoss
        >>> m = LogLoss()
        >>> for y, p in [(1, 0.9), (0, 0.1)]:
        ...     _ = m.update(y, p)
        >>> round(m.get(), 4)
        0.1054
    """

    role_ = Classifier
    bigger_is_better = False

    def __init__(self, eps: float = 1e-15) -> None:
        super().__init__()
        self.eps = eps
        self.sum_ = 0.0
        self.total_ = 0.0

    def _update(self, y_true: Any, y_pred: Any, w: float) -> None:
        p = min(max(float(y_pred), self.eps), 1.0 - self.eps)
        y = float(y_true)
        self.sum_ += w * (-(y * math.log(p) + (1.0 - y) * math.log(1.0 - p)))
        self.total_ += w

    def _revert(self, y_true: Any, y_pred: Any, w: float) -> None:
        p = min(max(float(y_pred), self.eps), 1.0 - self.eps)
        y = float(y_true)
        self.sum_ -= w * (-(y * math.log(p) + (1.0 - y) * math.log(1.0 - p)))
        self.total_ -= w

    def get(self) -> float:
        return self.sum_ / self.total_ if self.total_ > 0 else 0.0

    def merge(self, other: Metric) -> "LogLoss":
        """Combine two partial states.

        Args:
            other: another :class:`LogLoss` with the same ``eps``.

        Returns:
            A new :class:`LogLoss` with the combined state.

        Raises:
            TypeError: if ``other`` is not a :class:`LogLoss`.
        """
        if not isinstance(other, LogLoss):
            raise TypeError(
                f"merge expects LogLoss, got {type(other).__name__}"
            )
        m = LogLoss(eps=self.eps)
        m.sum_ = self.sum_ + other.sum_
        m.total_ = self.total_ + other.total_
        m.n_ = self.n_ + other.n_
        m.n_missing_ = self.n_missing_ + other.n_missing_
        return m


class BrierScore(Metric):
    """Mean squared error of predicted probabilities.

    ``mean((p - y)**2)``. The lower the better; ``0.25`` is the score of
    a constant 0.5 forecast.

    Examples:
        >>> from dense_armor.utility.metrics import BrierScore
        >>> m = BrierScore()
        >>> for y, p in [(1, 0.9), (0, 0.1)]:
        ...     _ = m.update(y, p)
        >>> round(m.get(), 4)
        0.01
    """

    role_ = Classifier
    bigger_is_better = False

    def __init__(self) -> None:
        super().__init__()
        self.sum_ = 0.0
        self.total_ = 0.0

    def _update(self, y_true: Any, y_pred: Any, w: float) -> None:
        d = float(y_pred) - float(y_true)
        self.sum_ += w * d * d
        self.total_ += w

    def _revert(self, y_true: Any, y_pred: Any, w: float) -> None:
        d = float(y_pred) - float(y_true)
        self.sum_ -= w * d * d
        self.total_ -= w

    def get(self) -> float:
        return self.sum_ / self.total_ if self.total_ > 0 else 0.0

    def merge(self, other: Metric) -> "BrierScore":
        """Combine two partial states.

        Args:
            other: another :class:`BrierScore`.

        Returns:
            A new :class:`BrierScore` with the combined state.

        Raises:
            TypeError: if ``other`` is not a :class:`BrierScore`.
        """
        if not isinstance(other, BrierScore):
            raise TypeError(
                f"merge expects BrierScore, got {type(other).__name__}"
            )
        m = BrierScore()
        m.sum_ = self.sum_ + other.sum_
        m.total_ = self.total_ + other.total_
        m.n_ = self.n_ + other.n_
        m.n_missing_ = self.n_missing_ + other.n_missing_
        return m
