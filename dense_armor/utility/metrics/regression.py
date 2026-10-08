"""Online regression metrics.

Every metric keeps only a handful of running sums and updates them in
constant time per sample. ``merge`` combines two partial states
exactly, so the same metric can be computed over sharded streams.

Scalar metrics also accept a joint vector as ``y_true`` / ``y_pred``:
the metric keeps a per-joint state alongside the aggregate one and
``get()`` returns ``{"per_joint": [...], "mean": ...}``. With
``feature="joint1"`` only that key is scored, from a dict input.

References
----------
Cook, R. D., Weisberg, S. (1982). Residuals and Influence in
    Regression. Chapman & Hall.
Good, I. J. (1952). Rational decisions. J. Royal Statistical Society B
    14(1), 107-114.
Welford, B. P. (1962). Technometrics 4(3), 419-420.
Chan, T. F., Golub, G. H., LeVeque, R. J. (1979). In Compstat.
Vovk, V., Gammerman, A., Shafer, G. (2005). Algorithmic Learning in a
    Random World. Springer.
"""

import math
from typing import Any

from dense_armor.roles import Regressor
from dense_armor.utility.metrics.base import Metric


def _is_vector(v: Any) -> bool:
    if isinstance(v, (list, tuple)):
        return len(v) > 1 and all(isinstance(x, (int, float)) for x in v)
    return False


def _pairs(
    y_true: Any, y_pred: Any, feature: str | None
) -> list[tuple[float, float]]:
    """Return the list of scalar (y_true, y_pred) pairs to feed.

    A ``feature`` selects one key from dict inputs, a vector pair is
    expanded joint by joint, anything else is a single scalar pair.
    """
    if feature is not None:
        return [(float(y_true[feature]), float(y_pred[feature]))]
    if _is_vector(y_true) and _is_vector(y_pred) and len(y_true) == len(y_pred):
        return [(float(a), float(b)) for a, b in zip(y_true, y_pred)]
    return [(float(y_true), float(y_pred))]


class _PerJoint:
    """Per-joint accumulator: sum and total per channel."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.sums = [0.0] * n
        self.totals = [0.0] * n

    def add(self, j: int, value: float, w: float) -> None:
        self.sums[j] += w * value
        self.totals[j] += w

    def sub(self, j: int, value: float, w: float) -> None:
        self.sums[j] -= w * value
        self.totals[j] -= w

    def merged(self, other: "_PerJoint") -> "_PerJoint":
        out = _PerJoint(self.n)
        for j in range(self.n):
            out.sums[j] = self.sums[j] + other.sums[j]
            out.totals[j] = self.totals[j] + other.totals[j]
        return out

    def values(self) -> list[float]:
        return [
            self.sums[j] / self.totals[j] if self.totals[j] > 0 else 0.0
            for j in range(self.n)
        ]


class MeanAbsoluteError(Metric):
    """Mean absolute error, one sample at a time.

    ``MAE = mean(|y - y_hat|)``. Robust to outliers compared to MSE and
    the natural error for a robot torque loop, where all deviations
    are equally bad.

    Args:
        feature: key of the channel to read from dict inputs. ``None``
            accepts a scalar or a joint vector.

    Examples:
        >>> from dense_armor.utility.metrics import MeanAbsoluteError
        >>> m = MeanAbsoluteError()
        >>> for a, b in [(1.0, 1.5), (2.0, 2.5), (3.0, 2.5)]:
        ...     _ = m.update(a, b)
        >>> round(m.get(), 4)
        0.5
    """

    role_ = Regressor
    bigger_is_better = False

    def __init__(self, feature: str | None = None) -> None:
        super().__init__()
        self.feature = feature
        self.sum_ = 0.0
        self.total_ = 0.0
        self.per_joint_: _PerJoint | None = None

    def _update(self, y_true: Any, y_pred: Any, w: float) -> None:
        pairs = _pairs(y_true, y_pred, self.feature)
        if len(pairs) > 1:
            if self.per_joint_ is None:
                self.per_joint_ = _PerJoint(len(pairs))
            elif self.per_joint_.n != len(pairs):
                raise ValueError(
                    f"joint count changed from {self.per_joint_.n} "
                    f"to {len(pairs)}"
                )
        for j, (a, b) in enumerate(pairs):
            v = abs(a - b)
            self.sum_ += w * v
            self.total_ += w
            if self.per_joint_ is not None:
                self.per_joint_.add(j, v, w)

    def _revert(self, y_true: Any, y_pred: Any, w: float) -> None:
        pairs = _pairs(y_true, y_pred, self.feature)
        for j, (a, b) in enumerate(pairs):
            v = abs(a - b)
            self.sum_ -= w * v
            self.total_ -= w
            if self.per_joint_ is not None:
                self.per_joint_.sub(j, v, w)

    def get(self) -> Any:
        if self.per_joint_ is None:
            return self.sum_ / self.total_ if self.total_ > 0 else 0.0
        per = self.per_joint_.values()
        mean = self.sum_ / self.total_ if self.total_ > 0 else 0.0
        return {"per_joint": per, "mean": mean}

    def merge(self, other: Metric) -> "MeanAbsoluteError":
        """Combine two partial states.

        Args:
            other: another :class:`MeanAbsoluteError`.

        Returns:
            A new :class:`MeanAbsoluteError` with the combined state.

        Raises:
            TypeError: if ``other`` is not a :class:`MeanAbsoluteError`.
        """
        if not isinstance(other, MeanAbsoluteError):
            raise TypeError(
                f"merge expects MeanAbsoluteError, got {type(other).__name__}"
            )
        m = MeanAbsoluteError(feature=self.feature)
        m.sum_ = self.sum_ + other.sum_
        m.total_ = self.total_ + other.total_
        if (
            self.per_joint_ is not None
            and other.per_joint_ is not None
            and self.per_joint_.n == other.per_joint_.n
        ):
            m.per_joint_ = self.per_joint_.merged(other.per_joint_)
        m.n_ = self.n_ + other.n_
        m.n_missing_ = self.n_missing_ + other.n_missing_
        return m


class MeanSquaredError(Metric):
    """Mean squared error, one sample at a time.

    Examples:
        >>> from dense_armor.utility.metrics import MeanSquaredError
        >>> m = MeanSquaredError()
        >>> for a, b in [(1.0, 1.5), (2.0, 2.5), (3.0, 2.5)]:
        ...     _ = m.update(a, b)
        >>> round(m.get(), 4)
        0.25
    """

    role_ = Regressor
    bigger_is_better = False

    def __init__(self, feature: str | None = None) -> None:
        super().__init__()
        self.feature = feature
        self.sum_ = 0.0
        self.total_ = 0.0
        self.per_joint_: _PerJoint | None = None

    def _update(self, y_true: Any, y_pred: Any, w: float) -> None:
        pairs = _pairs(y_true, y_pred, self.feature)
        if len(pairs) > 1:
            if self.per_joint_ is None:
                self.per_joint_ = _PerJoint(len(pairs))
            elif self.per_joint_.n != len(pairs):
                raise ValueError(
                    f"joint count changed from {self.per_joint_.n} "
                    f"to {len(pairs)}"
                )
        for j, (a, b) in enumerate(pairs):
            d = a - b
            v = d * d
            self.sum_ += w * v
            self.total_ += w
            if self.per_joint_ is not None:
                self.per_joint_.add(j, v, w)

    def _revert(self, y_true: Any, y_pred: Any, w: float) -> None:
        pairs = _pairs(y_true, y_pred, self.feature)
        for j, (a, b) in enumerate(pairs):
            d = a - b
            v = d * d
            self.sum_ -= w * v
            self.total_ -= w
            if self.per_joint_ is not None:
                self.per_joint_.sub(j, v, w)

    def get(self) -> Any:
        if self.per_joint_ is None:
            return self.sum_ / self.total_ if self.total_ > 0 else 0.0
        per = self.per_joint_.values()
        mean = self.sum_ / self.total_ if self.total_ > 0 else 0.0
        return {"per_joint": per, "mean": mean}

    def merge(self, other: Metric) -> "MeanSquaredError":
        """Combine two partial states.

        Args:
            other: another :class:`MeanSquaredError`.

        Returns:
            A new :class:`MeanSquaredError` with the combined state.

        Raises:
            TypeError: if ``other`` is not a :class:`MeanSquaredError`.
        """
        if not isinstance(other, MeanSquaredError):
            raise TypeError(
                f"merge expects MeanSquaredError, got {type(other).__name__}"
            )
        m = MeanSquaredError(feature=self.feature)
        m.sum_ = self.sum_ + other.sum_
        m.total_ = self.total_ + other.total_
        if (
            self.per_joint_ is not None
            and other.per_joint_ is not None
            and self.per_joint_.n == other.per_joint_.n
        ):
            m.per_joint_ = self.per_joint_.merged(other.per_joint_)
        m.n_ = self.n_ + other.n_
        m.n_missing_ = self.n_missing_ + other.n_missing_
        return m


class RootMeanSquaredError(MeanSquaredError):
    """Square root of the mean squared error.

    Examples:
        >>> from dense_armor.utility.metrics import RootMeanSquaredError
        >>> m = RootMeanSquaredError()
        >>> for a, b in [(1.0, 1.5), (2.0, 2.5), (3.0, 2.5)]:
        ...     _ = m.update(a, b)
        >>> round(m.get(), 4)
        0.5
    """

    def get(self) -> Any:
        base = super().get()
        if isinstance(base, dict):
            per = [math.sqrt(v) for v in base["per_joint"]]
            return {"per_joint": per, "mean": math.sqrt(base["mean"])}
        return math.sqrt(base)

    def merge(self, other: Metric) -> "RootMeanSquaredError":
        """Combine two partial states.

        Args:
            other: another :class:`RootMeanSquaredError`.

        Returns:
            A new :class:`RootMeanSquaredError` with the combined state.

        Raises:
            TypeError: if ``other`` is not a
                :class:`RootMeanSquaredError`.
        """
        if not isinstance(other, RootMeanSquaredError):
            raise TypeError(
                f"merge expects RootMeanSquaredError, "
                f"got {type(other).__name__}"
            )
        m = RootMeanSquaredError(feature=self.feature)
        m.sum_ = self.sum_ + other.sum_
        m.total_ = self.total_ + other.total_
        if (
            self.per_joint_ is not None
            and other.per_joint_ is not None
            and self.per_joint_.n == other.per_joint_.n
        ):
            m.per_joint_ = self.per_joint_.merged(other.per_joint_)
        m.n_ = self.n_ + other.n_
        m.n_missing_ = self.n_missing_ + other.n_missing_
        return m


class R2Score(Metric):
    """Coefficient of determination, Welford running mean.

    ``R^2 = 1 - SS_res / SS_tot`` with ``SS_tot`` from Welford's ``M2``
    accumulator (Welford 1962; Chan et al. 1979). ``1`` is a perfect
    fit, ``0`` is the mean-only predictor, negative is worse than the
    mean.

    Examples:
        >>> from dense_armor.utility.metrics import R2Score
        >>> m = R2Score()
        >>> for a, b in [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]:
        ...     _ = m.update(a, b)
        >>> m.get()
        1.0
    """

    role_ = Regressor
    bigger_is_better = True

    def __init__(self, feature: str | None = None) -> None:
        super().__init__()
        self.feature = feature
        self.ss_res_ = 0.0
        self.mean_ = 0.0
        self.m2_ = 0.0
        self.total_ = 0.0

    def _update(self, y_true: Any, y_pred: Any, w: float) -> None:
        pairs = _pairs(y_true, y_pred, self.feature)
        for a, b in pairs:
            self.ss_res_ += w * (a - b) ** 2
            n_new = self.total_ + w
            delta = a - self.mean_
            self.mean_ += w * delta / n_new
            self.m2_ += w * delta * (a - self.mean_)
            self.total_ = n_new

    def _revert(self, y_true: Any, y_pred: Any, w: float) -> None:
        raise NotImplementedError(
            "R2Score uses a Welford state and cannot be reverted in place"
        )

    def get(self) -> float:
        if self.total_ <= 0 or self.m2_ <= 0.0:
            return 0.0
        return 1.0 - self.ss_res_ / self.m2_

    def merge(self, other: Metric) -> "R2Score":
        """Combine two partial states.

        Uses the parallel formula for Welford's ``M2`` (Chan et al.
        1979).

        Args:
            other: another :class:`R2Score`.

        Returns:
            A new :class:`R2Score` with the combined state.

        Raises:
            TypeError: if ``other`` is not a :class:`R2Score`.
        """
        if not isinstance(other, R2Score):
            raise TypeError(
                f"merge expects R2Score, got {type(other).__name__}"
            )
        m = R2Score(feature=self.feature)
        na, nb = self.total_, other.total_
        m.ss_res_ = self.ss_res_ + other.ss_res_
        m.total_ = na + nb
        if na == 0:
            m.mean_, m.m2_ = other.mean_, other.m2_
        elif nb == 0:
            m.mean_, m.m2_ = self.mean_, self.m2_
        else:
            d = other.mean_ - self.mean_
            m.mean_ = self.mean_ + d * nb / m.total_
            m.m2_ = self.m2_ + other.m2_ + d * d * na * nb / m.total_
        m.n_ = self.n_ + other.n_
        m.n_missing_ = self.n_missing_ + other.n_missing_
        return m


class IntervalCoverage(Metric):
    """Fraction of true labels inside a prediction interval.

    Examples:
        >>> from dense_armor.utility.metrics import IntervalCoverage
        >>> m = IntervalCoverage()
        >>> for y, lo, hi in [(0.5, 0.0, 1.0), (2.0, 0.0, 1.0)]:
        ...     _ = m.update(y, (lo, hi))
        >>> m.get()
        0.5
    """

    role_ = Regressor
    bigger_is_better = True

    def __init__(self) -> None:
        super().__init__()
        self.hits_ = 0.0
        self.total_ = 0.0

    def _update(self, y_true: Any, y_pred: Any, w: float) -> None:
        lo, hi = float(y_pred[0]), float(y_pred[1])
        if lo <= float(y_true) <= hi:
            self.hits_ += w
        self.total_ += w

    def _revert(self, y_true: Any, y_pred: Any, w: float) -> None:
        lo, hi = float(y_pred[0]), float(y_pred[1])
        if lo <= float(y_true) <= hi:
            self.hits_ -= w
        self.total_ -= w

    def get(self) -> float:
        return self.hits_ / self.total_ if self.total_ > 0 else 0.0

    def merge(self, other: Metric) -> "IntervalCoverage":
        """Combine two partial states.

        Args:
            other: another :class:`IntervalCoverage`.

        Returns:
            A new :class:`IntervalCoverage` with the combined state.

        Raises:
            TypeError: if ``other`` is not an :class:`IntervalCoverage`.
        """
        if not isinstance(other, IntervalCoverage):
            raise TypeError(
                f"merge expects IntervalCoverage, got {type(other).__name__}"
            )
        m = IntervalCoverage()
        m.hits_ = self.hits_ + other.hits_
        m.total_ = self.total_ + other.total_
        m.n_ = self.n_ + other.n_
        m.n_missing_ = self.n_missing_ + other.n_missing_
        return m


class MeanIntervalWidth(Metric):
    """Mean width of a prediction interval.

    Examples:
        >>> from dense_armor.utility.metrics import MeanIntervalWidth
        >>> m = MeanIntervalWidth()
        >>> for y, lo, hi in [(0.5, 0.0, 1.0), (1.0, 0.0, 2.0)]:
        ...     _ = m.update(y, (lo, hi))
        >>> m.get()
        1.5
    """

    role_ = Regressor
    bigger_is_better = False

    def __init__(self) -> None:
        super().__init__()
        self.sum_ = 0.0
        self.total_ = 0.0

    def _update(self, y_true: Any, y_pred: Any, w: float) -> None:
        lo, hi = float(y_pred[0]), float(y_pred[1])
        self.sum_ += w * (hi - lo)
        self.total_ += w

    def _revert(self, y_true: Any, y_pred: Any, w: float) -> None:
        lo, hi = float(y_pred[0]), float(y_pred[1])
        self.sum_ -= w * (hi - lo)
        self.total_ -= w

    def get(self) -> float:
        return self.sum_ / self.total_ if self.total_ > 0 else 0.0

    def merge(self, other: Metric) -> "MeanIntervalWidth":
        """Combine two partial states.

        Args:
            other: another :class:`MeanIntervalWidth`.

        Returns:
            A new :class:`MeanIntervalWidth` with the combined state.

        Raises:
            TypeError: if ``other`` is not a :class:`MeanIntervalWidth`.
        """
        if not isinstance(other, MeanIntervalWidth):
            raise TypeError(
                f"merge expects MeanIntervalWidth, got {type(other).__name__}"
            )
        m = MeanIntervalWidth()
        m.sum_ = self.sum_ + other.sum_
        m.total_ = self.total_ + other.total_
        m.n_ = self.n_ + other.n_
        m.n_missing_ = self.n_missing_ + other.n_missing_
        return m


class GaussianNLL(Metric):
    """Gaussian negative log-likelihood.

    Examples:
        >>> from dense_armor.utility.metrics import GaussianNLL
        >>> m = GaussianNLL()
        >>> for y, mu, v in [(0.0, 0.0, 1.0), (0.0, 0.0, 1.0)]:
        ...     _ = m.update(y, (mu, v))
        >>> round(m.get(), 4)
        0.9189
    """

    role_ = Regressor
    bigger_is_better = False

    def __init__(self, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = eps
        self.sum_ = 0.0
        self.total_ = 0.0

    def _mu_v(self, y_pred: Any) -> tuple[float, float]:
        if hasattr(y_pred, "mean") and hasattr(y_pred, "var"):
            return float(y_pred.mean), float(y_pred.var)
        return float(y_pred[0]), float(y_pred[1])

    def _nll(self, y: float, mu: float, v: float) -> float:
        v = max(v, self.eps)
        return 0.5 * math.log(2.0 * math.pi * v) + 0.5 * (y - mu) ** 2 / v

    def _update(self, y_true: Any, y_pred: Any, w: float) -> None:
        mu, v = self._mu_v(y_pred)
        self.sum_ += w * self._nll(float(y_true), mu, v)
        self.total_ += w

    def _revert(self, y_true: Any, y_pred: Any, w: float) -> None:
        mu, v = self._mu_v(y_pred)
        self.sum_ -= w * self._nll(float(y_true), mu, v)
        self.total_ -= w

    def get(self) -> float:
        return self.sum_ / self.total_ if self.total_ > 0 else 0.0

    def merge(self, other: Metric) -> "GaussianNLL":
        """Combine two partial states.

        Args:
            other: another :class:`GaussianNLL` with the same ``eps``.

        Returns:
            A new :class:`GaussianNLL` with the combined state.

        Raises:
            TypeError: if ``other`` is not a :class:`GaussianNLL`.
        """
        if not isinstance(other, GaussianNLL):
            raise TypeError(
                f"merge expects GaussianNLL, got {type(other).__name__}"
            )
        m = GaussianNLL(eps=self.eps)
        m.sum_ = self.sum_ + other.sum_
        m.total_ = self.total_ + other.total_
        m.n_ = self.n_ + other.n_
        m.n_missing_ = self.n_missing_ + other.n_missing_
        return m
