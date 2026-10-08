"""Rolling ROC-AUC, exact on the window via the Mann-Whitney form.

The area under the ROC curve is the probability that a randomly chosen
positive is scored higher than a randomly chosen negative, with ties
counted as half (Mann and Whitney 1947; Bamber 1975):

    AUC = (1 / (n_pos * n_neg)) * sum_{i in pos} sum_{j in neg}
              ( 1 if s_i > s_j, 0.5 if s_i == s_j, 0 otherwise )

The estimator keeps a sliding window of the last ``window`` pairs
``(y_true, y_score)`` and recomputes the sum on demand. Cost is
``O(n_pos * n_neg)`` per query, which for a window of a few hundred
samples is negligible. The result equals ``sklearn.metrics.roc_auc_score``
on the same window, including ties.

References
----------
Mann, H. B., Whitney, D. R. (1947). On a test of whether one of two
    random variables is stochastically larger than the other. Annals
    of Mathematical Statistics 18(1), 50-60.
Bamber, D. (1975). The area above the ordinal dominance graph and the
    area below the receiver operating characteristic graph. Journal
    of Mathematical Psychology 12(4), 387-415.
"""

from collections import deque
from typing import Any

from dense_armor.roles import Classifier
from dense_armor.utility.metrics.base import Metric


def _auc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


class RollingAUC(Metric):
    """Rolling ROC-AUC, exact on the last ``window`` samples.

    Args:
        window: window size in samples (default 200).

    Examples:
        >>> from dense_armor.utility.metrics import RollingAUC
        >>> m = RollingAUC(window=4)
        >>> for y, s in [(0, 0.1), (1, 0.9), (0, 0.2), (1, 0.8)]:
        ...     _ = m.update(y, s)
        >>> m.get()
        1.0
    """

    role_ = Classifier
    bigger_is_better = True

    def __init__(self, window: int = 200) -> None:
        super().__init__()
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.window = window
        self.buf_: deque[tuple[int, float]] = deque(maxlen=window)

    def _update(self, y_true: Any, y_pred: Any, w: float) -> None:
        self.buf_.append((1 if y_true == 1 or y_true is True else 0,
                          float(y_pred)))

    def get(self) -> float:
        if len(self.buf_) < 2:
            return 0.5
        labels = [a for a, _ in self.buf_]
        scores = [b for _, b in self.buf_]
        return _auc(scores, labels)

    def merge(self, other: Metric) -> "RollingAUC":
        """Combine two rolling windows by concatenation.

        Concatenation is not the state of the union in general; the
        result keeps the most recent ``window`` pairs from the union.

        Args:
            other: another :class:`RollingAUC` with the same window size.

        Returns:
            A new :class:`RollingAUC` with the combined window.

        Raises:
            TypeError: if ``other`` is not a :class:`RollingAUC`.
            ValueError: if ``window`` differs.
        """
        if not isinstance(other, RollingAUC):
            raise TypeError(
                f"merge expects RollingAUC, got {type(other).__name__}"
            )
        if self.window != other.window:
            raise ValueError("merge requires the same window")
        m = RollingAUC(window=self.window)
        for item in list(self.buf_) + list(other.buf_):
            m.buf_.append(item)
        m.n_ = self.n_ + other.n_
        m.n_missing_ = self.n_missing_ + other.n_missing_
        return m
