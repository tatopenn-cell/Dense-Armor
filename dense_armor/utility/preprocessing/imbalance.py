"""Queue-based resampling for online class imbalance.

The algorithm keeps two fixed-length queues, one per class, and at each
time step trains the wrapped classifier on the union of the two. The
per-class queue bounds the effective class ratio: after warm-up the
union is balanced (``L`` positives and ``L`` negatives), even if the
raw stream is 1 % positive. A queue that is not yet full simply does
not contribute, so the first samples are learned as they arrive.

Algorithm 1 of Malialis, Panayiotou, Polycarpou (2018), p. 5, maps to
this class as follows:

1. input ``L``: constructor argument ``queue_size``.
2. loop over time steps: ``learn_one`` is called once per step.
3. receive example ``x^t``: outside ``learn_one`` (the caller has it).
4. predict class ``y_hat^t``: outside ``learn_one``
   (``predict_one`` / ``predict_proba_one``, delegated to the wrapped
   classifier).
5. receive true label ``y^t``: the second argument of ``learn_one``.
6. ``z^t = (x^t, y^t)``: built inside ``learn_one``.
7. if ``y^t == 0``: append to the negative queue.
8. else: append to the positive queue.
9. ``q^t = q_p^t union q_n^t``: the union of the two queues.
10. calculate cost on ``q^t``: delegated to the wrapped classifier,
    which trains on each element of the union.
11. update classifier: ``learn_one`` on the wrapped classifier, once
    per element of the union.

The wrapper adds no state beyond the two queues, so memory is ``O(L)``
per class, fixed by ``queue_size``.

References
----------
Malialis, K., Panayiotou, C., Polycarpou, M. M. (2018). Queue-based
    resampling for online class imbalance learning. ICANN, section 3
    and Algorithm 1 on p. 5.
"""
from collections import deque
from typing import Any

from dense_armor.roles import Classifier
from dense_armor.roles._util import call_with_t


def _as_dict(x: Any) -> dict:
    if hasattr(x, "to_dict"):
        return dict(x.to_dict())
    return dict(x)


class QueueResampler(Classifier):
    """Resampling wrapper for imbalanced online streams.

    ``learn_one`` appends the labelled sample to the queue of its
    class, then trains the wrapped classifier once per element of the
    union of the two queues. ``predict_one`` and ``predict_proba_one``
    delegate straight to the wrapped classifier.

    Args:
        classifier: the wrapped classifier.
        queue_size: length ``L`` of each per-class queue.
        positive: label of the positive class.

    Examples:
        >>> from dense_armor.utility.preprocessing.imbalance import QueueResampler
        >>> from dense_armor.roles import Classifier
        >>> class Counter(Classifier):
        ...     def __init__(self):
        ...         self.n_seen_ = 0
        ...     def learn_one(self, x, y, t=None):
        ...         self.n_seen_ += 1
        ...         return self
        ...     def predict_one(self, x, t=None):
        ...         return 0
        >>> qr = QueueResampler(Counter(), queue_size=3)
        >>> for i in range(10):
        ...     _ = qr.learn_one({"x": i}, y=1 if i == 4 else 0)
        >>> qr.classifier.n_seen_ > 10
        True

    References:
        Malialis, K., Panayiotou, C., Polycarpou, M. M. (2018). ICANN,
        Algorithm 1 on p. 5.
    """

    budget_s = 1e-3
    memory_class = "O(window)"

    def __init__(
        self,
        classifier: Any,
        queue_size: int = 25,
        positive: Any = 1,
    ) -> None:
        self.classifier = classifier
        self.queue_size = queue_size
        self.positive = positive
        self.q_positive_: deque = deque(maxlen=queue_size)
        self.q_negative_: deque = deque(maxlen=queue_size)
        self.n_seen_ = 0

    def learn_one(
        self, x: Any, y: Any, t: float | None = None
    ) -> "QueueResampler":
        """Append the sample to its queue and train on the union."""
        self._time_step(t)
        z = (_as_dict(x), y)
        if y == self.positive:
            self.q_positive_.append(z)
        else:
            self.q_negative_.append(z)
        for xi, yi in list(self.q_positive_) + list(self.q_negative_):
            call_with_t(self.classifier.learn_one, xi, yi, t=t)
        self.n_seen_ += 1
        return self

    def predict_one(self, x: Any, t: float | None = None) -> Any:
        """Delegate to the wrapped classifier."""
        return call_with_t(
            self.classifier.predict_one, _as_dict(x), t=t
        )

    def predict_proba_one(self, x: Any, t: float | None = None) -> dict:
        """Delegate to the wrapped classifier."""
        return call_with_t(
            self.classifier.predict_proba_one, _as_dict(x), t=t
        )
